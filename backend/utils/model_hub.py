#!/usr/bin/env python3
# utils/model_hub.py — SHARED, STATELESS inference layer for multi-camera edge.
# ================================================================
# ONE set of models (person + PPE + pose) loaded once and shared by every camera,
# instead of the old design that built a full PPEEngine (its own model copies) per
# camera → VRAM grew ×N. Here VRAM is CONSTANT regardless of camera count; the only
# variable is per-forward activation, which batching amortises.
#
# The Hub owns NO camera state: no track ids, no temporal confirm, no zones/roles.
# It only turns frames → detections. Track ids come from a per-camera ByteTracker
# in CameraEngine; everything stateful lives there. Methods use `.predict()`
# (stateless → batchable) rather than `.track()` (stateful, per-stream).
# ================================================================
from __future__ import annotations

from typing import Optional

import config as cfg
from utils.detect_track import Detector, PersonDetector, _to_norm_dicts


class ModelHub:
    def __init__(self, device: Optional[str] = None,
                 want_ppe: bool = True, want_pose: bool = True):
        # PersonDetector loads the COCO person model (+ resolves the person class).
        # We use its .model.predict() (NOT .track()) so frames from many cameras can
        # be batched in one forward.
        self.person = PersonDetector(device=device)
        self.device = self.person.device

        # PPE fine-tune is optional: absent on a fresh clone → PPE just unavailable
        # (person/zone/fall still work), matching PPEEngine's tolerant behaviour.
        self.ppe: Optional[Detector] = None
        if want_ppe:
            try:
                self.ppe = Detector(device=device)
            except FileNotFoundError as e:
                print(f"[ModelHub] ⚠️ PPE model unavailable → PPE off: {e}")

        # Pose (visibility gate + keypoint-anchored association) — one shared model.
        self.pose_model = None
        if want_pose:
            try:
                from ultralytics import YOLO
                self.pose_model = YOLO(str(cfg.PPE_POSE_MODEL))
            except Exception as e:
                print(f"[ModelHub] ⚠️ pose unavailable: {e}")

        print(f"[ModelHub] 🧠 shared models ready (device={self.device}, "
              f"ppe={'on' if self.ppe else 'off'}, pose={'on' if self.pose_model else 'off'})")

    # ── batched person detection (no tracking) ────────────────
    def detect_persons(self, frames: list) -> list[list[dict]]:
        """One batched forward → per-frame list of person dicts (NO track_id).
        Runs at the low PERSON_CONF floor so the per-camera ByteTracker gets the
        low-score boxes it needs to hold ids through a confidence dip."""
        if not frames:
            return []
        conf  = getattr(cfg, "PERSON_CONF", 0.10)
        imgsz = getattr(cfg, "PERSON_IMGSZ", 960)
        results = self.person.model.predict(
            frames, device=self.device, imgsz=imgsz, conf=conf,
            iou=getattr(cfg, "INFERENCE_IOU", 0.45),
            classes=[self.person._person_cls], verbose=False)
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append([])
                continue
            names = ["person"] * len(r.boxes)
            out.append(_to_norm_dicts(r.boxes.xyxy.tolist(), names,
                                      r.boxes.conf.tolist(), None, None))
        return out

    # ── batched PPE item detection ────────────────────────────
    def detect_items(self, frames: list) -> list[list[dict]]:
        """One batched forward → per-frame list of PPE item dicts (person class
        dropped; persons come from detect_persons). Empty lists when PPE is off."""
        if self.ppe is None or not frames:
            return [[] for _ in frames]
        conf  = getattr(cfg, "PPE_TRACK_CONF", 0.10)
        imgsz = getattr(cfg, "PPE_IMGSZ", 960)
        results = self.ppe.model.predict(
            frames, device=self.device, imgsz=imgsz, conf=conf,
            iou=getattr(cfg, "INFERENCE_IOU", 0.45), verbose=False)
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append([])
                continue
            names = [self.ppe.names[int(c)] for c in r.boxes.cls.int().tolist()]
            dets = _to_norm_dicts(r.boxes.xyxy.tolist(), names,
                                  r.boxes.conf.tolist(), None, None)
            out.append([d for d in dets if not d["is_person"]])
        return out

    # ── batched pose (raw keypoints for CameraEngine to match) ─
    def pose(self, frames: list) -> list:
        """One batched pose forward. Returns per-frame {boxes, xy, conf} arrays (or
        None) — CameraEngine matches these to its persons by IoU (that matching is
        cheap CPU work and stays per-camera)."""
        import numpy as np
        if self.pose_model is None or not frames:
            return [None] * len(frames)
        try:
            results = self.pose_model.predict(
                frames, imgsz=int(getattr(cfg, "PPE_POSE_IMGSZ", 640)),
                device=self.device, conf=0.25, verbose=False)
        except Exception as e:
            print(f"[ModelHub] pose pass failed: {e}")
            return [None] * len(frames)
        out = []
        for r in results:
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                out.append(None)
                continue
            xy = r.keypoints.xy.cpu().numpy()
            cf = (r.keypoints.conf.cpu().numpy()
                  if r.keypoints.conf is not None else np.ones(xy.shape[:2], np.float32))
            out.append({"boxes": r.boxes.xyxy.cpu().numpy(), "xy": xy, "conf": cf})
        return out

    # ── crop "zoom" pass (stateless, bounded) ─────────────────
    def detect_items_crops(self, frame, persons: list) -> list[dict]:
        """Re-run the PPE model on padded person crops (small/distant workers) for
        ONE frame. Used by the single-camera path; multi-camera should prefer
        `detect_items_crops_multi` so every camera's crops share one forward."""
        if self.ppe is None:
            return []
        return self.ppe.detect_items_crops(frame, persons)

    def detect_items_crops_multi(self, jobs: list) -> list[list[dict]]:
        """Batched crop pass across cameras. `jobs` = [(frame, persons), ...];
        returns one item list per job.

        This was the last per-camera forward in the shared-model design: everything
        else (persons, items, pose) already batched across cameras, while the crop
        stage still fired once per camera and cost ~30% of a 4-camera tick. Now it
        batches like the rest, and the cost stops scaling with camera count."""
        if self.ppe is None or not jobs:
            return [[] for _ in jobs]
        return self.ppe.detect_items_crops_multi(jobs)
