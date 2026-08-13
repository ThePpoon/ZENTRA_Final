#!/usr/bin/env python3
# utils/detect_track.py — Detection + ByteTrack tracking in one call, on our own
# ppe_finetuned.pt via ultralytics (native Kalman ByteTrack, MPS, no Docker).
# ================================================================
# Returns detections in a normalized dict form the rest of the engine consumes:
#   {track_id:int|None, cls:str, conf:float, x1,y1,x2,y2 (pixels),
#    cx,cy (center), foot (bottom-center) }
# track_id is present only for tracked classes (persons here); PPE boxes may be
# untracked (id=None) and get associated to person tracks downstream.
#
# IMPORTANT: model.track(persist=True) is STATEFUL per stream and NOT thread-safe.
# Use ONE Detector per pipeline loop; call .reset() on stream (re)start so track
# IDs don't leak across sessions.
# ================================================================
from __future__ import annotations
from pathlib import Path
from typing import Optional

import config as cfg


def _device() -> str:
    """Auto-pick: env override → MPS (Mac) → CUDA (Docker GPU) → CPU (Docker CPU)."""
    d = getattr(cfg, "PPE_INFER_DEVICE", None)
    if d:
        return d
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _to_norm_dicts(xyxy, class_names, confs, track_ids, person_idx_or_name) -> list[dict]:
    """Shared converter → the normalized detection dicts the engine consumes."""
    out = []
    n = len(xyxy)
    if track_ids is None:
        track_ids = [None] * n
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        name = class_names[i]
        tid = track_ids[i]
        out.append({
            "track_id": int(tid) if tid is not None else None,
            "cls": name,
            "taxo": cfg.ppe_taxo_index(name),
            "conf": float(confs[i]) if confs is not None else 0.0,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "foot": ((x1 + x2) / 2, y2),
            "is_person": cfg.ppe_taxo_index(name) == cfg.PPE_TAXONOMY.index("person"),
        })
    return out


class Detector:
    """Thin wrapper over ultralytics YOLO.track() for the PPE model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        from ultralytics import YOLO
        # PDPA / offline: disable ultralytics usage telemetry (phone-home).
        try:
            from ultralytics import settings as _s
            if _s.get("sync", True):
                _s.update({"sync": False})
        except Exception:
            pass
        self.model_path = model_path or cfg.PPE_LOCAL_MODEL
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"PPE model not found: {self.model_path}")
        self.device = device or _device()
        self.model = YOLO(self.model_path)
        self.names = self.model.names            # id -> class string
        self._person_idx = self._resolve_person_idx()

    def _resolve_person_idx(self) -> int:
        for i, n in self.names.items():
            if cfg.ppe_taxo_index(n) == cfg.PPE_TAXONOMY.index("person"):
                return i
        return -1

    def reset(self):
        """Clear ByteTrack state — call on pipeline.start() / new stream.
        Dropping the whole predictor makes ultralytics lazily rebuild it (and a
        fresh tracker with track IDs starting from 1) on the next .track() call.
        (Do NOT set predictor.trackers=None — the postprocess callback indexes
        trackers[0] and would crash.)"""
        try:
            self.model.predictor = None
        except Exception:
            pass

    def track(self, frame, conf: Optional[float] = None,
              imgsz: Optional[int] = None) -> list[dict]:
        """Run detect+track on one BGR frame → list of detection dicts.

        Runs at the LOW tracking floor (PPE_TRACK_CONF), NOT the PPE slider: that
        feeds ByteTrack the low-score boxes it needs to keep a person's id stable
        through a confidence dip. PPE precision is restored downstream by filtering
        items at INFERENCE_CONFIDENCE in ppe_association. Uses the ZENTRA-tuned
        tracker yaml (PPE_TRACKER_CONFIG) so track_buffer/new_track_thresh actually
        apply — the old hard-coded 'bytetrack.yaml' ignored every config knob."""
        conf = getattr(cfg, "PPE_TRACK_CONF", 0.10) if conf is None else conf
        imgsz = getattr(cfg, "PPE_IMGSZ", 960) if imgsz is None else imgsz
        tracker = getattr(cfg, "PPE_TRACKER_CONFIG", "bytetrack.yaml")
        r = self.model.track(
            frame, persist=True, tracker=tracker,
            device=self.device, imgsz=imgsz, conf=conf,
            iou=getattr(cfg, "INFERENCE_IOU", 0.45), verbose=False,
        )[0]
        out: list[dict] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        ids = r.boxes.id
        ids = ids.int().tolist() if ids is not None else [None] * len(r.boxes)
        xyxy = r.boxes.xyxy.tolist()
        clss = r.boxes.cls.int().tolist()
        confs = r.boxes.conf.tolist()
        names = [self.names[int(c)] for c in clss]
        return _to_norm_dicts(xyxy, names, confs, ids, None)

    def detect_items(self, frame, conf: Optional[float] = None,
                     imgsz: Optional[int] = None) -> list[dict]:
        """PPE ITEMS only — no person boxes, no tracking.

        Person boxes + track ids now come from PersonDetector, so this model's
        weak `person` class is discarded and the (stateful) tracker is skipped
        entirely: items are matched to people by containment in ppe_association,
        which never needed item track ids. Runs at a low floor so the per-class
        thresholds in associate() (PPE_CLASS_CONF, e.g. glasses/gloves 0.25) can
        actually see those boxes."""
        conf = getattr(cfg, "PPE_TRACK_CONF", 0.10) if conf is None else conf
        imgsz = getattr(cfg, "PPE_IMGSZ", 960) if imgsz is None else imgsz
        r = self.model.predict(
            frame, device=self.device, imgsz=imgsz, conf=conf,
            iou=getattr(cfg, "INFERENCE_IOU", 0.45), verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        clss = r.boxes.cls.int().tolist()
        names = [self.names[int(c)] for c in clss]
        dets = _to_norm_dicts(r.boxes.xyxy.tolist(), names,
                              r.boxes.conf.tolist(), None, None)
        return [d for d in dets if not d["is_person"]]

    @staticmethod
    def _build_crops(frame, persons: list[dict], pad_frac: float):
        """Padded person crops + each crop's (x, y) offset within `frame`."""
        H, W = frame.shape[:2]
        crops, offsets = [], []
        for p in persons:
            pw = p["x2"] - p["x1"]; ph = p["y2"] - p["y1"]
            px = pad_frac * pw; py = pad_frac * ph
            x1 = max(0, int(p["x1"] - px)); y1 = max(0, int(p["y1"] - py))
            x2 = min(W, int(p["x2"] + px)); y2 = min(H, int(p["y2"] + py))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crops.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
        return crops, offsets

    def _crop_result_to_dets(self, r, ox: int, oy: int) -> list[dict]:
        """One crop's YOLO result → item dets in FULL-FRAME pixel coords."""
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.tolist()
        names = [self.names[int(c)] for c in r.boxes.cls.int().tolist()]
        out = []
        for d in _to_norm_dicts(xyxy, names, r.boxes.conf.tolist(), None, None):
            if d["is_person"]:
                continue
            # crop-space → frame-space
            d["x1"] += ox; d["x2"] += ox; d["y1"] += oy; d["y2"] += oy
            d["cx"] += ox; d["cy"] += oy
            d["foot"] = (d["foot"][0] + ox, d["foot"][1] + oy)
            out.append(d)
        return out

    def detect_items_crops(self, frame, persons: list[dict],
                           imgsz: Optional[int] = None,
                           conf: Optional[float] = None,
                           pad_frac: Optional[float] = None) -> list[dict]:
        """Two-stage "zoom": crop each person, run the PPE model on the crops as ONE
        batch, and return item dets in FULL-FRAME pixel coords (offset back).

        Small/distant workers give the full-frame pass only a handful of pixels per
        glove/glass; a crop re-runs the SAME weights at native scale on that person
        and recovers those items. Batched so N crops cost ~one padded forward, not N
        (measured: 5 crops @320 = 35ms batched vs 65ms sequential). No tracking, no
        person boxes — items are re-associated by containment downstream, exactly
        like detect_items()."""
        if not persons:
            return []
        conf = getattr(cfg, "PPE_TRACK_CONF", 0.10) if conf is None else conf
        imgsz = getattr(cfg, "PPE_CROP_IMGSZ", 320) if imgsz is None else imgsz
        pad_frac = getattr(cfg, "PPE_CROP_PAD_FRAC", 0.08) if pad_frac is None else pad_frac
        crops, offsets = self._build_crops(frame, persons, pad_frac)
        if not crops:
            return []
        results = self.model.predict(
            crops, device=self.device, imgsz=imgsz, conf=conf,
            iou=getattr(cfg, "INFERENCE_IOU", 0.45), verbose=False,
        )
        out: list[dict] = []
        for r, (ox, oy) in zip(results, offsets):
            out.extend(self._crop_result_to_dets(r, ox, oy))
        return out

    def detect_items_crops_multi(self, jobs: list,
                                 imgsz: Optional[int] = None,
                                 conf: Optional[float] = None,
                                 pad_frac: Optional[float] = None) -> list[list[dict]]:
        """Same "zoom" pass, but for SEVERAL cameras in ONE forward.

        `jobs` = [(frame, persons), ...] one entry per camera; returns one item
        list per job, in the same order.

        Why: the crop pass is the only stage that was still per-camera, so a
        4-camera tick fired four separate forwards of ~1-6 crops each and paid the
        per-call launch/pre/post overhead four times. Measured on a 4-camera tick:
        229 ms with the per-camera crop pass vs 161 ms without it — the crop stage
        alone was ~30% of the tick, far more than its few dozen small crops should
        cost. Batching every camera's crops into one forward amortises that the
        same way `detect_persons` / `detect_items` already do.

        Routing is by construction: this function built the batch, so it knows
        which slice belongs to which camera — no index guessing (architecture
        review §13)."""
        conf = getattr(cfg, "PPE_TRACK_CONF", 0.10) if conf is None else conf
        imgsz = getattr(cfg, "PPE_CROP_IMGSZ", 320) if imgsz is None else imgsz
        pad_frac = getattr(cfg, "PPE_CROP_PAD_FRAC", 0.08) if pad_frac is None else pad_frac

        out: list[list[dict]] = [[] for _ in jobs]
        crops, offsets, owner = [], [], []
        for j, (frame, persons) in enumerate(jobs):
            if frame is None or not persons:
                continue
            c, o = self._build_crops(frame, persons, pad_frac)
            crops.extend(c); offsets.extend(o); owner.extend([j] * len(c))
        if not crops:
            return out
        results = self.model.predict(
            crops, device=self.device, imgsz=imgsz, conf=conf,
            iou=getattr(cfg, "INFERENCE_IOU", 0.45), verbose=False,
        )
        for r, (ox, oy), j in zip(results, offsets, owner):
            out[j].extend(self._crop_result_to_dets(r, ox, oy))
        return out


class PersonDetector:
    """COCO-pretrained YOLO used ONLY to find + track people.

    The PPE fine-tune's `person` class is a byproduct (recall ~0.63 on our own
    val) and was the real reason crowds went undetected. A COCO model is trained
    on far denser people-scenes, so it recovers small / distant / occluded
    workers. It also OWNS the tracker now, so ByteTrack ids follow the strong
    detector instead of the weak one. Emits the same normalized dicts as
    Detector (is_person=True, with track_id), so the engine is unchanged."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        from ultralytics import YOLO
        self.model_path = model_path or cfg.PERSON_MODEL
        # models/ is gitignored, so a fresh clone / slim image won't have the
        # vendored weights. Fall back to the bare name → ultralytics fetches it
        # once. (Vendor the .pt for a fully offline factory box.)
        if not Path(self.model_path).exists():
            fallback = Path(self.model_path).name or "yolo11s.pt"
            print(f"[PersonDetector] {self.model_path} missing → downloading {fallback}")
            self.model_path = fallback
        self.device = device or _device()
        self.model = YOLO(self.model_path)
        # COCO: class 0 is "person". Restrict the head so no car/dog boxes leak out.
        self._person_cls = next(
            (i for i, n in self.model.names.items() if str(n).lower() == "person"), 0)

    def reset(self):
        """Clear ByteTrack state — call on stream (re)start so ids restart at 1."""
        try:
            self.model.predictor = None
        except Exception:
            pass

    def track(self, frame, conf: Optional[float] = None,
              imgsz: Optional[int] = None) -> list[dict]:
        conf = getattr(cfg, "PERSON_CONF", 0.10) if conf is None else conf
        imgsz = getattr(cfg, "PERSON_IMGSZ", 960) if imgsz is None else imgsz
        tracker = getattr(cfg, "PPE_TRACKER_CONFIG", "bytetrack.yaml")
        r = self.model.track(
            frame, persist=True, tracker=tracker, device=self.device,
            imgsz=imgsz, conf=conf, iou=getattr(cfg, "INFERENCE_IOU", 0.45),
            classes=[self._person_cls], verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        ids = r.boxes.id
        ids = ids.int().tolist() if ids is not None else [None] * len(r.boxes)
        names = ["person"] * len(r.boxes)   # forced by classes=[person]
        return _to_norm_dicts(r.boxes.xyxy.tolist(), names,
                              r.boxes.conf.tolist(), ids, None)


# NOTE: The old ServerDetector / make_detector (Roboflow inference server on :9001
# via inference_sdk + supervision ByteTrack) was removed. The deployed engine runs
# the models in-process through Detector / PersonDetector above (ultralytics, native
# ByteTrack, no Docker inference server) — see README. Nothing listens on :9001.
