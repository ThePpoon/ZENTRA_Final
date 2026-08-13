"""
pipeline/scheduler.py — ZENTRA shared-model Inference Scheduler (edge multi-camera)

ONE worker thread drives inference for ALL cameras against a single shared
`ModelHub`. Each tick it grabs the latest frame from every active camera, runs
ONE batched forward per model, and hands each camera's slice to that camera's
`CameraEngine.step()` (which owns its ByteTracker + confirmers + zones).

Why one synchronous worker: the GPU is serial, so multiple inference threads only
fight over CUDA. One worker that batches across cameras is the efficient fit and
makes result routing trivially correct (the worker built the batch, so it knows
which slice belongs to which camera — no async race, no stale-result mixups).

Graceful degradation is automatic: more cameras → each tick's batch is larger and
takes longer → per-camera inference FPS drops smoothly. Nothing crashes; the
per-camera display keeps running at capture FPS (decoupled).

Requirements gate: PPE-item and pose forwards are run ONLY for cameras whose role
needs them, so a person-only camera never pays for the PPE/pose models.
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class _Cam:
    __slots__ = ("engine", "get_frame", "on_result")

    def __init__(self, engine, get_frame: Callable, on_result: Callable):
        self.engine    = engine        # CameraEngine (PPEEngine with hub=...)
        self.get_frame = get_frame     # () -> latest BGR frame | None
        self.on_result = on_result     # (recs, events) -> None


class InferenceScheduler(threading.Thread):
    def __init__(self, hub, fps: int = 15):
        super().__init__(daemon=True, name="InferenceScheduler")
        self._hub  = hub
        self._cams: dict[str, _Cam] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._interval = 1.0 / max(1, fps)
        # Light telemetry (for benchmark / status).
        self.last_batch = 0
        self.last_tick_ms = 0.0

    # ── registration ─────────────────────────────────────────
    def register(self, camera_id: str, engine, get_frame, on_result):
        with self._lock:
            self._cams[camera_id] = _Cam(engine, get_frame, on_result)

    def unregister(self, camera_id: str):
        with self._lock:
            self._cams.pop(camera_id, None)

    def active_ids(self):
        with self._lock:
            return list(self._cams.keys())

    def stop(self):
        self._stop.set()

    # ── main loop ─────────────────────────────────────────────
    def run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            self._tick()
            self.last_tick_ms = (time.monotonic() - t0) * 1000.0
            dt = self._interval - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)
            else:
                time.sleep(0.001)   # yield; batch already over budget (graceful)

    def _tick(self):
        with self._lock:
            cams = list(self._cams.items())
        if not cams:
            return

        # Gather the newest frame from each camera that has one.
        ctxs = []   # (camera_id, _Cam, frame)
        for cid, cam in cams:
            try:
                f = cam.get_frame()
            except Exception:
                f = None
            if f is not None:
                ctxs.append((cid, cam, f))
        if not ctxs:
            return
        frames = [f for _, _, f in ctxs]

        # 1) person detection — batched for ALL cameras (everyone needs persons).
        try:
            persons = self._hub.detect_persons(frames)
        except Exception as e:
            print(f"[Scheduler] person batch error: {e}")
            return

        # 2) PPE items — batched ONLY for cameras whose role needs PPE (gate).
        ppe_idx  = [i for i, (_, cam, _) in enumerate(ctxs) if cam.engine.ppe_enabled]
        items_by_i: dict[int, list] = {}
        if ppe_idx:
            try:
                res = self._hub.detect_items([frames[i] for i in ppe_idx])
                for k, i in enumerate(ppe_idx):
                    items_by_i[i] = res[k]
            except Exception as e:
                print(f"[Scheduler] ppe batch error: {e}")

        # 3) pose — batched ONLY for cameras that use the visibility gate / kp-assign.
        pose_idx = [i for i, (_, cam, _) in enumerate(ctxs)
                    if getattr(cam.engine, "_vis_gate", False)
                    or getattr(cam.engine, "_kp_assign", False)]
        pose_by_i: dict[int, object] = {}
        if pose_idx:
            try:
                res = self._hub.pose([frames[i] for i in pose_idx])
                for k, i in enumerate(pose_idx):
                    pose_by_i[i] = res[k]
            except Exception as e:
                print(f"[Scheduler] pose batch error: {e}")

        # 4) per-camera stateful PREPARE — track ids, poses, and which people need
        #    the crop "zoom" pass. No inference here.
        prepared: list = []          # (i, cid, cam, frame, state, crop_persons)
        for i, (cid, cam, f) in enumerate(ctxs):
            try:
                state, crops = cam.engine.step_prepare(
                    f, persons[i], items_by_i.get(i, []), pose_by_i.get(i))
                prepared.append((i, cid, cam, f, state, crops))
            except Exception as e:
                print(f"[Scheduler] prepare error ({cid}): {e}")

        # 5) crop "zoom" — ONE batched forward for EVERY camera's crops. This was
        #    the last per-camera forward left in the shared-model design; it cost
        #    ~30% of a 4-camera tick purely in per-call overhead (four launches of
        #    a handful of small crops each). Skipped entirely when nobody needs it.
        crop_by_i: dict[int, list] = {}
        jobs = [(f, crops) for (_, _, _, f, _, crops) in prepared if crops]
        if jobs:
            try:
                res = self._hub.detect_items_crops_multi(jobs)
                k = 0
                for (i, _cid, _cam, _f, _st, crops) in prepared:
                    if crops:
                        crop_by_i[i] = res[k]
                        k += 1
            except Exception as e:
                print(f"[Scheduler] crop batch error: {e}")

        # 6) per-camera FINISH — each camera decides for itself.
        finished = []
        for (i, cid, cam, f, state, _crops) in prepared:
            try:
                recs, events = cam.engine.step_finish(state, crop_by_i.get(i, []))
                finished.append((cid, cam, recs, events))
            except Exception as e:
                print(f"[Scheduler] step error ({cid}): {e}")

        # 7) publish (routing is by construction).
        for (cid, cam, recs, events) in finished:
            try:
                cam.on_result(recs, events)
            except Exception as e:
                print(f"[Scheduler] publish error ({cid}): {e}")

        self.last_batch = len(ctxs)
