"""
pipeline/manager.py — ZENTRA Multi-Camera Manager

Runs several `Pipeline` instances at once, one per camera_id, each with its own
`FrameBroadcaster` that tags frames with that id. The UI's CCTV/NVR grid routes
each frame to its own <img> tile by camera_id, so N cameras share the ONE
WebSocket without their frames overwriting each other.

Why a manager instead of a bare singleton: the app was single-camera (one global
`Pipeline`). Feedback asked for >1 camera + a Hikvision-style multi-view panel.
Each camera keeps its OWN Pipeline (and therefore its own ByteTrack state, temporal
confirmers, and zones) because `model.track(persist=True)` is stateful per stream
and must never be shared across cameras.

Alerts/status callbacks are widened with a `camera_id` first argument so every
event can be traced to the camera that produced it (History, grid tile flashing).
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from pipeline.pipeline import Pipeline
from pipeline.frame_broadcaster import FrameBroadcaster


class _Cam:
    """One camera: its pipeline + its frame broadcaster."""
    __slots__ = ("pipeline", "broadcaster", "source_config")

    def __init__(self, pipeline: Pipeline, broadcaster: FrameBroadcaster,
                 source_config: dict):
        self.pipeline      = pipeline
        self.broadcaster   = broadcaster
        self.source_config = source_config


class MultiCameraManager:
    def __init__(self, ws_manager, loop, stream_fps: int = 15,
                 stream_width: int = 960, stream_height: int = 540,
                 stream_quality: int = 70):
        self._ws_manager = ws_manager
        self._loop       = loop
        self._fps        = stream_fps
        self._width      = stream_width
        self._height     = stream_height
        self._quality    = stream_quality

        self._lock: threading.Lock = threading.Lock()
        self._cams: dict[str, _Cam] = {}

        # Widened callbacks (camera_id, ...). api.py sets these once; the manager
        # binds a per-camera closure onto each Pipeline so events carry their id.
        # (camera_id, msg, level, ev_type, meta) — meta carries the structured
        # event fields (category, ids, contributing cameras) that the plain
        # message string cannot.
        self.on_alert:  Optional[Callable[[str, str, str, str, dict], None]] = None
        self.on_status: Optional[Callable[[str, dict], None]] = None

        # Last settings dict applied, so a camera that starts later inherits it.
        self._last_settings: dict = {}

        # SHARED-MODEL edge multi-camera: ONE ModelHub + ONE InferenceScheduler
        # shared by every LOCAL camera → VRAM constant as cameras are added. Built
        # lazily on the first local-camera start (loading models is heavy). Cloud
        # cameras bypass this (they run remote inference).
        self._hub = None
        self._scheduler = None
        self._shared_lock = threading.Lock()

    def _ensure_shared(self):
        """Build the shared ModelHub + InferenceScheduler once (first local camera).
        Returns (hub, scheduler) or (None, None) if models fail to load (caller then
        falls back to a per-camera engine)."""
        with self._shared_lock:
            if self._hub is not None and self._scheduler is not None:
                return self._hub, self._scheduler
            try:
                from utils.model_hub import ModelHub
                from pipeline.scheduler import InferenceScheduler
                self._hub = ModelHub()
                self._scheduler = InferenceScheduler(self._hub, fps=self._fps)
                self._scheduler.start()
                print("[Manager] 🧩 shared ModelHub + scheduler ready")
            except Exception as e:
                print(f"[Manager] ⚠️ shared hub unavailable → per-camera engines: {e}")
                self._hub = self._scheduler = None
            return self._hub, self._scheduler

    # ── lifecycle ─────────────────────────────────────────────
    def start(self, camera_id: str, source_config: dict) -> bool:
        """Start (or restart) one camera. Returns True on success."""
        source_config = dict(source_config)
        source_config["camera_id"] = camera_id

        with self._lock:
            existing = self._cams.get(camera_id)
        if existing is not None:
            self.stop(camera_id)

        pipe = Pipeline()
        # LOCAL cameras share ONE ModelHub via the central scheduler (VRAM constant).
        # Cloud cameras (cloud_enabled + url) run remote inference and skip this.
        is_cloud = bool(source_config.get("cloud_enabled") and source_config.get("cloud_url"))
        if not is_cloud:
            hub, sched = self._ensure_shared()
            if hub is not None and sched is not None:
                pipe._shared_hub = hub
                pipe._shared_scheduler = sched
        # Bind this camera's id onto the shared callbacks.
        if self.on_alert is not None:
            pipe.on_alert = (lambda msg, level, ev_type="ppe", meta=None, _cid=camera_id:
                             self.on_alert(_cid, msg, level, ev_type, meta))
        if self.on_status is not None:
            pipe.on_status = (lambda status, _cid=camera_id:
                              self.on_status(_cid, status))

        # Apply the last-known settings so a freshly-started camera is configured
        # (LINE tokens, AI thresholds, per-camera roles) before its first frame.
        if self._last_settings:
            try:
                pipe.apply_settings(self._last_settings)
            except Exception as e:
                print(f"[Manager] settings preload for {camera_id}: {e}")

        ok = pipe.start(source_config)
        if not ok:
            return False

        bc = FrameBroadcaster(pipe, self._ws_manager, self._loop,
                              fps=self._fps, width=self._width,
                              height=self._height, quality=self._quality,
                              camera_id=camera_id)
        bc.start()

        with self._lock:
            self._cams[camera_id] = _Cam(pipe, bc, source_config)
        print(f"[Manager] ▶️  camera '{camera_id}' started ({source_config.get('source')})")
        return True

    def stop(self, camera_id: str) -> bool:
        with self._lock:
            cam = self._cams.pop(camera_id, None)
        if cam is None:
            return False
        try:
            cam.broadcaster.stop()
        except Exception:
            pass
        try:
            cam.pipeline.stop()
        except Exception:
            pass
        print(f"[Manager] ⏹️  camera '{camera_id}' stopped")
        return True

    def stop_all(self):
        for cid in self.active_ids():
            self.stop(cid)
        # Tear down the shared scheduler once no camera is running.
        with self._shared_lock:
            if self._scheduler is not None:
                try:
                    self._scheduler.stop()
                except Exception:
                    pass
                self._scheduler = None
            self._hub = None

    # ── queries ───────────────────────────────────────────────
    def active_ids(self) -> list[str]:
        with self._lock:
            return list(self._cams.keys())

    def get(self, camera_id: str) -> Optional[Pipeline]:
        with self._lock:
            cam = self._cams.get(camera_id)
        return cam.pipeline if cam else None

    def primary(self) -> Optional[Pipeline]:
        """The first running pipeline — a back-compat handle for legacy
        single-camera endpoints (/api/status fallback, snapshot). Returns None
        when nothing is running."""
        with self._lock:
            for cam in self._cams.values():
                return cam.pipeline
        return None

    def is_running(self, camera_id: str) -> bool:
        p = self.get(camera_id)
        return bool(p and p.is_running())

    def any_running(self) -> bool:
        return len(self.active_ids()) > 0

    def statuses(self) -> dict:
        """Per-camera status snapshots keyed by camera_id (for the grid)."""
        out = {}
        with self._lock:
            cams = dict(self._cams)
        for cid, cam in cams.items():
            try:
                with cam.pipeline._lock:
                    s = dict(cam.pipeline.status)
                s["uptime"] = cam.pipeline.get_uptime()
                # Detection latency (ms) — how old the boxes on this tile are.
                # Rises as cameras are added; the honest signal that this box has
                # run out of budget, which raw FPS hides.
                s["detect_age_ms"] = cam.pipeline.get_result_age_ms()
                out[cid] = s
            except Exception:
                pass
        return out

    def get_snapshot(self, camera_id: str):
        p = self.get(camera_id)
        return p.get_snapshot() if p else None

    def get_latest_frame(self, camera_id: str):
        p = self.get(camera_id)
        return p.get_latest_frame() if p else None

    # ── broadcast helpers to every camera ─────────────────────
    def reload_zones(self):
        with self._lock:
            cams = list(self._cams.values())
        for cam in cams:
            try:
                cam.pipeline.reload_zones()
            except Exception:
                pass

    # Fields that decide WHICH physical stream a camera is pulling. A change to
    # any of them cannot be hot-applied: the capture thread is already connected
    # to the old one.
    _SOURCE_KEYS = ("source", "webcam_index", "rtsp_url", "video_file_path")

    def _source_changed(self, cam: "_Cam", new_cfg: dict) -> bool:
        old = cam.source_config or {}
        return any(str(old.get(k, "")) != str(new_cfg.get(k, ""))
                   for k in self._SOURCE_KEYS)

    def apply_settings(self, settings: dict):
        self._last_settings = dict(settings)
        self.apply_stream_settings(settings.get("display") or {})
        with self._lock:
            items = list(self._cams.items())

        # Roles, thresholds and LINE routing hot-apply. The SOURCE cannot: the
        # capture thread is already connected to the old URL, so a camera whose
        # address changed keeps streaming the previous one until something
        # restarts it. That is how two tiles ended up showing the SAME camera
        # after two entries were pointed at one IP and then corrected — the file
        # was right, the running pipelines were not. Restart exactly those.
        restart: list = []
        cams_cfg = settings.get("cameras") or {}
        for cid, cam in items:
            try:
                cam.pipeline.apply_settings(settings)
            except Exception as e:
                print(f"[Manager] apply_settings ({cid}): {e}")
            new_cfg = cams_cfg.get(cid)
            if isinstance(new_cfg, dict) and self._source_changed(cam, new_cfg):
                # Keep everything the camera was started with (cloud settings and
                # anything else that never appears in the cameras registry) and
                # overlay only what actually changed.
                merged = dict(cam.source_config or {})
                merged.update(new_cfg)
                merged["camera_id"] = cid
                restart.append((cid, merged))

        if not restart:
            return
        # Off the caller's thread: stop() joins capture threads for up to a few
        # seconds each, and this runs inside the Settings-save request.
        def _restart():
            for cid, merged in restart:
                print(f"[Manager] 🔄 '{cid}' source changed → reconnecting "
                      f"({merged.get('source')})")
                try:
                    self.start(cid, merged)     # start() stops the old one first
                except Exception as e:
                    print(f"[Manager] restart of '{cid}' failed: {e}")
        threading.Thread(target=_restart, daemon=True,
                         name="CameraSourceRestart").start()

    def apply_stream_settings(self, display: dict):
        """Apply Settings → display (stream fps / size / JPEG quality) to every
        running broadcaster AND to cameras started later.

        These knobs existed in settings.json but nothing ever read them, so the
        WebSocket always ran at the hardcoded 15 fps x 960x540 x q70 — per camera.
        With 4 cameras that is ~6 MB/s of base64 JPEG through ONE WebSocket and one
        WebView; when the socket cannot drain, the broadcaster's in-flight send
        never completes and EVERY tile stops updating (frozen on its last frame).
        Being able to turn these down is the direct lever on that.
        """
        if not display:
            return
        fps     = int(display.get("stream_fps") or 0) or None
        width   = int(display.get("stream_width") or 0) or None
        height  = int(display.get("stream_height") or 0) or None
        quality = int(display.get("stream_jpeg_quality") or 0) or None
        if fps:     self._fps     = fps
        if width:   self._width   = width
        if height:  self._height  = height
        if quality: self._quality = quality
        with self._lock:
            cams = list(self._cams.values())
        for cam in cams:
            try:
                cam.broadcaster.set_stream(fps, width, height, quality)
            except Exception as e:
                print(f"[Manager] apply_stream_settings: {e}")
