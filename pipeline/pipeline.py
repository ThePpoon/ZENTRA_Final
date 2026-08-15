"""
pipeline/pipeline.py — ZENTRA Camera Pipeline (passthrough, no detection)
Reads frames from a camera source, annotates nothing, and exposes them
for WebSocket broadcast.  Detection modules will be added one by one.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

_APP_DATA = Path(__file__).parent.parent / "data"
_APP_DATA.mkdir(exist_ok=True)

# Put the AI backend on the path so `config`, `utils.*`, `alerts.*` import here
# (single-repo: backend lives at <repo>/backend).
_BACKEND = Path(__file__).parent.parent / "backend"
if _BACKEND.is_dir() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ================================================================
# FRAME READER
# ================================================================
class _FrameReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, loop: bool = False, src_fps: float = 0.0):
        super().__init__(daemon=True, name="FrameReader")
        self.cap   = cap
        # Small queue + drop-oldest (see _push): keep only the freshest 1-2 frames
        # so the detector/display always work on near-live frames instead of a
        # backlog. Larger buffers just add standing latency for a real-time feed.
        self.q     = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._loop = loop     # file sources: rewind at EOF so demo/testing never freezes
        # File sources decode instantly; throttle to the clip's native fps so the
        # frame stream stays time-ordered (a live camera self-paces, so src_fps=0
        # there). Without this, files replay at hundreds of fps and loop backwards,
        # which breaks ByteTrack's temporal continuity → track IDs churn → the
        # 3-of-5 zone/PPE confirm never accumulates.
        self._interval = (1.0 / src_fps) if src_fps and src_fps > 0 else 0.0

    def run(self):
        errors = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            ret, frame = self.cap.read()
            if not ret:
                if self._loop:
                    # EOF on a video file → seek back to the start and keep playing
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        errors = 0
                        self._push(frame)
                        self._throttle(t0)
                        continue
                errors += 1
                if errors > 30:
                    break
                time.sleep(0.05)
                continue
            errors = 0
            self._push(frame)
            self._throttle(t0)

    def _throttle(self, t0: float):
        if self._interval:
            dt = self._interval - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    def _push(self, frame):
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(frame)

    def read(self):
        try:
            return True, self.q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop.set()


def _draw_remote_overlay(frame, draw):
    """Overlay cloud-returned primitives (person boxes + danger zones + labels)
    onto a LIVE edge frame. Reuses the engine's exact text renderer so remote
    boxes look identical to local ones. `draw` = {"boxes":[...], "zones":[...]}."""
    if not draw:
        return frame
    boxes = draw.get("boxes") or []
    zones = draw.get("zones") or []
    if not boxes and not zones:
        return frame
    # Same proportional sizing as the local renderer: this frame is the camera's
    # native resolution and gets shrunk for the browser afterwards, so fixed pixel
    # sizes come out unreadable on a multi-megapixel stream.
    s = max(1.0, frame.shape[1] / 960.0)
    thick = max(2, int(round(2 * s)))
    size = max(12, int(round(18 * s)))
    pad = int(round(22 * s))
    dot = max(3, int(round(4 * s)))

    texts = []   # (x, y_top, text, rgb)
    for z in zones:
        pts = z.get("points") or []
        if len(pts) < 2:
            continue
        col = tuple(int(c) for c in z.get("color", (0, 0, 220)))
        cv2.polylines(frame, [np.array(pts, np.int32).reshape(-1, 1, 2)], True, col, thick)
        if z.get("name"):
            x0, y0 = pts[0]
            texts.append((int(x0), max(2, int(y0) - pad), z["name"],
                          (col[2], col[1], col[0])))
    for b in boxes:
        col = tuple(int(c) for c in b.get("color", (255, 190, 0)))
        cv2.rectangle(frame, (int(b["x1"]), int(b["y1"])),
                      (int(b["x2"]), int(b["y2"])), col, thick)
        foot = b.get("foot")
        if foot:
            cv2.circle(frame, (int(foot[0]), int(foot[1])), dot, col, -1)
        texts.append((int(b["x1"]), max(2, int(b["y1"]) - pad), b.get("label", ""),
                      (col[2], col[1], col[0])))
    try:
        from utils.ppe_engine import draw_texts_on
        return draw_texts_on(frame, texts, size)
    except Exception:
        # Fallback: cv2 text (no Thai shaping) so labels still show something.
        for (x, y, t, rgb) in texts:
            if t:
                cv2.putText(frame, t, (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (rgb[2], rgb[1], rgb[0]), 1)
        return frame


def apply_global_settings(settings: dict) -> dict:
    """Apply the settings that live in the `config` MODULE, not on a pipeline:
    LINE credentials + routing, AI thresholds, collection and cooldowns.

    This used to sit inside Pipeline.apply_settings, so it only ran for cameras
    that were already RUNNING. With no camera started the manager's loop had
    nothing to iterate, cfg.LINE_OA_CHANNEL_ACCESS_TOKEN stayed empty, and the
    History page's "send to LINE" button reported "no token configured" for a
    site whose token was set and valid. Same for the AI sliders: they did not
    take effect until a camera happened to be up.

    Returns the per-level alert switches, which ARE per pipeline.
    """
    levels = {"warning": True, "alert": True, "emergency": True}
    line = settings.get("line", {})
    try:
        import config as cfg
        if "channel_access_token" in line:
            cfg.LINE_OA_CHANNEL_ACCESS_TOKEN = line["channel_access_token"]

        # ── Group routing ────────────────────────────────────────────
        # NEW model: a flat list of groups; EVERY alert (any level) goes to
        # EVERY enabled group. No per-level routing. Legacy per-level keys
        # (group_supervisor/safety/emergency) are still honoured as a
        # fallback so old settings.json files keep working.
        #
        # Guard on a REAL group (non-empty id), not merely a present list:
        # _load_settings injects the default groups=[{id:""}], so the list is
        # always present after merge. Taking this branch on the empty default
        # would shadow a legacy user's group_supervisor id and route alerts to
        # nobody until they re-save. Mirrors the UI's `hasRealGroup` check.
        _groups = line.get("groups")
        _has_real_group = isinstance(_groups, list) and any(
            isinstance(g, dict) and str(g.get("id", "")).strip() for g in _groups)
        if _has_real_group:
            groups = _groups
            enabled_ids: list[str] = []
            cooldowns: dict[str, int] = {}
            for g in groups:
                if not isinstance(g, dict):
                    continue
                gid = str(g.get("id", "")).strip()
                if not gid:
                    continue
                cooldowns[gid] = int(g.get("cooldown", 30) or 0)
                if g.get("enabled", True) and gid not in enabled_ids:
                    enabled_ids.append(gid)
            # CRITICAL: config.ALERT_RECIPIENTS is built ONCE at import time
            # from the (then-empty) group ids, and send_line_notify() picks
            # recipients from it per level. Rebuild it here with the live ids
            # so real detections actually reach LINE — and point every level
            # at the SAME full list so all alerts go to all groups.
            cfg.ALERT_RECIPIENTS = {
                cfg.ALERT_LEVEL_WARNING:   list(enabled_ids),
                cfg.ALERT_LEVEL_ALERT:     list(enabled_ids),
                cfg.ALERT_LEVEL_EMERGENCY: list(enabled_ids),
            }
            cfg.LINE_ALL_GROUPS   = list(enabled_ids)   # daily report target
            cfg.LINE_GROUP_COOLDOWN = cooldowns          # per-group throttle
            # Keep legacy vars pointed at the first enabled group so any code
            # (and the send-line validation) that still reads them stays valid.
            first = enabled_ids[0] if enabled_ids else ""
            cfg.LINE_OA_GROUP_SUPERVISOR = first
            cfg.LINE_OA_GROUP_SAFETY     = first
            cfg.LINE_OA_GROUP_EMERGENCY  = first
        elif any(k in line for k in ("group_supervisor", "group_safety", "group_emergency")):
            sup = line.get("group_supervisor", getattr(cfg, "LINE_OA_GROUP_SUPERVISOR", ""))
            saf = line.get("group_safety",     getattr(cfg, "LINE_OA_GROUP_SAFETY", ""))
            emg = line.get("group_emergency",  getattr(cfg, "LINE_OA_GROUP_EMERGENCY", ""))
            cfg.LINE_OA_GROUP_SUPERVISOR = sup
            cfg.LINE_OA_GROUP_SAFETY     = saf
            cfg.LINE_OA_GROUP_EMERGENCY  = emg
            allg = [g for g in dict.fromkeys([sup, saf, emg]) if g]
            cfg.ALERT_RECIPIENTS = {
                cfg.ALERT_LEVEL_WARNING:   list(allg),
                cfg.ALERT_LEVEL_ALERT:     list(allg),
                cfg.ALERT_LEVEL_EMERGENCY: list(allg),
            }
            cfg.LINE_ALL_GROUPS = list(allg)

        # ── AI thresholds (INFERENCE_CONFIDENCE is read per-frame in
        # detect_track → hot; confirm/cooldown need refresh_tunables) ──
        ai = settings.get("ai", {})
        if "ppe_confidence" in ai:
            cfg.INFERENCE_CONFIDENCE = float(ai["ppe_confidence"])
        if "abstain_weight" in ai:
            # Minimum evidence weight before a camera may accuse anyone.
            # Read per-frame inside _cat_hit, so the slider takes effect
            # immediately — no engine rebuild. 0 disables the gate, which
            # restores the exact pre-abstention behaviour.
            w = float(ai["abstain_weight"])
            cfg.PPE_ABSTAIN_W = w
            cfg.PPE_ABSTAIN_ENABLED = w > 0.0
        if "fall_bbox_ratio" in ai:
            # This slider wrote FALL_BBOX_RATIO_THRESH, which the current
            # fall_detector never reads — moving it did nothing. It now drives
            # the value it was always meant to: FALL_AR_ABS_MIN, the absolute
            # width/height floor above which a box counts as prone (read
            # per-frame in _posture, so it takes effect without a model reload).
            cfg.FALL_AR_ABS_MIN = float(ai["fall_bbox_ratio"])
            cfg.FALL_BBOX_RATIO_THRESH = float(ai["fall_bbox_ratio"])  # legacy mirror
        if "fall_confirm_frames" in ai:
            # CLAMP to the window. The confirmer is N-of-M over a deque of
            # maxlen=FALL_CONFIRM_WINDOW, so N > M can never be satisfied and a
            # saved value above the window makes fall alarms mathematically
            # impossible in the live app — silently. A safety module that cannot
            # fire must never be reachable from a settings slider.
            win = int(getattr(cfg, "FALL_CONFIRM_WINDOW", 5))
            want = int(ai["fall_confirm_frames"])
            if want > win:
                print(f"[Pipeline] ⚠️ fall_confirm_frames={want} > window={win} "
                      f"→ ตรึงไว้ที่ {win} (ค่าเดิมทำให้แจ้งเตือนการล้มไม่ได้เลย)")
            cfg.FALL_CONFIRM_FRAMES = max(1, min(want, win))
        if "fall_mode" in ai:
            cfg.FALL_MODE = str(ai["fall_mode"])

        # ── Dataset collection (Settings → ข้อมูล) ──
        data = settings.get("data", {})
        if "auto_collect" in data:
            cfg.AUTO_COLLECT_FRAMES = bool(data["auto_collect"])
        if "normal_interval_sec" in data:
            cfg.COLLECT_NORMAL_INTERVAL_SEC = float(data["normal_interval_sec"])

        alerts = settings.get("alerts", {})
        if "violation_cooldown_seconds" in alerts:
            cfg.VIOLATION_COOLDOWN_SECONDS = int(alerts["violation_cooldown_seconds"])
        if "zone_cooldown_seconds" in alerts:
            cfg.ZONE_COOLDOWN_SECONDS = int(alerts["zone_cooldown_seconds"])
        if "fall_cooldown_seconds" in alerts:
            cfg.FALL_COOLDOWN_SECONDS = int(alerts["fall_cooldown_seconds"])

        # Per-level alert switches (disabled level → fully suppressed)
        for lvl, key in (("warning", "warning_enabled"),
                         ("alert", "alert_enabled"),
                         ("emergency", "emergency_enabled")):
            if key in alerts:
                levels[lvl] = bool(alerts[key])
    except ImportError:
        pass
    return levels


# ================================================================
# PIPELINE
# ================================================================
class Pipeline:
    """Camera capture pipeline — passthrough (no AI detection yet)."""

    def __init__(self):
        self._lock        = threading.Lock()
        self._frame_lock  = threading.Lock()
        self._stop_evt    = threading.Event()
        self._running     = False

        self._latest_frame: Optional[np.ndarray] = None
        self._cap         = None
        self._reader      = None
        self._reader_loop = False    # reader options, kept for reconnects
        self._reader_fps  = 0.0
        self._proc_thr    = None
        self._start_time: Optional[float] = None
        self._flip_override: Optional[bool] = None

        self.on_alert:  Optional[Callable[[str, str, bool], None]] = None
        self.on_status: Optional[Callable[[dict], None]] = None

        # Per-camera detection roles (None = all on) + per-level alert switches.
        self._roles: Optional[set] = None
        self._ppe_items: Optional[list] = None   # which PPE categories to enforce (None = cfg default)
        self._alert_levels: dict = {"warning": True, "alert": True, "emergency": True}

        self._source_config: dict = {}
        self._engine = None          # PPEEngine (lazy — built on start)
        # Remote (cloud) inference: when set, this pipeline does NOT build a local
        # engine. It captures frames, POSTs them to the cloud inference server, and
        # displays the annotated frames it returns — the notebook becomes a thin
        # client and the heavy models run on a cloud GPU. {url, token, fps}.
        self._remote: Optional[dict] = None
        # SHARED-MODEL (edge multi-camera): when a ModelHub + InferenceScheduler are
        # injected by the manager, this pipeline builds a CameraEngine that SHARES
        # the hub's models (VRAM constant across cameras) and registers with the
        # central scheduler. It only captures + displays; the scheduler drives
        # inference. Fall still runs per-camera (its own loop).
        self._shared_hub = None
        self._shared_scheduler = None
        # FRAME PACKET: (frame, frame_id, t_capture_monotonic) published as ONE
        # tuple so a reader can never pair a frame with another frame's id/time.
        # The id and the timestamp are stamped at CAPTURE, so both travel with the
        # result rather than being inferred from call order (architecture review
        # §12) — that is what makes detect_age_ms in /api/status truthful.
        self._shared_pkt = None            # latest packet the scheduler reads
        self._last_handed_pkt = None       # packet last given out (fall pairing)
        self._last_result_t = 0.0          # capture time of the newest result
        # Decoupled detection: display loop runs at camera FPS (draws latest
        # boxes); a worker thread runs the slow inference on the newest frame.
        # (frame, frame_id) published as ONE tuple. Two separate fields could not
        # be read atomically: the detect loop read `raw` then `_raw_id`, and a new
        # frame landing between those two reads paired an old frame with a newer
        # id — advancing last_seen past a frame that was never processed.
        self._raw_pair: Optional[tuple] = None
        self._latest_recs: list = []
        self._recs_pair = None      # (frame, recs, frame_id) published together for the fall loop
        self._detect_thr = None
        self._fall_thr = None
        self._last_collect_ts: dict = {}   # category → monotonic ts of its last saved frame

        self._engine_error: Optional[str] = None   # why the engine failed to build

        self.status: dict = {
            "running":        False,
            "source":         None,
            "camera":         "disconnected",
            "modules":        {"ppe": "standby", "zone": "standby", "fall": "standby"},
            "alerts":         {"total": 0, "warning": 0, "alert": 0, "emergency": 0},
            "uptime_seconds": 0,
            "last_emergency": None,
            "engine_error":   None,
        }

    @staticmethod
    def _zero_alerts() -> dict:
        return {"total": 0, "warning": 0, "alert": 0, "emergency": 0}

    # ── Public API ────────────────────────────────────────────

    def start(self, source_config: dict) -> bool:
        if self._running:
            self.stop()
        self._stop_evt.clear()
        self._source_config = dict(source_config)
        # Read the mirror choice from THIS camera's start config. The manager
        # calls apply_settings() before start(), when _source_config is still
        # empty and the camera id therefore reads as "cam0" — so without this
        # every camera would inherit cam0's setting.
        if source_config.get("flip_horizontal") is not None:
            self._flip_override = bool(source_config["flip_horizontal"])
        # Cloud offload: when the source_config carries a cloud_url and is enabled,
        # run in remote mode (no local engine — POST frames to the cloud GPU).
        self._remote = None
        if source_config.get("cloud_enabled") and source_config.get("cloud_url"):
            self._remote = {
                "url":   str(source_config["cloud_url"]).rstrip("/"),
                "token": source_config.get("cloud_token", ""),
                "fps":   float(source_config.get("cloud_fps", 10) or 10),
            }
        try:
            self._apply_config(source_config)
            self._cap = self._open_camera(source_config)
        except Exception as e:
            print(f"[Pipeline] ❌ start failed: {e}")
            self._set_camera_state("disconnected")
            return False

        self._start_time = time.time()
        self._running    = True
        self._engine_error = None
        with self._lock:
            self.status["running"] = True
            self.status["source"]  = source_config.get("source", "webcam")
            self.status["modules"] = {"ppe": "standby", "zone": "standby", "fall": "standby"}
            # A new session starts from zero. History (SQLite) is the durable
            # record; these are live per-session counters and used to accumulate
            # across restarts, disagreeing with the History page.
            self.status["alerts"] = self._zero_alerts()
            self.status["engine_error"]   = None
            self.status["last_emergency"] = None
        self._set_camera_state("connected")

        self._proc_thr = threading.Thread(
            target=self._process_loop, daemon=True, name="PipelineLoop"
        )
        self._proc_thr.start()
        print(f"[Pipeline] ✅ Started (passthrough) — {source_config.get('source', 'webcam')}")
        return True

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_evt.set()
        if self._reader:
            try:
                self._reader.stop()
            except Exception:
                pass
        if self._fall_thr and self._fall_thr.is_alive():
            self._fall_thr.join(timeout=3.0)
        if self._engine is not None:
            try:
                self._engine.close_fall()      # release mediapipe/tflite graphs
            except Exception:
                pass
        if self._detect_thr and self._detect_thr.is_alive():
            self._detect_thr.join(timeout=3.0)
        if self._proc_thr and self._proc_thr.is_alive():
            self._proc_thr.join(timeout=3.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        # Uptime is "how long has this session been live". Leaving _start_time set
        # made get_uptime() keep counting after the camera stopped.
        self._start_time = None
        with self._lock:
            self.status["running"] = False
        self._set_camera_state("disconnected")
        print("[Pipeline] ⏹️  Stopped")

    def is_running(self) -> bool:
        return self._running

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_snapshot(self) -> Optional[bytes]:
        frame = self.get_latest_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def get_uptime(self) -> int:
        if self._start_time is None:
            return 0
        return int(time.time() - self._start_time)

    def _should_flip(self, is_webcam: bool) -> bool:
        """Mirror this camera's frames horizontally?

        The Settings toggle used to be honoured ONLY for webcams, so on an RTSP
        site — every real deployment — flipping it did nothing at all and the
        control looked broken. An explicit choice now applies to any source; the
        webcam-only rule survives just as the DEFAULT when no choice has been
        made (a laptop webcam is a selfie view and is expected to be mirrored,
        while an IP camera is not).
        """
        if self._flip_override is not None:
            return bool(self._flip_override)
        return bool(is_webcam)

    def get_result_age_ms(self) -> int:
        """How stale the boxes currently drawn on screen are, in ms (-1 = none yet).

        This is DETECTION LATENCY, not FPS — the architecture review (§26) calls
        it out as the number that actually matters: "Inference FPS = 10, Latency =
        2.5 s" is a bad system. Now that capture stamps every frame, this is
        measurable instead of inferred, and it is the metric that tells you when
        adding another camera has started to hurt."""
        if not self._last_result_t:
            return -1
        return int((time.monotonic() - self._last_result_t) * 1000)

    def reload_zones(self):
        if self._engine is not None:
            try:
                self._engine.reload_zones()
                print("[Pipeline] 🗺️  zones reloaded")
            except Exception as e:
                print(f"[Pipeline] reload_zones: {e}")

    def _build_engine(self):
        """Build the PPE/zone engine (ultralytics + ByteTrack). On any failure,
        return None → the loop falls back to passthrough so the app never crashes.

        A passthrough pipeline shows clean video with no boxes, which reads as
        "nobody is violating anything". For a safety system that silent success is
        the worst possible failure, so the reason is recorded in status.engine_error
        and every module is marked "error" for the UI to shout about.
        """
        try:
            from utils.ppe_engine import PPEEngine
            cam_id = self._source_config.get("camera_id", "cam0")
            roles  = self._source_config.get("roles")
            self._roles = set(roles) if roles is not None else None
            self._ppe_items = self._source_config.get("ppe_items")
            # In shared mode pass the ModelHub → the engine references shared models
            # (VRAM constant) + gets its own ByteTracker instead of loading models.
            eng = PPEEngine(zones_path=str(_APP_DATA / "zones.json"),
                            camera_id=cam_id, roles=self._roles,
                            ppe_items=self._ppe_items, hub=self._shared_hub)
            import os as _os
            # ppe_detector is None whenever the PPE role is off (zone-only / fall-only
            # / no roles) — dereferencing .model_path here unconditionally crashed the
            # WHOLE engine build, so picking any PPE-excluding role silently killed all
            # detection ("ไม่ขึ้นบนกล้อง"). Guard it.
            _pd = eng.person_detector
            _person_name = _os.path.basename(_pd.model_path) if getattr(_pd, "model_path", None) else "off"
            _ppe_name = (_os.path.basename(eng.ppe_detector.model_path)
                         if eng.ppe_detector is not None and getattr(eng.ppe_detector, "model_path", None)
                         else "off")
            print(f"[Pipeline] 🧠 PPE engine ready ("
                  f"person={_person_name}, "
                  f"ppe={_ppe_name}, "
                  f"device={eng.detector.device}, camera={cam_id}, zones={len(eng.zones)}, "
                  f"roles={sorted(self._roles) if self._roles is not None else 'all'})")
            return eng
        except Exception as e:
            print(f"[Pipeline] ❌ engine unavailable → NO DETECTION. reason: {e}")
            traceback.print_exc()
            self._engine_error = str(e)
            with self._lock:
                self.status["engine_error"] = str(e)
                self.status["modules"] = {"ppe": "error", "zone": "error", "fall": "error"}
                snapshot = dict(self.status)
            if self.on_status:
                try:
                    self.on_status(snapshot)
                except Exception:
                    pass
            return None

    def _detect_loop(self):
        """Worker: run the heavy inference on the NEWEST raw frame, publish boxes
        (`_latest_recs`) for the display loop, and emit events. Always grabs the
        latest frame → auto-skips frames it can't keep up with (smooth video)."""
        last_seen = -1
        while not self._stop_evt.is_set() and self._running:
            pair = self._raw_pair            # one read → frame, id and time agree
            if self._engine is None or pair is None or pair[1] == last_seen:
                time.sleep(0.005)
                continue
            raw, rid, t_cap = pair
            last_seen = rid
            self._last_result_t = t_cap
            try:
                recs, events = self._engine.detect(raw)
                self._latest_recs = recs
                # Publish frame+boxes ATOMICALLY. The fall loop used to pair the
                # newest raw frame with these (older) boxes; during a fall the person
                # moves fast enough that the crop / keypoint match lands on the wrong
                # place. A slightly older frame that MATCHES its boxes is correct.
                self._recs_pair = (raw, recs, rid)
                for ev in events:
                    self._emit_event(ev)
                # `raw` is the CLEAN frame (the display loop draws on its own copy)
                # and last_dets came from THIS frame — the only point in the app
                # where a trainable image/label pair can be cut.
                self._collect(raw, self._engine.last_dets, events)
            except Exception as e:
                print(f"[Pipeline] detect error: {e}")

    def _collect(self, frame, dets: list, events: list):
        """Save frame + pseudo-labels for Roboflow labelling / site fine-tuning.

        Event frames go to their incident category; the rest are sampled as
        `normal` so the dataset also carries this site's ordinary background. The
        collector owns the on/off switch, quota and near-duplicate gate, so this
        is a no-op most of the time — an ~10ms JPEG write every few seconds at
        worst."""
        try:
            import config as cfg
            from utils.collector import get_collector

            # Every label we write lives in the PPE taxonomy. With the PPE role
            # off the engine never runs the PPE detector, so `dets` holds people
            # and nothing else — and a label file that says "one person, no helmet
            # box anywhere" is not missing data, it is FALSE data: train on it and
            # the model learns that a helmet is background. A camera that cannot
            # see PPE must not contribute PPE labels.
            if self._engine is None or not self._engine.ppe_enabled:
                return

            kinds = {ev.get("type") for ev in events or []}
            if "ppe" in kinds:
                category = "ppe_violations"
            elif "zone" in kinds:
                category = "zone_intrusions"
            else:
                category = "normal"

            # The interval throttles EVERY category, not just `normal`. An event
            # frame used to be saved the moment the engine alarmed, so one worker
            # standing without a helmet filled the folder at the alert cooldown's
            # rate (30 s per person per category — faster still with several
            # people) — dozens of near-identical frames of one incident, which is
            # labelling work without new information. Time between saved frames is
            # what buys dataset diversity, so it applies per category.
            gap = float(getattr(cfg, "COLLECT_NORMAL_INTERVAL_SEC", 10.0))
            now = time.monotonic()
            if now - self._last_collect_ts.get(category, 0.0) < gap:
                return
            self._last_collect_ts[category] = now

            get_collector().collect_dets(frame, dets, category)
        except Exception as e:
            print(f"[Pipeline] collect error: {e}")

    def _fall_loop(self):
        """Fall runs on its OWN fixed-cadence loop, not on the detect loop.

        The classifier consumes 30 evenly-spaced frames (~1-2 s of motion); the
        detect loop drops frames whenever inference lags, which would stretch that
        window unpredictably. Ticking at FALL_LOOP_FPS keeps dt uniform. The loop
        stays alive even while the role is off so it can pick the role up live."""
        try:
            import config as cfg
            interval = 1.0 / max(1, int(getattr(cfg, "FALL_LOOP_FPS", 15)))
        except ImportError:
            interval = 1.0 / 15
        overruns = 0
        last_fid = -1
        while not self._stop_evt.is_set() and self._running:
            t0 = time.monotonic()
            eng = self._engine
            if eng is not None and getattr(eng, "fall_ready", False):
                pair = self._recs_pair          # one read → frame and boxes agree
                raw, recs, fid = pair if pair else (None, [], -1)
                # Detect runs slower than FALL_LOOP_FPS, so this loop would otherwise
                # append the SAME frame to the 30-frame window several times: the
                # sequence freezes while `now` keeps advancing, and a quick fall looks
                # like a slow one to a model trained on real motion. Only step on a
                # frame we have not already consumed.
                fresh, last_fid = (fid != last_fid), fid
                if raw is not None and recs and fresh:
                    try:
                        fall_evs = eng.fall_step(raw, recs)
                        for ev in fall_evs:
                            self._emit_event(ev)
                        if fall_evs:
                            # Label from `recs`, not engine.last_dets: a newer detect()
                            # may have overwritten last_dets since this (frame, recs)
                            # pair was published, and boxes from another frame would be
                            # a wrong label.
                            from utils.collector import get_collector, DataCollector
                            get_collector().collect_dets(
                                raw, DataCollector.dets_from_recs(recs), "fall_events")
                    except Exception as e:
                        print(f"[Pipeline] fall error: {e}")
                spent = time.monotonic() - t0
                if spent > interval:
                    overruns += 1
                    if overruns % 30 == 1:      # never drop a person; just report
                        print(f"[Pipeline] ⚠️ fall tick {spent*1000:.0f}ms > "
                              f"{interval*1000:.0f}ms budget ({len(recs)} people)")
            dt = interval - (time.monotonic() - t0)
            time.sleep(dt if dt > 0 else 0.001)

    def _sync_module_status(self):
        """Reflect engine roles into status['modules'] (ok / standby / off)."""
        if self._engine is None:
            return
        with self._lock:
            self.status["modules"]["ppe"] = "ok" if self._engine.ppe_enabled else "off"
            if not self._engine.zone_enabled:
                self.status["modules"]["zone"] = "off"
            else:
                self.status["modules"]["zone"] = "ok" if self._engine.zones else "standby"
            if not self._engine.fall_enabled:
                self.status["modules"]["fall"] = "off"
            else:                                # enabled but models failed → visible
                self.status["modules"]["fall"] = "ok" if self._engine.fall_ready else "err"
            snapshot = dict(self.status)
        if self.on_status:
            try:
                self.on_status(snapshot)
            except Exception:
                pass

    def _emit_event(self, ev: dict):
        """Route an engine event → alert counters + UI/History callback (on_alert)."""
        level = ev.get("level", "warning")
        # Per-level alert switch (Settings): a disabled level is fully suppressed.
        if not self._alert_levels.get(level, True):
            return
        with self._lock:
            a = self.status["alerts"]
            a["total"] = a.get("total", 0) + 1
            # One bucket per level. A zone intrusion is level "alert" and used to
            # fall into the `else` branch here, so every intrusion incremented the
            # EMERGENCY counter and lit the dashboard's red alarm state.
            if level in ("warning", "alert", "emergency"):
                a[level] = a.get(level, 0) + 1
        if self.on_alert:
            try:
                # `meta` carries the structured fields the three positional args
                # cannot: the PPE category, the track/global id, and which cameras
                # contributed. Without it the alert layer could only recover the
                # category by reverse-matching a Thai sentence, which no
                # cross-camera rule should ever be built on. Optional, so any
                # caller still using the 3-arg form keeps working.
                meta = {k: v for k, v in ev.items()
                        if k not in ("msg", "level", "type")}
                if isinstance(ev.get("key"), (tuple, list)) and len(ev["key"]) == 2:
                    meta.setdefault("cat", ev["key"][1])
                self.on_alert(ev.get("msg", ""), level, ev.get("type", "ppe"), meta)
            except Exception as e:
                print(f"[Pipeline] on_alert error: {e}")

    def apply_settings(self, settings: dict):
        try:
            # Config-module settings (LINE, AI thresholds, cooldowns) —
            # shared by the whole process, applied by one helper so they
            # also take effect when no camera is running.
            self._alert_levels = apply_global_settings(settings)

            # Mirroring is PER CAMERA. A site mixes a laptop webcam (a selfie
            # view, expected mirrored) with ceiling IP cameras (must not be),
            # so one global switch could only ever be wrong for half of them.
            # The legacy settings.camera.flip_horizontal is still honoured as
            # the default for cameras that have no setting of their own.
            cam = settings.get("camera", {})
            if "flip_horizontal" in cam:
                self._flip_override = bool(cam["flip_horizontal"])
            _cid = self._source_config.get("camera_id", "cam0")
            _mine = (settings.get("cameras") or {}).get(_cid)
            if isinstance(_mine, dict) and "flip_horizontal" in _mine:
                self._flip_override = bool(_mine["flip_horizontal"])
                self._source_config["flip_horizontal"] = self._flip_override

            # ── Per-camera detection roles + PPE items for the running camera ──
            cams   = settings.get("cameras", {}) or {}
            cam_id = self._source_config.get("camera_id", "cam0")
            camcfg = cams.get(cam_id) if isinstance(cams.get(cam_id), dict) else None
            if camcfg is not None:
                if "roles" in camcfg:
                    self._roles = set(camcfg["roles"] or [])
                    self._source_config["roles"] = list(self._roles)
                if "ppe_items" in camcfg:
                    self._ppe_items = list(camcfg["ppe_items"] or [])
                    self._source_config["ppe_items"] = self._ppe_items

            # ── Hot-apply to a live engine (no model reload) ──
            if self._engine is not None:
                try:
                    self._engine.apply_roles(self._roles)
                    self._engine.apply_ppe_items(self._ppe_items)
                    self._engine.refresh_tunables()
                    self._sync_module_status()
                except Exception as e:
                    print(f"[Pipeline] engine hot-apply: {e}")

            print("[Pipeline] ⚙️  Settings applied")
        except Exception as e:
            print(f"[Pipeline] apply_settings: {e}")

    # ── Private helpers ───────────────────────────────────────

    def _apply_config(self, src_cfg: dict):
        try:
            import config as cfg
            cfg.CAMERA_SOURCE   = src_cfg.get("source", "webcam")
            cfg.WEBCAM_INDEX    = int(src_cfg.get("webcam_index", 0))
            cfg.RTSP_URL        = src_cfg.get("rtsp_url", getattr(cfg, "RTSP_URL", ""))
            cfg.VIDEO_FILE_PATH = src_cfg.get("video_file_path", "")
            cfg.ZONE_POLYGON_FILE = str(_APP_DATA / "zones.json")
        except ImportError:
            pass

    def _open_camera(self, src_cfg: dict) -> cv2.VideoCapture:
        src = src_cfg.get("source", "webcam")
        if src == "webcam":
            # CAP_DSHOW is Windows-only; use the platform default elsewhere
            # (AVFoundation on macOS, V4L2 in a Linux container).
            idx = int(src_cfg.get("webcam_index", 0))
            cap = (cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32"
                   else cv2.VideoCapture(idx))
        elif src == "rtsp":
            # Bound how long a dead camera may hold this call.
            #
            # With no timeout, OpenCV/FFMPEG waits ~30 s for the stream and then
            # retries — measured at 48.5 s for one unreachable camera. The UI
            # awaits these starts, and a browser allows only ~6 connections per
            # origin, so three unreachable cameras fill the pool and every later
            # request queues behind them, including the fetch that loads the next
            # screen. The app then looks frozen: the menu is drawn but clicking
            # it does nothing for a minute or more. A camera that is actually on
            # the LAN answers in well under a second, so a few seconds is a
            # generous ceiling, not a tight one.
            try:
                import config as _cfg
                to = int(getattr(_cfg, "RTSP_OPEN_TIMEOUT_MS", 6000))
            except ImportError:
                to = 6000
            cap = cv2.VideoCapture(
                src_cfg.get("rtsp_url", ""), cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, to,
                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, to])
        elif src == "file":
            cap = cv2.VideoCapture(src_cfg.get("video_file_path", ""))
        else:
            raise ValueError(f"Unknown source: {src}")
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera (source={src})")
        # BUFFERSIZE=1: for a live feed we always want the NEWEST frame, not a
        # queue of stale ones. A larger driver buffer adds standing latency (each
        # buffered frame is ~1/fps behind) with no benefit for real-time safety.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _set_camera_state(self, state: str):
        changed = False
        with self._lock:
            if self.status.get("camera") != state:
                self.status["camera"] = state
                changed = True
            snapshot = dict(self.status)
        if changed and self.on_status:
            try:
                self.on_status(snapshot)
            except Exception as e:
                print(f"[Pipeline] on_status callback: {e}")

    def _reconnect_camera(self) -> bool:
        self._set_camera_state("reconnecting")
        if self._reader:
            try:
                self._reader.stop()
            except Exception:
                pass
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        delays  = [1.0, 2.0, 3.0, 5.0]
        attempt = 0
        while not self._stop_evt.is_set() and self._running:
            try:
                self._cap = self._open_camera(self._source_config)
                # Rebuild the reader with the SAME options the first one had.
                # Constructing it bare dropped both of them, so a source that
                # reconnected came back without its EOF-rewind and without its
                # native-fps throttle — it would then replay a file at decode
                # speed, which breaks tracking continuity.
                reader    = _FrameReader(self._cap, loop=self._reader_loop,
                                         src_fps=self._reader_fps)
                reader.start()
                self._reader = reader
                self._set_camera_state("connected")
                print("[Pipeline] 🔌 Camera reconnected")
                return True
            except Exception as e:
                wait = delays[min(attempt, len(delays) - 1)]
                attempt += 1
                print(f"[Pipeline] reconnect attempt {attempt} failed ({e}); retry in {wait}s")
                slept = 0.0
                while slept < wait and not self._stop_evt.is_set() and self._running:
                    time.sleep(0.2)
                    slept += 0.2
        return False

    def _run_remote(self, is_webcam: bool):
        """Cloud-offload loop — DECOUPLED so the video stays smooth.

        The display loop runs at full camera FPS showing LIVE frames with the
        latest cloud boxes overlaid; a separate uploader thread POSTs the newest
        frame to the cloud and stores the boxes it returns. So the video never
        waits on the network round-trip (the old design displayed the annotated
        JPEG the cloud returned → choppy + laggy). Boxes trail the video by ~one
        round-trip, which is the correct trade for smooth playback."""
        import base64
        try:
            import requests
        except Exception as e:
            print(f"[Pipeline] ❌ remote mode needs 'requests': {e}")
            self._engine_error = "requests not installed"
            return

        cid   = self._source_config.get("camera_id", "cam0")
        roles = self._source_config.get("roles")          # list ([] = nothing on)
        ppe   = self._source_config.get("ppe_items")      # list | None
        url   = self._remote["url"] + "/infer"
        headers = {"Content-Type": "application/json",
                   # RunPod's proxy can 403 a default Python UA as a bot — send a
                   # browser-like UA so frames get through reliably.
                   "User-Agent": "Mozilla/5.0 (ZENTRA)"}
        if self._remote.get("token"):
            headers["Authorization"] = "Bearer " + self._remote["token"]
        interval = 1.0 / max(1.0, self._remote.get("fps", 10.0))
        sess = requests.Session()

        # Fresh session → clear the cloud's ByteTrack ids for this camera.
        try:
            sess.post(self._remote["url"] + "/reset", json={"camera_id": cid},
                      headers=headers, timeout=5)
        except Exception:
            pass

        rset = set(roles or [])
        with self._lock:
            self.status["modules"] = {
                "ppe":  "ok" if "ppe"  in rset else "off",
                "zone": "ok" if "zone" in rset else "off",
                "fall": "ok" if "fall" in rset else "off",
            }
        self._sync_status_to_ui()
        print(f"[Pipeline] ☁️  remote inference (decoupled) → {self._remote['url']} "
              f"(camera={cid}, roles={sorted(rset) if rset else 'none'})")

        # Shared state between display + uploader.
        self._remote_draw = {"boxes": [], "zones": []}   # latest cloud result
        newest = {"pair": None}                          # (frame, id) to upload
        newest_lock = threading.Lock()

        def _uploader():
            last_sent = -1
            errors = 0
            while not self._stop_evt.is_set() and self._running:
                t0 = time.monotonic()
                with newest_lock:
                    pair = newest["pair"]
                if pair is None or pair[1] == last_sent:
                    time.sleep(0.005)
                    continue
                raw, rid = pair
                last_sent = rid
                try:
                    ok, enc = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ok:
                        continue
                    payload = {
                        "camera_id": cid, "roles": roles, "ppe_items": ppe,
                        "frame": base64.b64encode(enc.tobytes()).decode("ascii"),
                        "draw": False,    # boxes only — the edge draws them itself
                    }
                    r = sess.post(url, json=payload, headers=headers, timeout=15)
                    data = r.json()
                    if data.get("ok"):
                        self._remote_draw = {"boxes": data.get("boxes", []),
                                             "zones": data.get("zones", [])}
                        for ev in data.get("events", []):
                            self._emit_event(ev)
                        errors = 0
                    else:
                        errors += 1
                        if errors == 1 or errors % 60 == 0:
                            print(f"[Pipeline] ☁️ cloud error: {data.get('error')}")
                except Exception as e:
                    errors += 1
                    if errors == 1 or errors % 60 == 0:
                        print(f"[Pipeline] ☁️ cloud unreachable ({e}) — video still live")
                dt = interval - (time.monotonic() - t0)
                if dt > 0:
                    time.sleep(dt)

        up = threading.Thread(target=_uploader, daemon=True, name="RemoteUploader")
        up.start()

        # DISPLAY loop — full camera FPS, overlays the latest cloud boxes.
        frame_id = 0
        read_failures = 0
        while not self._stop_evt.is_set() and self._running:
            ret, raw = (self._reader.read() if self._reader else (False, None))
            if not ret or raw is None:
                read_failures += 1
                if read_failures > 40 and self._source_config.get("source") != "file":
                    print("[Pipeline] ⚠️  Camera signal lost — reconnecting")
                    if not self._reconnect_camera():
                        break
                    read_failures = 0
                continue
            read_failures = 0
            frame_id += 1
            if self._should_flip(is_webcam):
                raw = cv2.flip(raw, 1)

            # Hand the PRISTINE frame to the uploader (drop-to-newest).
            with newest_lock:
                newest["pair"] = (raw, frame_id)

            # Draw the latest known boxes on a COPY for display (fast, no network).
            try:
                out = _draw_remote_overlay(raw.copy(), self._remote_draw)
            except Exception as e:
                print(f"[Pipeline] overlay error: {e}")
                out = raw
            with self._frame_lock:
                self._latest_frame = out

            if frame_id % 150 == 0:
                with self._lock:
                    self.status["uptime_seconds"] = self.get_uptime()

        if up.is_alive():
            up.join(timeout=3.0)
        self._running = False
        with self._lock:
            self.status["running"] = False
        self._set_camera_state("disconnected")
        print("[Pipeline] ☁️  remote loop ended")

    def _sync_status_to_ui(self):
        with self._lock:
            snapshot = dict(self.status)
        if self.on_status:
            try:
                self.on_status(snapshot)
            except Exception:
                pass

    # ── SHARED-MODEL edge multi-camera ────────────────────────
    def _get_shared_frame(self):
        """Called by the central scheduler each tick — hand it the latest pristine
        frame and remember the WHOLE packet so the fall pairing uses the SAME frame
        the scheduler processed (not a newer one the capture loop has since made).

        Returns the bare frame because that is the scheduler's batching contract;
        the (frame_id, capture time) that travel with it are kept here."""
        pkt = self._shared_pkt              # one read → frame/id/time always agree
        self._last_handed_pkt = pkt
        return pkt[0] if pkt is not None else None

    def _on_shared_result(self, recs, events):
        """Scheduler → this camera: store boxes for the display loop, pair them with
        the processed frame for the fall loop, emit events, and feed the collector.
        Runs on the SCHEDULER thread; assignments are atomic ref swaps."""
        self._latest_recs = recs
        pkt = self._last_handed_pkt
        if pkt is None:
            return
        frame, fid, t_cap = pkt
        # frame_id comes from CAPTURE, not from a counter bumped once per result.
        # A result-side counter changes even when the scheduler re-processes the
        # very same frame (capture stalled, file EOF, reconnect) — the fall loop's
        # freshness check then reads it as a new frame and appends a duplicate to
        # its 30-frame motion window, stretching a fast fall into a slow one.
        self._recs_pair = (frame, recs, fid)
        self._last_result_t = t_cap
        for ev in events:
            self._emit_event(ev)
        # Training-data collection (same as the single-cam detect loop).
        try:
            if frame is not None and self._engine is not None:
                self._collect(frame, self._engine.last_dets, events)
        except Exception:
            pass

    def _run_shared(self, is_webcam: bool, is_file: bool):
        """Capture + display only; the shared scheduler runs inference. Fall stays
        per-camera on its own loop. On cloud/hub failure falls back to passthrough."""
        self._engine = self._build_engine()          # built with hub → CameraEngine
        cid = self._source_config.get("camera_id", "cam0")
        if self._engine is not None:
            self._sync_module_status()
            self._latest_recs = []
            # Register with the central scheduler (it batches this camera with the
            # others against the shared models).
            self._shared_scheduler.register(
                cid, self._engine, self._get_shared_frame, self._on_shared_result)
            # Fall is per-camera: its own fixed-cadence loop reads _recs_pair.
            self._fall_thr = threading.Thread(
                target=self._fall_loop, daemon=True, name="FallLoop")
            self._fall_thr.start()
            print(f"[Pipeline] 🧩 shared-model mode (camera={cid}) — "
                  f"registered with scheduler")
        else:
            print(f"[Pipeline] ⚠️ shared engine unavailable → passthrough ({cid})")

        frame_id = 0
        read_failures = 0
        try:
            while not self._stop_evt.is_set() and self._running:
                ret, raw = (self._reader.read() if self._reader else (False, None))
                if not ret or raw is None:
                    read_failures += 1
                    if read_failures > 40 and not is_file:
                        print("[Pipeline] ⚠️  Camera signal lost — reconnecting")
                        if not self._reconnect_camera():
                            break
                        read_failures = 0
                    continue
                read_failures = 0
                frame_id += 1
                if self._should_flip(is_webcam):
                    raw = cv2.flip(raw, 1)

                # Publish the pristine frame + its identity for the scheduler.
                # ONE tuple: two fields could not be read atomically, and a frame
                # landing between the two reads would pair this frame with the
                # next one's id/timestamp.
                self._shared_pkt = (raw, frame_id, time.monotonic())

                # DISPLAY: draw the latest boxes (set by the scheduler) on a COPY,
                # at full capture FPS — decoupled from inference cadence.
                out = raw
                if self._engine is not None:
                    try:
                        out = self._engine.draw_on(raw.copy(), self._latest_recs)
                    except Exception as e:
                        print(f"[Pipeline] draw_on error: {e}")
                        out = raw
                with self._frame_lock:
                    self._latest_frame = out

                if frame_id % 150 == 0:
                    with self._lock:
                        self.status["uptime_seconds"] = self.get_uptime()
        except Exception:
            print("[Pipeline] ❌ shared loop crashed:")
            traceback.print_exc()
        finally:
            try:
                self._shared_scheduler.unregister(cid)
            except Exception:
                pass
            self._running = False
            with self._lock:
                self.status["running"] = False
            self._set_camera_state("disconnected")
            print(f"[Pipeline] 🧩 shared loop ended ({cid})")

    def _process_loop(self):
        try:
            # Source type comes from THIS pipeline's own config, never from the
            # global cfg.CAMERA_SOURCE. _apply_config() writes that global on every
            # start(), so with several cameras the LAST one to start decided the
            # source type for ALL of them — a webcam started after a file made the
            # file pipeline stop throttling to native fps and start mirroring
            # itself. Per-camera state must come from per-camera config.
            src       = self._source_config.get("source", "webcam")
            is_file   = (src == "file")
            is_webcam = (src == "webcam")

            # File → throttle to the clip's native fps (time-ordered stream);
            # live camera self-paces (src_fps=0 = no throttle).
            src_fps = 0.0
            if is_file:
                try:
                    src_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
                    if src_fps <= 0 or src_fps > 120:
                        src_fps = 25.0
                except Exception:
                    src_fps = 25.0
            # Remembered so _reconnect_camera() can rebuild an identical reader.
            self._reader_loop = is_file
            self._reader_fps  = src_fps
            reader = _FrameReader(self._cap, loop=is_file, src_fps=src_fps)
            reader.start()
            self._reader = reader

            # REMOTE MODE: heavy inference runs on the cloud GPU. Capture → POST →
            # display the annotated frame it returns. No local engine, no detect/
            # fall threads.
            if self._remote:
                self._run_remote(is_webcam)
                return

            # SHARED MODE: models shared via ModelHub; a central scheduler drives
            # inference for all cameras. This pipeline only captures + displays
            # (+ per-camera fall). VRAM stays constant as cameras are added.
            if self._shared_hub is not None and self._shared_scheduler is not None:
                self._run_shared(is_webcam, is_file)
                return

            # Build the detection engine (falls back to passthrough on failure)
            self._engine = self._build_engine()
            if self._engine is not None:
                self._sync_module_status()
                self._latest_recs = []
                self._detect_thr = threading.Thread(
                    target=self._detect_loop, daemon=True, name="DetectLoop")
                self._detect_thr.start()
                # Always started: it idles while the fall role is off, so toggling
                # the role in Settings takes effect without restarting the camera.
                self._fall_thr = threading.Thread(
                    target=self._fall_loop, daemon=True, name="FallLoop")
                self._fall_thr.start()

            frame_id      = 0
            read_failures = 0
            mode = "detection (decoupled)" if self._engine else "passthrough"
            print(f"[Pipeline] ▶️  Process loop running ({mode})")

            while not self._stop_evt.is_set() and self._running:
                ret, raw = (self._reader.read() if self._reader else (False, None))
                if not ret or raw is None:
                    read_failures += 1
                    if read_failures > 40 and not is_file:
                        print("[Pipeline] ⚠️  Camera signal lost — reconnecting")
                        if not self._reconnect_camera():
                            break
                        read_failures = 0
                    continue
                read_failures = 0

                frame_id += 1
                # Mirroring defaults to on for a webcam (selfie view) and off for
                # everything else, but an explicit Settings choice wins for any
                # source — see _should_flip().
                if self._should_flip(is_webcam):
                    raw = cv2.flip(raw, 1)

                # publish newest raw for the detection worker (skips old frames).
                # MUST stay pristine: draw_on() annotates in place, so the display
                # draws on a COPY — otherwise the detector would run on a frame with
                # boxes already painted on it and fail to detect anyone.
                self._raw_pair = (raw, frame_id, time.monotonic())

                # DISPLAY = fast: draw the latest known boxes onto a copy of the frame
                out = raw
                if self._engine is not None:
                    try:
                        out = self._engine.draw_on(raw.copy(), self._latest_recs)
                    except Exception as e:
                        print(f"[Pipeline] draw_on error: {e}")
                        out = raw

                with self._frame_lock:
                    self._latest_frame = out

                if frame_id % 150 == 0:
                    with self._lock:
                        self.status["uptime_seconds"] = self.get_uptime()

        except Exception:
            print("[Pipeline] ❌ Process loop crashed:")
            traceback.print_exc()
        finally:
            self._running = False
            with self._lock:
                self.status["running"] = False
            self._set_camera_state("disconnected")
            print("[Pipeline] Process loop ended")
