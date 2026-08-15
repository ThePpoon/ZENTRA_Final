"""
server/api.py — ZENTRA FastAPI Server (Stage B — Real Pipeline)
REST + SSE + WebSocket endpoints backed by the real AI pipeline.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import mimetypes
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Bundled web fonts: Python's mimetypes doesn't know woff2 → StaticFiles would
# serve it with a generic type. Register the correct type so the browser accepts
# the locally-hosted Kanit font without complaint.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

# ── Path setup ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
UI_DIR   = BASE_DIR / "ui"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Backend (AI) project + its auto-collected data. Single-repo: the backend lives
# at <repo>/backend. This used to be BASE_DIR.parent/"ZENTRA" — a leftover from
# when ZENTRA_application/ sat beside ZENTRA/. It resolved to the repo root by
# coincidence (repo is named ZENTRA), so /api/data/stats and /api/training/metrics
# silently pointed at directories that never existed and always returned empty.
ZENTRA_BACKEND = BASE_DIR / "backend"
COLLECTED_DIR  = ZENTRA_BACKEND / "data" / "collected"
_DATA_CATEGORIES = ["ppe_violations", "zone_intrusions", "fall_events", "normal"]

# ── Optional API auth ───────────────────────────────────────
# The API is unauthenticated by default and is bound to localhost for exactly that
# reason (see README). Set ZENTRA_API_TOKEN to require a shared
# secret on every /api and /ws call, so the app can sit behind a LAN / reverse
# proxy without exposing worker evidence photos, the event log, or the training
# subprocess spawner to anyone who can reach the port. Empty (default) = no auth,
# behaviour unchanged.
_API_TOKEN = os.getenv("ZENTRA_API_TOKEN", "").strip()


def _token_ok(provided: str | None) -> bool:
    """Constant-time token check. Always True when no token is configured.
    Compares UTF-8 bytes so a non-ASCII token (Thai / emoji) can't raise the
    TypeError that hmac.compare_digest throws on non-ASCII str."""
    if not _API_TOKEN:
        return True
    return bool(provided) and hmac.compare_digest(
        provided.encode("utf-8"), _API_TOKEN.encode("utf-8"))


# ── App lifespan (replaces the deprecated @app.on_event startup/shutdown) ────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(title="ZENTRA API", docs_url=None, redoc_url=None, lifespan=lifespan)

# NO CORS MIDDLEWARE — deliberately.
#
# The UI is served by this same app, so every request it makes is same-origin and
# needs no CORS headers at all. The previous `allow_origins=["*"]` did nothing for
# the app and everything for an attacker: because the API has no authentication,
# `Access-Control-Allow-Origin: *` let ANY web page the operator happened to visit
# read http://localhost:7788/api/settings (which returns the LINE channel access
# token) and POST /api/zones (whose zone name lands in the History page).
#
# If you ever need a browser on another origin to call this API, add authentication
# FIRST, then allow that exact origin — never "*".


@app.middleware("http")
async def _auth_gate(request, call_next):
    """Gate /api/* behind ZENTRA_API_TOKEN when it is set. The static UI shell
    (/, /ui/*) stays open — it holds no secrets — so the app still loads and can
    then attach the token to its calls. No token configured → every request passes
    and behaviour is identical to before."""
    if _API_TOKEN and request.url.path.startswith("/api"):
        auth = request.headers.get("Authorization", "")
        tok = auth[7:] if auth.startswith("Bearer ") else request.query_params.get("token")
        if not _token_ok(tok):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


class _NoCacheStatic(StaticFiles):
    """Serve UI assets with Cache-Control: no-cache so browsers always revalidate
    (ETag → cheap 304 when unchanged). Prevents the SPA from showing stale
    screens/JS after an edit without a hard refresh."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app.mount("/ui", _NoCacheStatic(directory=str(UI_DIR)), name="ui")

# ── Globals (set on startup) ─────────────────────────────────
_loop:       asyncio.AbstractEventLoop | None = None
manager      = None   # MultiCameraManager (runs N cameras at once)
_retention_task = None   # periodic PDPA purge task


def _primary_pipeline():
    """Back-compat handle: the first running pipeline, for the few endpoints that
    still speak single-camera (legacy /api/status fallback, snapshot). None when
    nothing is running."""
    return manager.primary() if manager is not None else None

# LINE pushes are blocking HTTP with retries. They must never run on the
# detect/fall loop (that thread has a frame budget) nor on the event loop.
# ONE worker → pushes are serialised, so a burst can't open N sockets at once.
_line_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="line-push")


def _camera_label(cam_id: str = "") -> str:
    """Human label for the camera that produced an event (for History rows).

    With multi-camera the id is passed in explicitly by the alert callback. A
    human name from settings (`cameras.<id>.name`) wins when the operator has set
    one; otherwise the id itself is used, which is at least true. Falls back to
    "unknown" rather than inventing a camera.
    """
    if not cam_id:
        return "unknown"
    try:
        cam = (_load_settings().get("cameras") or {}).get(cam_id)
        if isinstance(cam, dict) and str(cam.get("name", "")).strip():
            return str(cam["name"]).strip()
    except Exception:
        pass
    return cam_id


def _push_line(event_id: int, msg: str, level: str, ev_type: str, frame) -> None:
    """Send one alert to LINE and record the real outcome against the event.

    Returns quietly when LINE is unconfigured (no token / no group): that is the
    normal offline state, not an error, and the event stays in History either way.
    """
    from server import store
    try:
        from alerts.line_notify import send_line_notify
    except Exception as e:
        print(f"[API] LINE unavailable: {e}")
        return
    try:
        # The engine already gates every alert through its own per-track confirm
        # window + CooldownGate, so line_notify must NOT gate a second time — its
        # default key is global and would swallow unrelated alerts.
        ok = send_line_notify(msg, image=frame, level=level,
                              cooldown_key=f"{ev_type}:{msg}", cooldown_sec=0,
                              async_send=False)
        store.mark_line_sent(event_id, bool(ok))
    except Exception as e:
        print(f"[API] LINE push failed for event {event_id}: {e}")


# ================================================================
# WEBSOCKET MANAGER
# ================================================================
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


connections = ConnectionManager()   # WebSocket clients (renamed from `manager` to
                                     # free that name for the MultiCameraManager)


# Events are persisted locally via server/store.py (SQLite + snapshot files);
# nothing is kept only in memory and nothing leaves the device (PDPA).


# ================================================================
# STARTUP / SHUTDOWN
# ================================================================
# PDPA data minimisation runs at startup AND on this timer. The startup purge
# alone is not enough: a factory box runs for weeks, and without a periodic sweep
# it would never drop events/snapshots that age past retention_days until the next
# restart — so old worker evidence would pile up indefinitely. 6h keeps disk and
# retention honest at negligible cost (one indexed DELETE).
_RETENTION_INTERVAL_SEC = 6 * 3600


def _run_retention_purge() -> None:
    """Read retention_days from settings and drop older events. Safe to call
    repeatedly (idempotent — deletes anything past the day cutoff)."""
    rdays = int((_load_settings().get("data") or {}).get("retention_days", 0) or 0)
    if rdays <= 0:
        return
    from server import store
    removed = store.purge_before(rdays)
    if removed:
        print(f"[API] PDPA retention: purged {removed} event(s) older than {rdays} day(s)")


async def _retention_loop():
    """Re-run the retention purge every _RETENTION_INTERVAL_SEC so a long-running
    instance keeps honouring retention_days without needing a restart."""
    while True:
        try:
            await asyncio.sleep(_RETENTION_INTERVAL_SEC)
            await _in_executor(_run_retention_purge)   # off the event loop (SQLite)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[API] retention loop error: {e}")


async def _startup():
    global _loop, manager, _retention_task

    _loop = asyncio.get_running_loop()

    if _API_TOKEN:
        print("[API] 🔒 Token auth ENABLED (ZENTRA_API_TOKEN set).")
    else:
        print("[API] ⚠️  API is UNAUTHENTICATED — keep it bound to 127.0.0.1 only "
              "(set ZENTRA_API_TOKEN to require a token behind a proxy).")

    # Host power state. Measured on this app: the SAME inference code runs 229 ms
    # per 4-camera tick on AC and 1051 ms while the dGPU is power-limited — 4.6x,
    # more than any optimisation in the pipeline. A box left on battery presents
    # as "the AI is slow / cameras freeze", so it is reported at startup rather
    # than left to be rediscovered by benchmarking the wrong layer.
    try:
        from server.power import log_power_status
        log_power_status()
    except Exception as e:
        print(f"[Power] check skipped: {e}")

    # PDPA data minimisation: drop events + evidence snapshots older than the
    # configured retention window (local only). Once now, then every 6h via
    # _retention_loop so a long-running instance keeps purging.
    try:
        _run_retention_purge()
    except Exception as e:
        print(f"[API] retention purge skipped: {e}")
    _retention_task = asyncio.create_task(_retention_loop())

    try:
        # Import the multi-camera manager (adds ZENTRA backend to sys.path via the
        # pipeline import, brings in cv2/numpy).
        from pipeline.manager import MultiCameraManager

        _stream_fps = int(os.getenv("STREAM_FPS", "15"))
        manager = MultiCameraManager(ws_manager=connections, loop=_loop,
                                     stream_fps=_stream_fps)
        # Honour Settings → display. Every camera streams over the SAME WebSocket,
        # so these are per-camera costs multiplied by the camera count; on a
        # multi-camera box they are the knobs that keep the socket drainable.
        manager.apply_stream_settings(_load_settings().get("display") or {})

        # Wire alert callback → WebSocket broadcast + history + LINE push. The
        # manager passes the ORIGINATING camera_id so every event is traceable to
        # its place (History rows, grid tile flashing).
        def _on_alert(camera_id: str, msg: str, level: str, ev_type: str = "ppe",
                      meta: dict | None = None):
            # `meta` carries the structured detail behind the Thai message
            # (category, track id, ...). Nothing consumes it today; it is kept
            # because the alert text is a poor thing to parse later.
            meta = meta or {}
            # Capture an evidence snapshot (local only) of THAT camera's frame.
            snap = frame = None
            try:
                frame = manager.get_latest_frame(camera_id)
                snap  = manager.get_snapshot(camera_id)
            except Exception:
                snap = frame = None
            from server import store
            camera = _camera_label(camera_id)
            # Persist FIRST: evidence must survive a LINE/network failure. The
            # line_sent flag is corrected by _push_line once the push resolves.
            event = store.add_event(level=level, message=msg, camera=camera,
                                    frame_jpeg=snap, line_sent=False, type_=ev_type)
            _line_pool.submit(_push_line, event["id"], msg, level, ev_type, frame)
            broadcast_msg = {
                "type":         "event",
                "event":        "alert",
                "id":           event["id"],
                "level":        level,
                "kind":         ev_type,
                "message":      event["message"],
                "timestamp":    event["time"],
                "camera":       camera,
                "camera_id":    camera_id,
                # Aggregate counts across ALL cameras (dashboard shows the total).
                "alerts":       _aggregate_alerts(),
                "has_snapshot": event["has_snapshot"],
            }
            asyncio.run_coroutine_threadsafe(
                connections.broadcast(broadcast_msg), _loop
            )

        manager.on_alert = _on_alert

        # Wire status changes (camera connect/reconnect/disconnect) → WebSocket.
        # camera_id lets the grid update just that tile.
        def _on_status(camera_id: str, status: dict):
            asyncio.run_coroutine_threadsafe(
                connections.broadcast({
                    "type":      "event",
                    "event":     "status",
                    "camera_id": camera_id,
                    "modules":   status.get("modules", {}),
                    "alerts":    _aggregate_alerts(),
                    "camera":    status.get("camera", "disconnected"),
                    "running":   status.get("running", False),
                    # Non-null → the AI engine failed to load and NOTHING is being
                    # detected. The UI must say so loudly; clean video with no boxes
                    # otherwise reads as "no violations".
                    "engine_error": status.get("engine_error"),
                }),
                _loop,
            )

        manager.on_status = _on_status

        # Store saved settings so cameras started later inherit them (LINE tokens,
        # AI thresholds, per-camera roles). No camera runs until /api/pipeline/start.
        try:
            manager.apply_settings(_load_settings())
        except Exception as e:
            print(f"[API] settings preload skipped: {e}")

        print("[API] Startup complete ✅ (multi-camera manager ready)")

    except Exception as e:
        print(f"[API] ⚠️  Startup warning (manager not loaded): {e}")
        print("[API] Server running in UI-only mode")


def _aggregate_alerts() -> dict:
    """Sum the live per-session alert counters across every running camera, so the
    dashboard KPIs reflect the whole site, not one camera."""
    total = {"total": 0, "warning": 0, "alert": 0, "emergency": 0}
    if manager is None:
        return total
    for s in manager.statuses().values():
        a = s.get("alerts", {}) or {}
        for k in total:
            total[k] += int(a.get(k, 0) or 0)
    return total


async def _shutdown():
    global _retention_task
    if _retention_task:
        _retention_task.cancel()
        _retention_task = None
    if manager is not None:
        manager.stop_all()
    # Let any in-flight LINE push finish so an alert isn't lost on shutdown.
    _line_pool.shutdown(wait=True)
    print("[API] Shutdown complete")


# ================================================================
# STATIC / ROOT
# ================================================================
@app.get("/")
async def root():
    return FileResponse(str(UI_DIR / "index.html"))


# ================================================================
# SPLASH — SSE init progress
# ================================================================
@app.get("/api/init")
async def init_stream():
    steps = [
        (15,  "กำลังโหลดการตั้งค่า..."),
        (35,  "เริ่มต้นเครื่องตรวจจับ AI (ประมวลผลในเครื่อง)..."),
        (55,  "เตรียมโมดูล PPE / โซน / การล้ม..."),
        (75,  "โหลดข้อมูลโซนความปลอดภัย..."),
        (90,  "เริ่มต้นระบบ..."),
        (100, "พร้อมใช้งาน"),
    ]

    async def _gen():
        for pct, msg in steps:
            yield f"data: {json.dumps({'percent': pct, 'message': msg})}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ================================================================
# STATUS
# ================================================================
def _power_field() -> dict:
    """Host power state for /api/status. Cached briefly: the plan lookup shells
    out to powercfg and the dashboard polls this endpoint on a timer."""
    global _power_cache, _power_cache_at
    now = time.monotonic()
    if _power_cache is None or now - _power_cache_at > 30.0:
        try:
            from server.power import power_status
            _power_cache = power_status()
        except Exception:
            _power_cache = {"ok": True, "warnings": []}
        _power_cache_at = now
    return _power_cache


_power_cache: dict | None = None
_power_cache_at: float = 0.0


@app.get("/api/status")
async def status():
    """Legacy single-camera status: reports the PRIMARY (first running) camera's
    modules/camera-state, with SITE-WIDE aggregate alert counts + uptime. The
    dashboard reads this; the CCTV grid reads /api/cameras/status for per-camera
    detail."""
    if manager is None:
        return JSONResponse({
            "running": False, "source": None,
            "modules": {"ppe": "error", "zone": "error", "fall": "error"},
            "alerts":  {"total": 0, "warning": 0, "alert": 0, "emergency": 0},
            "uptime":  0, "last_emergency": None,
            "engine_error": "pipeline not initialised",
            "cameras_active": 0, "cameras_total": 0,
            "power": _power_field(),
        })
    prim = manager.primary()
    if prim is None:
        return JSONResponse({
            "running": False, "source": None,
            "modules": {"ppe": "standby", "zone": "standby", "fall": "standby"},
            "alerts":  _aggregate_alerts(),
            "uptime":  0, "last_emergency": None, "engine_error": None,
            "device":  os.getenv("PPE_INFER_DEVICE", "cpu"),
            "cameras_active": 0,
            "cameras_total": len(_load_settings().get("cameras") or {}),
            "power": _power_field(),
        })
    with prim._lock:
        s = dict(prim.status)
    s["alerts"] = _aggregate_alerts()
    s["uptime"] = prim.get_uptime()
    s["cameras_active"] = len(manager.active_ids())
    # Total CONFIGURED cameras, so the dashboard can show "2/3" instead of the
    # hardcoded "/1" it used to print — which read as "one camera exists" on a
    # site running three.
    s["cameras_total"] = len(_load_settings().get("cameras") or {})
    eng = getattr(prim, "_engine", None)
    s["device"] = (getattr(eng.detector, "device", None) if eng is not None
                   else os.getenv("PPE_INFER_DEVICE", "cpu"))
    # Surfaced so a throttled host can never be the silent explanation for slow
    # detection — the UI can show it next to the device readout.
    s["power"] = _power_field()
    return JSONResponse(s)


@app.post("/api/cloud/test")
async def cloud_test(body: dict[str, Any]):
    """Server-side health check of a cloud inference server (avoids browser CORS
    and keeps the token off the page's network log). Returns the /health JSON."""
    url = str(body.get("url", "")).strip().rstrip("/")
    token = str(body.get("token", "")).strip()
    if not url:
        return JSONResponse({"ok": False, "error": "ยังไม่ได้ใส่ URL"}, status_code=400)

    def _ping():
        import urllib.request
        req = urllib.request.Request(url + "/health")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        # RunPod's proxy (Cloudflare) 403s the default "Python-urllib" User-Agent
        # as a bot. Send a browser UA so the health check gets through, matching
        # what the requests-based frame POST already does.
        req.add_header("User-Agent", "Mozilla/5.0 (ZENTRA)")
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode())

    try:
        data = await _in_executor(_ping)
        return JSONResponse({"ok": True, "health": data})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.get("/api/cameras/status")
async def cameras_status():
    """Per-camera status snapshots keyed by camera_id — the CCTV/NVR grid polls
    this to light each tile (online/offline, modules, engine_error) and to know
    which cameras are live."""
    if manager is None:
        return JSONResponse({"cameras": {}, "active": []})
    return JSONResponse({"cameras": manager.statuses(),
                         "active": manager.active_ids()})


# ================================================================
# PIPELINE  start / stop
# ================================================================
@app.post("/api/pipeline/start")
async def pipeline_start(body: dict[str, Any]):
    if manager is None:
        return JSONResponse({"ok": False, "error": "pipeline not initialised"}, status_code=503)

    source    = body.get("source", "webcam")
    camera_id = body.get("camera_id", "cam0")
    # Per-camera detection roles + PPE items to enforce from settings.
    #
    # Default to [] (nothing on), NOT None. Down in PPEEngine, roles=None still
    # means "all modules on" — that is the standalone/dev convenience — but the
    # app must never hand it None, because a camera the operator has not
    # configured yet would then silently run PPE + zone detection while the
    # Cameras page, which reads the same settings, shows 0 active roles. The two
    # screens disagreed and it read as a bug: the Dashboard lit up green while
    # no model appeared to be selected anywhere. Nothing runs until the operator
    # picks roles in the Cameras page.
    # (ppe_items stays None = enforce cfg.PPE_REQUIRED, which is a real default,
    # not an unconfigured state.)
    roles = []
    ppe_items = None
    try:
        cams = (_load_settings().get("cameras") or {})
        c = cams.get(camera_id)
        if isinstance(c, dict):
            if isinstance(c.get("roles"), list):
                roles = c["roles"]
            if isinstance(c.get("ppe_items"), list):
                ppe_items = c["ppe_items"]
    except Exception:
        roles = []
    # Cloud offload config (shared by all cameras): when enabled the pipeline runs
    # in remote mode (POST frames to the cloud GPU) instead of local inference.
    cloud = {}
    try:
        cloud = (_load_settings().get("cloud") or {})
    except Exception:
        cloud = {}

    # WHERE the stream lives: the request may say, but the camera registry is the
    # authority. Falling back to it means a caller that knows only the camera_id
    # — which is all the registry-driven screens actually have — starts the right
    # stream instead of opening an empty URL and reporting "cannot open camera".
    reg = {}
    try:
        reg = ((_load_settings().get("cameras") or {}).get(camera_id)) or {}
    except Exception:
        reg = {}
    if not body.get("source") and reg.get("source"):
        source = reg["source"]

    src_cfg = {
        "source":          source,
        "webcam_index":    int(body.get("webcam_index", reg.get("webcam_index", 0)) or 0),
        "rtsp_url":        body.get("rtsp_url") or reg.get("rtsp_url", ""),
        "video_file_path": body.get("video_file_path") or reg.get("video_file_path", ""),
        "camera_id":       camera_id,
        "roles":           roles,
        "ppe_items":       ppe_items,
        # Per-camera mirror. None = this camera has no setting of its own, and
        # the pipeline falls back to the global default (and then to "webcam
        # only", which is the sensible default for a selfie-view laptop cam).
        "flip_horizontal": reg.get("flip_horizontal"),
        "cloud_enabled":   bool(cloud.get("enabled")),
        "cloud_url":       cloud.get("url", ""),
        "cloud_token":     cloud.get("token", ""),
        "cloud_fps":       cloud.get("fps", 10),
    }

    # Run blocking start() in thread pool so we don't block the event loop
    loop   = asyncio.get_running_loop()
    ok     = await loop.run_in_executor(None, manager.start, camera_id, src_cfg)
    if not ok:
        return JSONResponse({"ok": False, "error": "ไม่สามารถเปิดกล้อง / แหล่งภาพได้"}, status_code=400)

    return JSONResponse({"ok": True, "source": source, "camera_id": camera_id})


@app.post("/api/pipeline/stop")
async def pipeline_stop(body: dict[str, Any] | None = None):
    """Stop one camera (body {"camera_id": ...}) or ALL cameras when no id given."""
    if manager is None:
        return JSONResponse({"ok": True})
    body = body or {}
    camera_id = body.get("camera_id")
    loop = asyncio.get_running_loop()
    if camera_id:
        await loop.run_in_executor(None, manager.stop, camera_id)
    else:
        await loop.run_in_executor(None, manager.stop_all)
    return JSONResponse({"ok": True})


# ================================================================
# ZONES
# ================================================================
ZONES_FILE = DATA_DIR / "zones.json"


def _load_zones() -> list:
    if ZONES_FILE.exists():
        return json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    return []


def _save_zones(zones: list) -> None:
    ZONES_FILE.write_text(json.dumps(zones, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_zone_role(camera_id: str) -> bool:
    """Turn the 'zone' role ON for the camera a zone was just drawn on.

    Drawing a danger zone for a camera IS the request to watch it, but the role
    lives in a different screen (กล้อง → ⚙). Without this the zone is saved,
    loaded by the engine, and then ignored: `zone_enabled` is False so the
    polygon is never drawn and intrusion is never tested. The operator sees an
    empty picture and a success toast — the exact "I drew it and nothing shows"
    report. Returns True when a role was actually added.
    """
    if not camera_id:
        return False
    try:
        s = _load_settings()
        cams = s.get("cameras") or {}
        cam = cams.get(camera_id)
        if not isinstance(cam, dict):
            return False        # zone for a camera not in the registry — leave it
        roles = list(cam.get("roles") or [])
        if "zone" in roles:
            return False
        roles.append("zone")
        cam["roles"] = roles
        cams[camera_id] = cam
        s["cameras"] = cams
        _save_settings(s)
        if manager is not None:
            manager.apply_settings(s)     # hot-applies to the running pipeline
        print(f"[Zones] เปิดบทบาท 'เฝ้าพื้นที่อันตราย' ให้ {camera_id} อัตโนมัติ "
              "(วาดพื้นที่ = ต้องการให้เฝ้า)")
        return True
    except Exception as e:
        print(f"[Zones] ensure zone role ({camera_id}): {e}")
        return False


@app.get("/api/zones")
async def get_zones():
    return JSONResponse(_load_zones())


@app.post("/api/zones")
async def create_zone(body: dict[str, Any]):
    zones = _load_zones()
    zone  = {
        "id":        max((z.get("id", 0) for z in zones), default=0) + 1,
        "name":      body.get("name", f"Zone {len(zones) + 1}"),
        "color":     body.get("color", "#ef4444"),
        "points":    body.get("points", []),
        "type":      body.get("type", "danger"),   # danger (detect) | exclusion (ignore)
        "camera_id": body.get("camera_id", "cam0"),  # which camera this zone belongs to
        "enabled":   True,
    }
    zones.append(zone)
    _save_zones(zones)
    role_added = _ensure_zone_role(zone["camera_id"])
    if manager:
        manager.reload_zones()
    return JSONResponse({**zone, "role_enabled": role_added})


@app.put("/api/zones/{zone_id}")
async def update_zone(zone_id: int, body: dict[str, Any]):
    zones = _load_zones()
    for z in zones:
        if z.get("id") == zone_id:
            z.update({k: v for k, v in body.items() if k != "id"})
            _save_zones(zones)
            role_added = _ensure_zone_role(z.get("camera_id") or "")
            if manager:
                manager.reload_zones()
            return JSONResponse({**z, "role_enabled": role_added})
    return JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: int):
    zones = [z for z in _load_zones() if z.get("id") != zone_id]
    _save_zones(zones)
    if manager:
        manager.reload_zones()
    return JSONResponse({"ok": True})


# ================================================================
# SETTINGS
# ================================================================
SETTINGS_FILE = DATA_DIR / "settings.json"

SETTINGS_DEFAULTS: dict[str, Any] = {
    "line": {
        "channel_access_token": "",
        # All enabled groups receive every alert regardless of level. Each group:
        # {name, id, cooldown, enabled}. Add/remove groups from Settings → LINE.
        "groups": [
            {"name": "กลุ่มหลัก", "id": "", "cooldown": 30, "enabled": True},
        ],
    },
    "ai": {
        # Person-centric engine needs decent person recall; 0.30 matches the
        # deployed default. Higher values (0.70) filtered detections out entirely.
        "ppe_confidence": 0.30,
        # Minimum evidence weight before a camera may raise a violation. A view
        # that cannot see the item well enough abstains instead of guessing —
        # measured on site footage, a rear view scores 0.12 for a vest against
        # 0.71 from the front. 0 disables the gate entirely.
        "abstain_weight": 0.35,
        "fall_bbox_ratio": 0.72,
        "fall_confirm_frames": 6,
        "fall_mode": "hybrid",          # hybrid | yolo | pose
    },
    "alerts": {
        "violation_cooldown_seconds": 30,
        "zone_cooldown_seconds": 20,
        "fall_cooldown_seconds": 15,
        "warning_enabled": True,
        "alert_enabled":   True,
        "emergency_enabled": True,
        # PDPA: OFF by default — LINE alerts are text-only, no image leaves the
        # device. Opt-in sends evidence photos via an external public host.
        "upload_images":   False,
    },
    "camera": {
        "source": "webcam",
        "webcam_index": 0,
        "rtsp_url": "",
        "video_file_path": "",
        "flip_horizontal": True,
    },
    # Live-video stream (WebSocket) tuning. These are PER CAMERA and every camera
    # shares one WebSocket + one WebView, so the cost is N x this. On a 4-camera
    # box, 960x540 @ 15fps q70 is ~6 MB/s of base64 — enough to back the socket up
    # and freeze every tile. Lower fps/size first if the grid stutters.
    "display": {
        "stream_fps": 10,
        "stream_width": 960,
        "stream_height": 540,
        "stream_jpeg_quality": 70,
    },
    # Cloud offload: when enabled, the edge POSTs frames to a cloud GPU inference
    # server (cloud/inference_server.py on RunPod / GCP L4) instead of running the
    # models locally — the notebook stays smooth and many cameras run at once.
    "cloud": {
        "enabled": False,
        "url": "",        # e.g. https://<pod-id>-8000.proxy.runpod.net
        "token": "",      # must match the server's ZENTRA_API_TOKEN
        "fps": 10,        # frames/sec sent to the cloud (inference cadence)
    },
    "data": {
        # PDPA data minimisation: auto-delete events + snapshots older than N days
        # on startup. 0 = keep forever.
        "retention_days": 90,
    },
    # Per-camera detection roles, e.g. {"cam0": {"roles": ["ppe","zone"]}}.
    # Absent/empty → all modules on for that camera.
    "cameras": {},
    # Organization identity printed on exported safety reports.
    "report": {
        "site": "",
        "company": "",
        "preparer": "",
    },
}


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        saved  = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = {**SETTINGS_DEFAULTS}
        for k, v in saved.items():
            if isinstance(v, dict) and k in merged:
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    return dict(SETTINGS_DEFAULTS)


def _save_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into `base` (nested dicts merged, not replaced),
    so a POST that only touches one section can't clobber another screen's keys."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_TOKEN_KEY = "channel_access_token"


def _redacted(settings: dict) -> dict:
    """Settings safe to hand to a client: the LINE channel access token never
    leaves the server. It is a bearer credential — anyone holding it can push
    messages to the customer's LINE groups — and the UI has no reason to read it
    back. `token_set` tells the UI whether to show "saved" or "not configured".
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in settings.items()}
    line = out.get("line") or {}
    line["token_set"] = bool(line.get(_TOKEN_KEY))
    line[_TOKEN_KEY] = ""
    out["line"] = line
    return out


@app.get("/api/settings")
async def get_settings():
    return JSONResponse(_redacted(_load_settings()))


@app.post("/api/settings")
async def save_settings(body: dict[str, Any]):
    # Deep-merge over what's on disk so concurrent screens don't clobber keys.
    on_disk = {}
    if SETTINGS_FILE.exists():
        try:
            on_disk = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            on_disk = {}
    body = {k: (dict(v) if isinstance(v, dict) else v) for k, v in body.items()}
    line = body.get("line")
    if isinstance(line, dict):
        line.pop("token_set", None)          # read-only marker from _redacted()
        # GET never returns the token, so the Settings form posts it back empty
        # unless the user typed a new one. Empty means "leave unchanged" — writing
        # it through would erase the token every time any setting was saved.
        if not line.get(_TOKEN_KEY):
            line.pop(_TOKEN_KEY, None)
    merged = _deep_merge(on_disk, body)
    _save_settings(merged)
    if manager:
        manager.apply_settings(merged)
    return JSONResponse({"ok": True})


# ================================================================
# SNAPSHOT  (for Zone Editor canvas background)
# ================================================================

# 1×1 dark pixel PNG fallback (no camera)
_DARK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@app.get("/api/frame/snapshot")
async def frame_snapshot(camera_id: str | None = None):
    """Latest JPEG of one camera (Zone Editor background, grid tile poster). With
    no camera_id it falls back to the primary running camera."""
    if manager is not None:
        snap = None
        if camera_id:
            p = manager.get(camera_id)
            snap = p.get_snapshot() if (p and p.is_running()) else None
        else:
            prim = manager.primary()
            snap = prim.get_snapshot() if (prim and prim.is_running()) else None
        if snap:
            return StreamingResponse(iter([snap]), media_type="image/jpeg")
    return StreamingResponse(iter([_DARK_PNG]), media_type="image/png")


# ================================================================
# HISTORY  (backed by local SQLite store — PDPA: on-device)
# ================================================================
async def _in_executor(fn, *args, **kwargs):
    """Run a blocking (SQLite) call off the event loop so it can't stall the
    WebSocket frame broadcast or any other request while it runs."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


@app.get("/api/history/today")
async def history_today(day: str | None = None):
    from server import store
    s = await _in_executor(store.today_stats, day)
    prim = _primary_pipeline()
    s["uptime_seconds"] = prim.get_uptime() if prim else 0
    return JSONResponse(s)


@app.get("/api/history/hourly")
async def history_hourly(day: str | None = None):
    from server import store
    return JSONResponse(await _in_executor(store.hourly, day))


@app.get("/api/history/events")
async def history_events(limit: int = 20, offset: int = 0, day: str | None = None,
                         start: str | None = None, end: str | None = None):
    from server import store
    return JSONResponse(await _in_executor(
        store.list_events, limit=limit, offset=offset, day=day, start=start, end=end))


@app.get("/api/history/days")
async def history_days():
    from server import store
    return JSONResponse({"days": await _in_executor(store.available_days)})


@app.get("/api/history/snapshot/{event_id}")
async def history_snapshot(event_id: int):
    from server import store
    p = await _in_executor(store.snapshot_path, event_id)
    if p:
        return FileResponse(str(p), media_type="image/jpeg")
    return StreamingResponse(iter([_DARK_PNG]), media_type="image/png")


@app.get("/api/history/export.csv")
async def history_export(day: str | None = None, start: str | None = None,
                         end: str | None = None):
    from server import store
    tag = (start + "_" + end) if (start and end) else (day or "all")
    csv = await _in_executor(store.export_csv, day, start=start, end=end)
    return StreamingResponse(
        iter([csv]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=zentra_history_{tag}.csv"},
    )


@app.post("/api/history/clear")
async def history_clear():
    from server import store
    removed = await _in_executor(store.purge_all)
    # The dashboard KPIs read each pipeline's LIVE session counters, not the DB, so
    # clearing History alone leaves stale counts on the dashboard. Zero EVERY
    # running camera's counters and broadcast the reset so open dashboards update
    # without a reload.
    if manager is not None:
        for cid in manager.active_ids():
            p = manager.get(cid)
            if p is None:
                continue
            with p._lock:
                p.status["alerts"] = p._zero_alerts()
                p.status["last_emergency"] = None
                snapshot = dict(p.status)
            if p.on_status:
                try:
                    p.on_status(snapshot)
                except Exception:
                    pass
    return JSONResponse({"ok": True, "removed": removed})


# ================================================================
# DAILY REPORT  (local PDF + LINE text summary)
# ================================================================
@app.get("/api/report/daily.pdf")
async def report_daily_pdf(day: str | None = None, start: str | None = None,
                           end: str | None = None):
    from server.report import build_daily_pdf
    report_cfg = (_load_settings().get("report") or {})
    loop = asyncio.get_running_loop()
    try:
        path = await loop.run_in_executor(
            None, lambda: build_daily_pdf(day=day, start=start, end=end, org=report_cfg))
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return FileResponse(str(path), media_type="application/pdf", filename=path.name)


@app.post("/api/report/send-line")
async def report_send_line(body: dict[str, Any] | None = None):
    """Push the daily summary to LINE and report the REAL outcome.

    This used to return ok:True unconditionally, so with LINE unconfigured (no
    token / no group id) the UI cheerfully said "sent" while nothing left the
    machine. A safety tool must never claim an alert was delivered when it wasn't.
    """
    body = body or {}
    day  = body.get("day")
    try:
        from server.report import daily_stats_for_line
        from alerts.line_notify import send_daily_report
        import config as cfg

        if not getattr(cfg, "LINE_OA_CHANNEL_ACCESS_TOKEN", ""):
            return JSONResponse(
                {"ok": False, "error": "ยังไม่ได้ตั้งค่า LINE Token — ไปที่ ตั้งค่า → การแจ้งเตือน LINE"},
                status_code=400)
        # The daily report goes to every enabled LINE group.
        if not (list(getattr(cfg, "LINE_ALL_GROUPS", []) or [])
                or any(getattr(cfg, g, "") for g in
                       ("LINE_OA_GROUP_SUPERVISOR", "LINE_OA_GROUP_SAFETY"))):
            return JSONResponse(
                {"ok": False, "error": "ยังไม่ได้ตั้งค่า Group ID — ต้องมีอย่างน้อย 1 กลุ่ม"},
                status_code=400)

        stats = daily_stats_for_line(day)
        loop  = asyncio.get_running_loop()
        sent  = await loop.run_in_executor(None, send_daily_report, stats)
        if not sent:
            return JSONResponse(
                {"ok": False, "error": "LINE ปฏิเสธคำขอ — ตรวจสอบว่า Token ถูกต้องและบอทอยู่ในกลุ่มนั้น"},
                status_code=502)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ================================================================
# DATA COLLECTION (training dataset) + JOBS (train / upload)
# ================================================================
def _dir_size_mb(path: Path) -> float:
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    return round(total / (1024 * 1024), 1)


@app.get("/api/data/stats")
async def data_stats():
    cats = {}
    for cat in _DATA_CATEGORIES:
        d = COLLECTED_DIR / cat
        cats[cat] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return JSONResponse({
        "categories":   cats,
        "total_images": sum(cats.values()),
        "size_mb":      _dir_size_mb(COLLECTED_DIR),
        # "labeled" = has at least one BOX. Every collected frame gets a .txt,
        # empty for a frame the model found nothing in — counting mere existence
        # made this number identical to total_images, i.e. useless.
        "labeled":      sum(
            1 for cat in _DATA_CATEGORIES
            for j in (COLLECTED_DIR / cat).glob("*.jpg")
            if (t := j.with_suffix(".txt")).exists() and t.stat().st_size > 0
        ) if COLLECTED_DIR.exists() else 0,
    })


@app.post("/api/data/clear")
async def data_clear(body: dict[str, Any] | None = None):
    body = body or {}
    cats = [body["category"]] if body.get("category") in _DATA_CATEGORIES else _DATA_CATEGORIES
    removed = 0
    for cat in cats:
        d = COLLECTED_DIR / cat
        if not d.exists():
            continue
        for f in list(d.glob("*.jpg")) + list(d.glob("*.txt")):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    # The live collector caches its per-category counts (that's the quota) in
    # memory. Deleting the files without telling it leaves the quota reading
    # "full" forever, so the next capture session collects nothing at all.
    try:
        from utils.collector import get_collector
        get_collector().reload_counts()
    except Exception as e:
        print(f"[API] collector count reload failed: {e}")
    return JSONResponse({"ok": True, "removed_files": removed})


@app.post("/api/jobs/train")
async def jobs_train(body: dict[str, Any]):
    from server.jobs import manager as jobs
    task    = body.get("task", "ppe")
    if task not in ("ppe", "fall"):
        return JSONResponse({"ok": False, "error": "task ต้องเป็น ppe หรือ fall"}, status_code=400)
    args = ["training.trainer", "--task", task, "--export"]
    project = body.get("project")
    if project:
        args += ["--project", str(project)]
    ok, msg = jobs.start(args, label=f"เทรน {task.upper()}")
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.post("/api/jobs/upload")
async def jobs_upload(body: dict[str, Any]):
    from server.jobs import manager as jobs
    task = body.get("task", "ppe")
    if task not in ("ppe", "fall", "zone"):
        return JSONResponse({"ok": False, "error": "task ไม่ถูกต้อง"}, status_code=400)
    args = ["training.upload", "--task", task]
    project = body.get("project")
    if project:
        args += ["--project", str(project)]
    ok, msg = jobs.start(args, label=f"อัปโหลด {task.upper()} → Roboflow")
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.get("/api/jobs/status")
async def jobs_status():
    from server.jobs import manager as jobs
    return JSONResponse(jobs.status())


@app.get("/api/training/metrics")
async def training_metrics():
    """Latest persisted training metrics (mAP/precision/recall) per task."""
    logs_dir = ZENTRA_BACKEND / "logs"
    out: dict[str, Any] = {}
    if logs_dir.exists():
        for task in ("ppe", "fall"):
            files = sorted(logs_dir.glob(f"metrics_{task}_*.json"))
            if files:
                try:
                    out[task] = json.loads(files[-1].read_text(encoding="utf-8"))
                except Exception:
                    pass
    return JSONResponse(out)


@app.post("/api/jobs/stop")
async def jobs_stop():
    from server.jobs import manager as jobs
    return JSONResponse({"ok": jobs.stop()})


# ================================================================
# WEBSOCKET
# ================================================================
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    # Auth (when ZENTRA_API_TOKEN is set): the HTTP middleware can't see the WS
    # handshake, so gate here on the ?token= query param before accepting.
    if not _token_ok(websocket.query_params.get("token")):
        await websocket.close(code=1008)
        return
    await connections.connect(websocket)
    try:
        # Send initial status on connect (real values, not placeholders). With
        # multi-camera this reflects the PRIMARY (first running) camera plus the
        # site-wide aggregate alert counts; per-camera detail arrives via
        # /api/cameras/status and per-tile status events.
        modules = {"ppe": "standby", "zone": "standby", "fall": "standby"}
        alerts  = _aggregate_alerts()
        camera  = "disconnected"
        engine_error = None
        prim = _primary_pipeline()
        if prim is not None:
            with prim._lock:
                modules = dict(prim.status.get("modules", modules))
                camera  = prim.status.get("camera", camera)
                engine_error = prim.status.get("engine_error")
        elif manager is None:
            engine_error = "pipeline not initialised"
        await websocket.send_json({
            "type":    "event",
            "event":   "status",
            "modules": modules,
            "alerts":  alerts,
            "camera":  camera,
            "engine_error": engine_error,
        })
        # Keep connection alive; frames arrive via FrameBroadcaster
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        connections.disconnect(websocket)
    except Exception:
        connections.disconnect(websocket)
