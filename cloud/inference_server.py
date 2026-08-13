"""
cloud/inference_server.py — ZENTRA Cloud Inference Server

Runs the SAME verified detection stack (utils.ppe_engine.PPEEngine) on a cloud
GPU (RunPod / GCP L4). The edge (the operator's notebook) becomes a thin client:
it captures camera frames, POSTs them here, and displays the annotated frames
this server returns. Heavy inference leaves the notebook entirely, so the edge
stays smooth and many cameras can run at once on one strong GPU.

Protocol — ONE HTTP POST per frame (simple, thread-friendly, keep-alive):

    POST /infer            (Authorization: Bearer <ZENTRA_API_TOKEN>)
    { "camera_id": "cam0",
      "roles":     ["ppe","zone","fall"] | null,   # null = all modules on
      "ppe_items": ["helmet","vest"] | null,        # null = cfg.PPE_REQUIRED
      "frame":     "<base64 jpeg>",
      "draw":      true }                            # false → return no image

  → { "ok": true,
      "frame":  "<base64 annotated jpeg>" | null,
      "events": [ {type, level, msg, track_id}, ... ],
      "count":  <n persons>,
      "device": "cuda" }

State (ByteTrack ids, temporal confirmers, WORN-hold) is kept PER camera_id
across requests — the engine dict below persists for the life of the process, so
successive frames from the same camera continue the same tracks. Same-camera
requests are serialised by a per-camera lock (the tracker is not thread-safe);
different cameras run concurrently and the GPU time-shares them.

Run:
    ZENTRA_API_TOKEN=<secret> python -m uvicorn cloud.inference_server:app \
        --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import hmac
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# The detection stack lives under <repo>/backend (config, utils.*). Put it on the
# path exactly like pipeline/pipeline.py does so the cloud server shares the ONE
# implementation instead of a divergent copy.
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if _BACKEND.is_dir() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

app = FastAPI(title="ZENTRA Cloud Inference", docs_url=None, redoc_url=None)

# ── Auth ────────────────────────────────────────────────────
_TOKEN = os.getenv("ZENTRA_API_TOKEN", "").strip()


def _token_ok(provided: Optional[str]) -> bool:
    if not _TOKEN:
        return True
    # compare_digest raises TypeError on non-ASCII str (e.g. a Thai/emoji token),
    # so compare the UTF-8 BYTES — any token then works, still constant-time.
    return bool(provided) and hmac.compare_digest(
        provided.encode("utf-8"), _TOKEN.encode("utf-8"))


def _bearer(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.query_params.get("token")


# ── Per-camera engines (persist across requests) ────────────
_engines: dict[str, Any] = {}
_locks: dict[str, threading.Lock] = {}
_roles_now: dict[str, tuple] = {}       # last (roles, ppe_items) applied per camera
_registry_lock = threading.Lock()


def _norm(roles, ppe_items) -> tuple:
    r = tuple(sorted(roles)) if roles is not None else None
    p = tuple(sorted(ppe_items)) if ppe_items is not None else None
    return (r, p)


def _get_engine(camera_id: str, roles, ppe_items):
    """Fetch (or lazily build) the persistent engine for this camera. If the
    requested roles/ppe_items changed since last call, hot-apply them without a
    model reload (apply_roles / apply_ppe_items) so a Settings change on the edge
    takes effect on the next frame."""
    from utils.ppe_engine import PPEEngine
    with _registry_lock:
        eng = _engines.get(camera_id)
        if eng is None:
            eng = PPEEngine(zones_path=None, camera_id=camera_id,
                            roles=(set(roles) if roles is not None else None),
                            ppe_items=ppe_items)
            _engines[camera_id] = eng
            _locks[camera_id] = threading.Lock()
            _roles_now[camera_id] = _norm(roles, ppe_items)
            print(f"[Cloud] 🧠 engine built for '{camera_id}' "
                  f"(device={eng.detector.device}, roles={roles})")
            return eng, _locks[camera_id]
        # Existing engine → apply role/item changes if they differ.
        want = _norm(roles, ppe_items)
        if want != _roles_now.get(camera_id):
            try:
                eng.apply_roles(set(roles) if roles is not None else None)
                eng.apply_ppe_items(ppe_items)
                _roles_now[camera_id] = want
                print(f"[Cloud] ⚙️  '{camera_id}' roles→{roles}, ppe→{ppe_items}")
            except Exception as e:
                print(f"[Cloud] role hot-apply failed for {camera_id}: {e}")
        return eng, _locks[camera_id]


def _clean_events(events: list) -> list:
    """Strip non-JSON-serialisable fields (the 'key' tuple) so events cross the
    wire. Keeps exactly what the edge needs to raise an alert."""
    out = []
    for e in events or []:
        out.append({
            "type":     e.get("type", "ppe"),
            "level":    e.get("level", "warning"),
            "msg":      e.get("msg", ""),
            "track_id": e.get("track_id"),
        })
    return out


# ── Endpoints ───────────────────────────────────────────────
@app.get("/health")
async def health():
    dev = "cpu"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return {"ok": True, "device": dev, "cameras": list(_engines.keys()),
            "auth": bool(_TOKEN)}


@app.post("/infer")
async def infer(request: Request):
    if not _token_ok(_bearer(request)):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)

    camera_id = str(body.get("camera_id", "cam0"))
    roles     = body.get("roles", None)          # None = all modules on
    ppe_items = body.get("ppe_items", None)
    draw      = bool(body.get("draw", True))
    frame_b64 = body.get("frame")
    if not frame_b64:
        return JSONResponse({"ok": False, "error": "no frame"}, status_code=400)

    # Decode JPEG → BGR ndarray
    try:
        buf = np.frombuffer(base64.b64decode(frame_b64), np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"decode: {e}"}, status_code=400)

    # Blocking inference runs in FastAPI's threadpool (sync-in-async). Serialise
    # per camera: the ByteTrack tracker is stateful and NOT thread-safe.
    def _run():
        eng, lock = _get_engine(camera_id, roles, ppe_items)
        with lock:
            h, w = frame.shape[:2]
            recs, events = eng.detect(frame)
            if getattr(eng, "fall_ready", False):
                try:
                    events = list(events) + list(eng.fall_step(frame, recs))
                except Exception as e:
                    print(f"[Cloud] fall_step {camera_id}: {e}")
            # ALWAYS return lightweight box coordinates so the edge can draw them on
            # its own live frames (smooth). The full annotated JPEG is optional and
            # only encoded when the caller asks (draw=true) — the edge sets it false.
            draw_data = eng.draw_items(recs, w, h)
            out_b64 = None
            if draw:
                annotated = eng.draw_on(frame.copy(), recs)
                ok, enc = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    out_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
            return (out_b64, _clean_events(events), len(recs),
                    eng.detector.device, draw_data)

    import anyio
    try:
        out_b64, events, count, device, draw_data = await anyio.to_thread.run_sync(_run)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"infer: {e}"}, status_code=500)

    return JSONResponse({"ok": True, "frame": out_b64, "events": events,
                         "count": count, "device": device,
                         "boxes": draw_data["boxes"], "zones": draw_data["zones"]})


@app.post("/reset")
async def reset(request: Request):
    """Drop a camera's engine (clears ByteTrack ids) — call on stream restart."""
    if not _token_ok(_bearer(request)):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    cam = body.get("camera_id")
    with _registry_lock:
        if cam and cam in _engines:
            try:
                _engines[cam].close_fall()
            except Exception:
                pass
            _engines.pop(cam, None); _locks.pop(cam, None); _roles_now.pop(cam, None)
            return JSONResponse({"ok": True, "reset": cam})
    return JSONResponse({"ok": True, "reset": None})
