"""
pipeline/frame_broadcaster.py — ZENTRA Frame Broadcaster
Reads latest annotated frames from Pipeline, encodes as JPEG,
and broadcasts base64 data over WebSocket at target FPS.
"""
from __future__ import annotations

import asyncio
import base64
import threading
import time
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from pipeline.pipeline import Pipeline


class FrameBroadcaster(threading.Thread):
    """
    Daemon thread that encodes Pipeline frames and broadcasts them
    to all WebSocket clients via the asyncio event loop.
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        manager,
        loop: asyncio.AbstractEventLoop,
        fps: int = 10,
        width: int = 960,
        height: int = 540,
        quality: int = 70,
        camera_id: str = "cam0",
    ):
        super().__init__(daemon=True, name=f"FrameBroadcaster-{camera_id}")
        self._pipeline = pipeline
        self._manager  = manager
        self._loop     = loop
        self._fps      = fps
        self._width    = width
        self._height   = height
        self._quality  = quality
        # Which camera these frames belong to. The UI (CCTV grid) routes each
        # frame to its own <img> tile by this id, so several cameras can stream
        # over the ONE shared WebSocket without their frames overwriting each
        # other. Single-camera callers keep the historical "cam0" default.
        self._camera_id = camera_id
        self._running  = True

    def stop(self):
        self._running = False

    def set_stream(self, fps=None, width=None, height=None, quality=None):
        """Re-tune the stream while running (Settings → display). Plain attribute
        writes: the loop re-reads them every iteration, so a change takes effect on
        the next frame without restarting the camera."""
        if fps:     self._fps     = int(fps)
        if width:   self._width   = int(width)
        if height:  self._height  = int(height)
        if quality: self._quality = int(quality)

    def run(self):
        # DROP-TO-LATEST: keep a handle on the previous broadcast. We only encode +
        # send a NEW frame once the previous send has fully completed. Without this,
        # a client (or its TCP send buffer) that can't keep up makes frames pile up
        # in the event loop; because WebSocket/TCP is reliable + ordered, the viewer
        # then plays an ever-growing backlog and the video drifts further behind real
        # time the longer it runs. Skipping while a send is in flight guarantees the
        # browser always gets the freshest frame, never a queue of stale ones.
        pending = None
        while self._running:
            t0 = time.monotonic()
            interval = 1.0 / max(self._fps, 1)   # re-read: set_stream() may change it

            # Only do any work when the pipeline is live AND someone is watching —
            # no viewers → no resize/encode/base64 (frees CPU for inference).
            if self._pipeline.is_running() and self._manager.active \
                    and (pending is None or pending.done()):
                frame = self._pipeline.get_latest_frame()
                if frame is not None:
                    try:
                        frame = cv2.resize(frame, (self._width, self._height))
                        ok, buf = cv2.imencode(
                            ".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self._quality],
                        )
                        if ok:
                            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                            pending = asyncio.run_coroutine_threadsafe(
                                self._manager.broadcast(
                                    {"type": "frame", "camera_id": self._camera_id,
                                     "data": b64}
                                ),
                                self._loop,
                            )
                    except Exception as e:
                        print(f"[Broadcaster] encode/send error: {e}")

            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)
