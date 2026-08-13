#!/usr/bin/env python3
# utils/temporal.py — generic per-key temporal confirmation + cooldown gate,
# shared by PPE (per track_id+category) and Zone (per track_id+zone_id).
# ================================================================
# TrackWindowConfirmer: a boolean is "confirmed" only when it was True on at
# least `confirm_frames` of the last `window` observations for that key. This is
# what turns a single-frame flicker into a real, alarm-worthy event.
#
# CooldownGate: after firing for a key, suppress re-firing for `seconds`.
# ================================================================
from __future__ import annotations
import time
from collections import defaultdict, deque


class TrackWindowConfirmer:
    def __init__(self, confirm_frames: int, window: int):
        self.confirm_frames = confirm_frames
        self.window = window
        self._hist: dict = defaultdict(lambda: deque(maxlen=window))

    def update(self, key, hit: bool) -> bool:
        """Record one observation for `key`; return True if now confirmed."""
        dq = self._hist[key]
        dq.append(bool(hit))
        return sum(dq) >= self.confirm_frames

    def is_confirmed(self, key) -> bool:
        dq = self._hist.get(key)
        return bool(dq) and sum(dq) >= self.confirm_frames

    def drop(self, key):
        self._hist.pop(key, None)

    def gc(self, live_keys: set):
        """Forget keys no longer present (e.g. tracks that left)."""
        for k in list(self._hist):
            if k not in live_keys:
                self._hist.pop(k, None)


class TimeWindowConfirmer:
    """FPS-independent confirmation. A key is confirmed when it was True for at
    least `min_ratio` of the observations in the last `window_sec` seconds, with an
    absolute floor of `min_hits` True observations (the jitter guard).

    Why not the frame-count TrackWindowConfirmer for zones: a fixed N-of-M frame
    window's TIME span swings with the frame rate (5 frames = 0.17s at 30 fps but
    ~1.3s at 4 fps), so a fast walk-through at low fps can fail 3-of-5 and evade the
    alarm. Measuring in seconds behaves the same at 4 or 30 fps. Same interface as
    TrackWindowConfirmer (update / is_confirmed / drop / gc) so it drops straight in.
    """

    def __init__(self, window_sec: float, min_ratio: float = 0.5, min_hits: int = 2):
        self.window_sec = float(window_sec)
        self.min_ratio  = float(min_ratio)
        self.min_hits   = int(min_hits)
        self._hist: dict = defaultdict(deque)   # key -> deque[(t, hit)]

    def _confirmed(self, dq) -> bool:
        if not dq:
            return False
        hits = sum(1 for _, h in dq if h)
        return hits >= self.min_hits and hits >= self.min_ratio * len(dq)

    def update(self, key, hit: bool, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        dq = self._hist[key]
        dq.append((now, bool(hit)))
        cutoff = now - self.window_sec
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        return self._confirmed(dq)

    def is_confirmed(self, key) -> bool:
        return self._confirmed(self._hist.get(key))

    def drop(self, key):
        self._hist.pop(key, None)

    def gc(self, live_keys: set):
        """Forget keys no longer present (e.g. tracks that left)."""
        for k in list(self._hist):
            if k not in live_keys:
                self._hist.pop(k, None)


class CooldownGate:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: dict = {}

    def ready(self, key, now: float | None = None) -> bool:
        """True if `key` hasn't fired within `seconds`; records the fire time."""
        now = time.time() if now is None else now
        last = self._last.get(key, 0.0)
        if now - last >= self.seconds:
            self._last[key] = now
            return True
        return False

    def mark(self, key, now: float | None = None) -> None:
        """Force-record a fire time for `key` (e.g. after firing on a rising edge
        that bypassed ready()) so the next ready() measures from here."""
        self._last[key] = time.time() if now is None else now
