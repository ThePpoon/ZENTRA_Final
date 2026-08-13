#!/usr/bin/env python3
# utils/zone_geometry.py — Safety-zone polygons: load, scale, point-in-polygon.
# ================================================================
# Polygons are stored in zones.json as NORMALIZED 0–1 coordinates (fixes the
# editor-canvas ↔ camera-frame coordinate-space mismatch: normalize once when
# saving, scale to the actual frame (w,h) at runtime). Each zone:
#   {id, name, color, points:[[x,y]|{x,y}...](0–1), type:"danger"|"exclusion", enabled}
#
# Intrusion test uses the person's FOOT point (bottom-center of bbox) when
# cfg.ZONE_USE_FOOT_POINT, else the bbox centre — so someone is "in" the zone
# when their feet are, not their head.
# ================================================================
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np

import config as cfg


def _pts_to_xy(points) -> list[tuple[float, float]]:
    out = []
    for p in points:
        if isinstance(p, dict):
            out.append((float(p["x"]), float(p["y"])))
        else:
            out.append((float(p[0]), float(p[1])))
    return out


class Zone:
    def __init__(self, raw: dict):
        self.id = raw.get("id")
        self.name = raw.get("name", f"Zone {self.id}")
        self.color = raw.get("color", "#ef4444")
        self.type = raw.get("type", "danger")          # "danger" | "exclusion"
        self.camera_id = raw.get("camera_id")           # which camera; None = all cameras
        self.enabled = raw.get("enabled", True)
        self.norm_pts = _pts_to_xy(raw.get("points", []))   # 0–1

    def is_ready(self) -> bool:
        return self.enabled and len(self.norm_pts) >= 3

    def polygon_px(self, w: int, h: int) -> np.ndarray:
        return np.array([[x * w, y * h] for x, y in self.norm_pts], dtype=np.int32)

    def contains(self, point_xy, w: int, h: int) -> bool:
        poly = self.polygon_px(w, h)
        return cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False) >= 0


def load_zones(path: str | None = None, camera_id: str | None = None) -> list[Zone]:
    """Load ready zones. If camera_id is given, keep only zones for that camera
    (a zone with camera_id=None applies to every camera — back-compat)."""
    p = Path(path or getattr(cfg, "ZONE_POLYGON_FILE", "data/zones.json"))
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text() or "[]")
    except Exception:
        return []
    zones = [Zone(z) for z in raw]
    if camera_id is not None:
        zones = [z for z in zones if z.camera_id in (None, camera_id)]
    return [z for z in zones if z.is_ready()][: getattr(cfg, "MAX_ZONES", 10)]


def anchor_of(person: dict) -> tuple[float, float]:
    """Foot point (bottom-center) or centre, per ZONE_USE_FOOT_POINT."""
    if getattr(cfg, "ZONE_USE_FOOT_POINT", True):
        return person.get("foot", (person["cx"], person["y2"]))
    return (person["cx"], person["cy"])


def in_any_exclusion(person: dict, zones: list[Zone], w: int, h: int) -> bool:
    a = anchor_of(person)
    return any(z.type == "exclusion" and z.contains(a, w, h) for z in zones)


def danger_hits(person: dict, zones: list[Zone], w: int, h: int) -> list[Zone]:
    """Danger zones whose polygon contains this person's anchor."""
    a = anchor_of(person)
    return [z for z in zones if z.type == "danger" and z.contains(a, w, h)]
