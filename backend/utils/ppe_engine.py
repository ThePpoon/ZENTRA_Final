#!/usr/bin/env python3
# utils/ppe_engine.py — one class that wraps the whole verified engine
# (ByteTrack detect_track + PPE association + temporal confirm + zone) behind a
# single `process(frame) -> (annotated_frame, events)` call, so the app pipeline
# (and the offline harness) share ONE implementation.
# ================================================================
from __future__ import annotations

from pathlib import Path

import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as cfg
from utils.detect_track import Detector, PersonDetector
from utils.ppe_association import associate, violations_of, CATEGORIES, cat_state_of
from utils.temporal import TrackWindowConfirmer, TimeWindowConfirmer, CooldownGate
from utils import zone_geometry
from utils.pose_visibility import (visible_cats, match_pose_to_person,
                                   cat_visibility, facing_score,
                                   orientation_weight, scale_weight)
from utils.tracker import ByteTracker

GREEN, RED, CYAN = (0, 210, 0), (0, 0, 220), (255, 190, 0)


def _iou_box(a: dict, b: dict) -> float:
    """IoU of two detection dicts (x1,y1,x2,y2 in pixels)."""
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
          + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter)
    return inter / ua if ua > 0 else 0.0

# Thai labels for confirmed missing-PPE categories (for alert messages)
_CAT_TH = {"helmet": "ไม่สวมหมวก", "vest": "ไม่สวมเสื้อกั๊ก", "gloves": "ไม่สวมถุงมือ",
           "glasses": "ไม่สวมแว่นตา", "boots": "ไม่สวมรองเท้าเซฟตี้"}
# Short Thai labels for on-frame box overlays
_CAT_SHORT_TH = {"helmet": "หมวก", "vest": "เสื้อกั๊ก", "gloves": "ถุงมือ",
                 "glasses": "แว่นตา", "boots": "รองเท้า"}

# Thai-capable font for on-frame overlays (cv2.putText can't render Thai → "???").
_FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Sarabun-SemiBold.ttf"
_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont"] = {}


def _font(size: int):
    f = _FONT_CACHE.get(size)
    if f is None:
        try:
            f = ImageFont.truetype(str(_FONT_PATH), size)
        except Exception:
            f = ImageFont.load_default()
        _FONT_CACHE[size] = f
    return f


# ── Thai-correct overlay text ────────────────────────────────────────────────
# Pillow has no complex-script shaping, so Thai syllables with a stacked upper
# vowel AND a tone mark (e.g. "พื้นที่") render with the marks in the wrong place
# — looks broken on the video overlay. Shape with harfbuzz and rasterise glyphs
# with freetype so Thai renders correctly. Optional deps: if either is missing we
# fall back to plain Pillow (Latin/ASCII stays fine; Thai just isn't reshaped).
try:
    import uharfbuzz as _hb
    import freetype as _ft
    _SHAPE_OK = True
except Exception:
    _SHAPE_OK = False

_HB_FONT = None
_FT_FACE = None
_TEXT_IMG_CACHE: dict = {}       # (text, size, rgb) -> (RGBA strip, ascent_px)
_TEXT_TILE_CACHE: dict = {}      # (text, size, rgb) -> (BGR tile uint8, alpha float32)


def _shape_text_img(text: str, size: int, rgb: tuple):
    """RGBA image of `text` with Thai marks positioned correctly (harfbuzz +
    freetype), transparent background. Cached per (text, size, rgb) because
    overlay labels repeat every frame. Returns None if shaping is unavailable."""
    if not _SHAPE_OK or not text:
        return None
    key = (text, size, rgb)
    hit = _TEXT_IMG_CACHE.get(key)
    if hit is not None:
        return hit
    global _HB_FONT, _FT_FACE
    if _HB_FONT is None:
        _HB_FONT = _hb.Font(_hb.Face(_FONT_PATH.read_bytes()))
    if _FT_FACE is None:
        _FT_FACE = _ft.Face(str(_FONT_PATH))
    hbf, ftf = _HB_FONT, _FT_FACE
    hbf.scale = (size * 64, size * 64)
    buf = _hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    _hb.shape(hbf, buf)
    ftf.set_pixel_sizes(0, size)
    ascent = int(ftf.size.ascender / 64) or int(size * 1.15)
    descent = int(-ftf.size.descender / 64)
    height = max(1, ascent + descent + 2)
    glyphs, pen = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        glyphs.append((info.codepoint, pen + pos.x_offset / 64.0, pos.y_offset / 64.0))
        pen += pos.x_advance / 64.0
    width = max(1, int(pen) + 2)
    strip = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for gid, gx, gy in glyphs:
        ftf.load_glyph(gid, _ft.FT_LOAD_RENDER)
        g = ftf.glyph
        w, h = g.bitmap.width, g.bitmap.rows
        if w == 0 or h == 0:
            continue
        alpha = np.frombuffer(bytes(g.bitmap.buffer), np.uint8).reshape(h, w)
        glyph = Image.new("RGBA", (w, h), tuple(rgb) + (0,))
        glyph.putalpha(Image.fromarray(alpha))
        strip.alpha_composite(glyph, (int(round(gx + g.bitmap_left)),
                                      int(round(ascent - g.bitmap_top - gy))))
    # Bound the cache. Labels no longer carry track ids, so the label set is now
    # small and fixed ("ปลอดภัย", "ขาด: หมวก", ...) and this rarely trips — it
    # used to churn a fresh entry per track id.
    if len(_TEXT_IMG_CACHE) > 512:
        _TEXT_IMG_CACHE.clear()
    _TEXT_IMG_CACHE[key] = (strip, ascent)
    return _TEXT_IMG_CACHE[key]


def _pill_text_img(text: str, size: int, rgb: tuple):
    """RGBA image of `text` on its dark pill, rendered with plain Pillow (no
    complex-script shaping). Used when harfbuzz/freetype are unavailable."""
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.textbbox((0, 0), text, font=font)
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    img = Image.new("RGBA", (w + 8, h + 6), (0, 0, 0, 150))
    ImageDraw.Draw(img, "RGBA").text((4 - box[0], 3 - box[1]), text, font=font,
                                     fill=tuple(rgb), stroke_width=1,
                                     stroke_fill=(0, 0, 0, 255))
    return img


def _text_tile(text: str, size: int, rgb: tuple):
    """Cached (BGR tile, alpha) for one label — the dark pill with the shaped text
    already composited onto it, as plain numpy arrays ready to blend into a frame.

    Building the pill+glyphs is the expensive part and it is fully determined by
    (text, size, colour), so it is done ONCE per distinct label and reused every
    frame. The label set is small and fixed ("ปลอดภัย", "ขาด: หมวก", zone names…),
    so in the steady state this is a pure dict hit.
    """
    key = (text, size, rgb)
    hit = _TEXT_TILE_CACHE.get(key)
    if hit is not None:
        return hit
    tile = None
    shaped = _shape_text_img(text, size, rgb)
    if shaped:
        strip, _asc = shaped
        w, h = strip.size
        tile = Image.new("RGBA", (w + 8, h + 4), (0, 0, 0, 150))
        tile.alpha_composite(strip, (4, 2))
    else:
        try:
            tile = _pill_text_img(text, size, rgb)
        except Exception:
            return None
    arr   = np.asarray(tile)                                   # RGBA
    bgr   = np.ascontiguousarray(arr[:, :, 2::-1]).astype(np.float32)
    alpha = np.ascontiguousarray(arr[:, :, 3:4]).astype(np.float32) / 255.0
    if len(_TEXT_TILE_CACHE) > 512:
        _TEXT_TILE_CACHE.clear()
    _TEXT_TILE_CACHE[key] = (bgr, alpha)
    return _TEXT_TILE_CACHE[key]


def draw_texts_on(frame, items, size: int = 18):
    """Blend overlay text (each with a dark pill) straight into the BGR frame, with
    harfbuzz shaping so Thai renders correctly. Module-level so the edge's remote
    overlay can reuse the exact same renderer as the engine. `items` is a list of
    (x, y_top, text, rgb). Draws IN PLACE and returns the same frame.

    Only the small label tiles are touched — NOT the whole frame. The previous
    version pushed every frame through BGR→RGB→PIL→RGB→BGR (~5ms on a 960x540
    frame, ~8ms with labels) purely to paste a few 120x24 strips. That ran on each
    camera's display loop at capture FPS, so with 4 cameras it burned ~1 CPU-second
    per second of pure GIL-held Python — starving the capture, broadcast and
    asyncio threads, which is what froze the tiles once PPE turned the labels on.
    Compositing only the tiles is ~7x cheaper and pixel-identical.
    """
    if not items:
        return frame
    H, W = frame.shape[:2]
    for (x, y, text, rgb) in items:
        if not text:
            continue
        tile = _text_tile(text, size, tuple(int(c) for c in rgb))
        if tile is None:
            continue
        bgr, alpha = tile
        th, tw = bgr.shape[:2]
        # Top-left of the pill (the strip itself sits 4px/2px inside it).
        x0, y0 = int(x) - 4, int(y) - 2
        # Clip against the frame — a label near an edge must not wrap or throw.
        sx, sy = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x0 + tw), min(H, y0 + th)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        w, h = dx1 - dx0, dy1 - dy0
        src = bgr[sy:sy + h, sx:sx + w]
        a   = alpha[sy:sy + h, sx:sx + w]
        roi = frame[dy0:dy1, dx0:dx1]
        roi[:] = (roi * (1.0 - a) + src * a).astype(np.uint8)
    return frame


class PPEEngine:
    @staticmethod
    def _resolve_required(ppe_items) -> set:
        """Which PPE categories THIS camera enforces (absence = violation).

        An EMPTY list is treated as "not configured" and falls back to
        cfg.PPE_REQUIRED, exactly like None. It used to be honoured literally,
        which produced a camera whose PPE role reads "ok" everywhere in the UI
        while it enforces nothing: it can never raise a violation and every
        person is drawn in the neutral colour. That is the silent-success
        failure a safety system must not have — and it was reachable from the
        Cameras page, which wrote [] whenever the item checkboxes had not been
        touched. A camera that should check nothing turns the PPE ROLE off; the
        item list is not the switch for that."""
        if not ppe_items:
            if ppe_items is not None:
                print("[PPEEngine] ⚠️ ppe_items ว่าง → ใช้ค่าเริ่มต้น "
                      f"{list(getattr(cfg, 'PPE_REQUIRED', ['helmet', 'vest']))} "
                      "(ถ้าไม่ต้องการตรวจ PPE ให้ปิดบทบาท PPE แทน)")
            ppe_items = getattr(cfg, "PPE_REQUIRED", ["helmet", "vest"])
        return set(ppe_items) & set(CATEGORIES)

    def __init__(self, zones_path: str | None = None, device: str | None = None,
                 camera_id: str | None = None, roles: set | None = None,
                 ppe_items=None, hub=None):
        # `hub` (ModelHub) = SHARED-MODEL mode for multi-camera edge: models are
        # loaded ONCE by the hub and shared; this engine owns only per-camera state
        # (its own ByteTracker + confirmers + zones). `hub=None` = the original
        # single-camera behaviour (this engine loads + owns its own models).
        self._hub = hub
        self._device = device
        self.ppe_enabled  = (roles is None) or ("ppe" in roles)
        self.zone_enabled = (roles is None) or ("zone" in roles)

        if hub is not None:
            # ── SHARED MODE ── reference the hub's models; own no model.
            self.person_detector = None
            self.ppe_detector = hub.ppe            # for crop pass + model_path logging
            if self.ppe_enabled and hub.ppe is None:
                self.ppe_enabled = False           # no PPE weights → PPE off (honest)
            self._pose = hub.pose_model
            self.detector = hub.ppe or hub.person  # status/device readout
            # This camera's OWN tracker (never shared) — replaces the hub's stateless
            # predict with stable per-camera ids. Params cfg-overridable for parity.
            # NOTE the config key names. These were BYTETRACK_BUFFER / BYTETRACK_MATCH
            # / BYTETRACK_LOW, none of which config.py defines — so getattr silently
            # fell back to the literals here and match_thresh ran at 0.8 instead of
            # the 0.50 that config.py tunes precisely to stop id churn (this tracker
            # has no Kalman, so a walking person drops below a high IoU between
            # frames and gets a NEW id, which resets that track's PPE confirmation).
            # Shared multi-camera mode is the ONLY user of these scalars.
            self._bytetrack = ByteTracker(
                track_thresh=float(getattr(cfg, "BYTETRACK_TRACK_THRESH", 0.5)),
                track_buffer=int(getattr(cfg, "BYTETRACK_TRACK_BUFFER", 30)),
                match_thresh=float(getattr(cfg, "BYTETRACK_MATCH_THRESH", 0.5)),
                low_thresh=float(getattr(cfg, "BYTETRACK_LOW_THRESH", 0.1)))
        else:
            # ── SINGLE-CAMERA MODE ── load + own the models (original behaviour).
            self.person_detector = PersonDetector(device=device)
            self.person_detector.reset()
            self.ppe_detector = None
            if self.ppe_enabled:
                try:
                    self.ppe_detector = Detector(device=device)
                except FileNotFoundError as e:
                    print(f"[PPEEngine] ⚠️ PPE model unavailable → PPE paused: {e}")
                    self.ppe_enabled = False
            self._pose = None
            if self.ppe_enabled and (bool(getattr(cfg, "PPE_VISIBILITY_GATE", True))
                                     or bool(getattr(cfg, "PPE_KP_ASSIGN", True))):
                try:
                    from ultralytics import YOLO
                    self._pose = YOLO(str(cfg.PPE_POSE_MODEL))
                    print("[PPEEngine] 👁️ PPE pose on")
                except Exception as e:
                    print(f"[PPEEngine] ⚠️ pose unavailable → gate/kp-assign off: {e}")
            self.detector = self.ppe_detector or self.person_detector
            self._bytetrack = None

        # Visibility gate / keypoint-assign need pose AND PPE on. Same rule in both
        # modes; disabled automatically when pose is unavailable.
        self._vis_gate  = (self.ppe_enabled and self._pose is not None
                           and bool(getattr(cfg, "PPE_VISIBILITY_GATE", True)))
        self._kp_assign = (self.ppe_enabled and self._pose is not None
                           and bool(getattr(cfg, "PPE_KP_ASSIGN", True)))
        self.pconf = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
        self.pcool = CooldownGate(cfg.VIOLATION_COOLDOWN_SECONDS)
        # WORN-hold: a second confirmer that latches "was CONFIRMED wearing it", plus
        # a per-(track,cat) expiry so an occlusion right after doesn't flip to violation.
        self._worn_hold = bool(getattr(cfg, "PPE_WORN_HOLD", True))
        self._worn_hold_sec = float(getattr(cfg, "PPE_WORN_HOLD_SEC", 4.0))
        self.wconf = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
        self._worn_until: dict = {}
        self._worn_places: dict = {}     # cat -> [(cx, cy, expiry)] — id-independent hold
        self._ppe_incidents: list = []   # (t, cat, cx, cy) — spatial alert de-dup (anti-spam)
        self._now = time.time()
        self.zconf = TimeWindowConfirmer(cfg.ZONE_CONFIRM_WINDOW_SEC,
                                         cfg.ZONE_CONFIRM_MIN_RATIO, cfg.ZONE_CONFIRM_MIN_HITS)
        self.zcool = CooldownGate(cfg.ZONE_COOLDOWN_SECONDS)
        self._zones_path = zones_path
        self._camera_id = camera_id            # only load zones for this camera (None = all)
        self.zones = zone_geometry.load_zones(zones_path, camera_id)
        # (tid, zone_id) currently confirmed-inside a danger zone. Drives rising-edge
        # alerts: a fresh entry (outside→inside) fires immediately, so leaving and
        # re-entering alerts again instead of being swallowed by the plain cooldown.
        self._zone_inside: set = set()
        # Fall is opt-IN even when roles is None: it loads a pose model + a TFLite
        # interpreter, which the PPE-only offline tools have no reason to pay for.
        self.fall_enabled = bool(roles) and ("fall" in roles)
        self.fconf = TrackWindowConfirmer(cfg.FALL_CONFIRM_FRAMES, cfg.FALL_CONFIRM_WINDOW)
        self.fcool = CooldownGate(cfg.FALL_COOLDOWN_SECONDS)
        self._fallen: set = set()      # confirmed-fallen track ids, for _draw
        self._fall_incidents: list = []   # (t, cx, cy) — spatial alert de-duplication
        self._fall = None
        self._build_fall()
        # Round-robin cursor for the person-crop "zoom" pass: when more people than
        # PPE_CROP_MAX_PER_FRAME need a crop, we crop a bounded slice this frame and
        # advance the cursor so the rest get their turn on following frames — the
        # 3-of-5 confirm bridges the gap, and the per-frame cost stays flat even when
        # a crowd walks in (the "delay on scene change" guard).
        self._crop_rr: int = 0
        # Per-camera set of PPE categories to enforce (absence = violation).
        # Selectable in the app's per-camera roles modal; None → cfg.PPE_REQUIRED.
        self._required = self._resolve_required(ppe_items)
        # Raw detections of the last detect() — persons + PPE items BEFORE
        # association drops the items it could not attach to anyone. The data
        # collector labels from these: an unassociated helmet still has to appear
        # in the .txt, or the pseudo-label teaches the model that helmet=background.
        self.last_dets: list = []
        # Graded per-person PPE evidence from the last frame (see _observe_ppe).
        # Read by the decision path below to decide when this camera can see
        # well enough to accuse anybody at all.
        self.last_observations: list = []

    def reload_zones(self):
        self.zones = zone_geometry.load_zones(self._zones_path, self._camera_id)

    def _build_fall(self):
        """Load the pose + fall models only for cameras that actually want fall.
        A failure must never take the pipeline down — the module just reports
        'standby' and PPE/zone keep running."""
        if not self.fall_enabled:
            self._fall = None
            return
        if self._fall is not None:
            return
        try:
            from utils.fall_detector import FallDetector
            self._fall = FallDetector()
            print(f"[PPEEngine] 🚑 fall detector ready "
                  f"(backend={self._fall.extractor.name}, mode={self._fall.mode})")
        except Exception as e:
            self._fall = None
            print(f"[PPEEngine] ⚠️ fall detector unavailable → {e}")

    @property
    def fall_ready(self) -> bool:
        return self.fall_enabled and self._fall is not None

    def apply_roles(self, roles: set | None):
        """Update which modules run/draw without reloading the model."""
        self.ppe_enabled  = (roles is None) or ("ppe" in roles)
        self.zone_enabled = (roles is None) or ("zone" in roles)
        # PPE just turned on for a camera that started without it → build the item
        # detector now. If the model is absent, pause PPE (off) rather than crash.
        # In SHARED mode the PPE model belongs to the hub — never load our own.
        if self.ppe_enabled and self.ppe_detector is None:
            if self._hub is not None:
                self.ppe_detector = self._hub.ppe
                if self.ppe_detector is None:
                    self.ppe_enabled = False        # hub has no PPE weights
            else:
                try:
                    self.ppe_detector = Detector(device=self._device)
                    self.detector = self.ppe_detector
                except FileNotFoundError as e:
                    print(f"[PPEEngine] ⚠️ PPE model unavailable → PPE paused: {e}")
                    self.ppe_enabled = False
        was = self.fall_enabled
        self.fall_enabled = bool(roles) and ("fall" in roles)
        if self.fall_enabled and not was:
            self._build_fall()
        elif was and not self.fall_enabled:
            self.close_fall()
        # Re-derive the pose-dependent gates from the NEW ppe_enabled. These were
        # computed once in __init__ and never updated here, so a live role change
        # left them stale in BOTH directions: turning PPE on gave that camera no
        # visibility gate (it could accuse someone whose head is hidden), and
        # turning PPE off kept the camera in the scheduler's pose batch — paying
        # for a pose forward it no longer uses.
        self._vis_gate  = (self.ppe_enabled and self._pose is not None
                           and bool(getattr(cfg, "PPE_VISIBILITY_GATE", True)))
        self._kp_assign = (self.ppe_enabled and self._pose is not None
                           and bool(getattr(cfg, "PPE_KP_ASSIGN", True)))

    def apply_ppe_items(self, ppe_items):
        """Update which PPE categories this camera enforces (no model reload)."""
        self._required = self._resolve_required(ppe_items)

    def refresh_tunables(self):
        """Re-read cfg-based gates (confirm windows + cooldowns) in place, so a
        Settings save takes effect without a costly YOLO reload. (Confidence is
        read per-call from cfg inside detect_track, so it needs nothing here.)"""
        self.pconf = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
        self.pcool = CooldownGate(cfg.VIOLATION_COOLDOWN_SECONDS)
        self._worn_hold = bool(getattr(cfg, "PPE_WORN_HOLD", True))
        self._worn_hold_sec = float(getattr(cfg, "PPE_WORN_HOLD_SEC", 4.0))
        self.wconf = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
        self._worn_until = {}
        self._worn_places: dict = {}     # cat -> [(cx, cy, expiry)] — id-independent hold
        self._ppe_incidents = []         # spatial alert de-dup
        self.zconf = TimeWindowConfirmer(cfg.ZONE_CONFIRM_WINDOW_SEC,
                                         cfg.ZONE_CONFIRM_MIN_RATIO, cfg.ZONE_CONFIRM_MIN_HITS)
        self.zcool = CooldownGate(cfg.ZONE_COOLDOWN_SECONDS)
        self.fconf = TrackWindowConfirmer(cfg.FALL_CONFIRM_FRAMES, cfg.FALL_CONFIRM_WINDOW)
        self.fcool = CooldownGate(cfg.FALL_COOLDOWN_SECONDS)
        if self._fall is not None:      # Settings sliders were dead until now
            self._fall.mode = cfg.FALL_MODE
            self._fall.threshold = float(cfg.FALL_PROB_THRESHOLD)
        # NOTE: _required is per-camera (set via ppe_items / apply_ppe_items), so it
        # is intentionally NOT reset here — refresh only re-reads cfg confirm/cooldown.

    def _cat_hit(self, rec, cat, obs=None) -> bool:
        """Is `cat` a violation for this person THIS frame?
        Only categories this camera is set to check (self._required) can fire, and
        for those, ABSENCE counts (state != WORN) — we don't wait for an unreliable
        no_* box. Unchecked categories never fire."""
        if cat not in self._required:
            return False
        if rec["states"][cat] == "WORN":
            return False
        # ABSTENTION. The boolean gate below asks "is the body part visible"; this
        # asks the harder question "is what I can see good enough to accuse someone
        # on". A camera looking at a worker's back still sees their shoulders, so
        # the boolean gate lets a vest charge through from a view that has never
        # seen the front of the vest — measured weight 0.122 there versus 0.705
        # from the front. Below the floor this camera says nothing at all, which
        # is the honest answer and leaves the verdict to a view that can see.
        if obs is not None and getattr(cfg, "PPE_ABSTAIN_ENABLED", True):
            ev = (obs.get("cats") or {}).get(cat)
            if ev is not None:
                floor = float(getattr(cfg, "PPE_ABSTAIN_W_BY_CAT", {}).get(
                    cat, getattr(cfg, "PPE_ABSTAIN_W", 0.35)))
                if float(ev.get("w", 0.0)) < floor:
                    return False
        # Visibility gate: don't accuse a body part we cannot see. `visible` is the
        # set of judgeable categories; None = pose found no match for this person.
        if self._vis_gate:
            vis = rec.get("visible")
            if vis is None:
                # No pose at all. strict → don't accuse; lenient (default) → keep the
                # required-absence charge so a pose miss can't hide a real violation.
                if getattr(cfg, "PPE_VISIBILITY_STRICT", False):
                    return False
            elif cat not in vis:
                return False           # body part provably not visible → don't accuse
        # WORN-hold: we CONFIRMED this person wearing it moments ago; an occlusion or
        # a confidence dip now must not flip that to a violation. Checked BOTH per
        # track id AND spatially (the id churns behind occlusions — a fresh id at the
        # same spot inherits the "recently seen wearing it" verdict).
        if self._worn_hold:
            p = rec["person"]
            tid = p.get("track_id")
            if tid is not None and self._worn_until.get((tid, cat), 0.0) > self._now:
                return False
            cx = (p["x1"] + p["x2"]) / 2.0; cy = (p["y1"] + p["y2"]) / 2.0
            r = float(getattr(cfg, "PPE_WORN_HOLD_RADIUS", 0.18)) * getattr(self, "_frame_diag", 1e9)
            for (wx, wy, exp) in self._worn_places.get(cat, ()):
                if exp > self._now and (cx - wx) ** 2 + (cy - wy) ** 2 <= r * r:
                    return False
        return True                    # required, not WORN, judgeable, not recently WORN

    def _attach_poses(self, frame, persons: list) -> None:
        """Single-camera path: run ONE yolo11n-pose pass then match keypoints to
        persons by IoU. (Shared/multi-camera path gets pose pre-batched from the
        ModelHub and calls _match_poses directly.)"""
        for p in persons:
            p["kp_xy"] = p["kp_conf"] = None
        if self._pose is None or not persons:
            return
        try:
            r = self._pose.predict(frame, imgsz=int(getattr(cfg, "PPE_POSE_IMGSZ", 640)),
                                   device=getattr(self.ppe_detector, "device", None),
                                   conf=0.25, verbose=False)[0]
        except Exception as e:
            print(f"[PPEEngine] pose pass failed: {e}")
            return
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return
        xy = r.keypoints.xy.cpu().numpy()
        cf = (r.keypoints.conf.cpu().numpy()
              if r.keypoints.conf is not None else np.ones(xy.shape[:2], np.float32))
        self._match_poses(persons, r.boxes.xyxy.cpu().numpy(), xy, cf)

    def _match_poses(self, persons: list, boxes, xy, cf) -> None:
        """Attach each person's matched keypoints (kp_xy px + kp_conf) by IoU. Used
        by both the single-camera pose pass and the shared ModelHub pose result."""
        for p in persons:
            pbox = (p["x1"], p["y1"], p["x2"], p["y2"])
            best, best_iou = None, 0.3
            for i in range(len(boxes)):
                s = _iou_box({"x1": boxes[i][0], "y1": boxes[i][1],
                              "x2": boxes[i][2], "y2": boxes[i][3]},
                             {"x1": pbox[0], "y1": pbox[1], "x2": pbox[2], "y2": pbox[3]})
                if s > best_iou:
                    best, best_iou = i, s
            if best is not None:
                p["kp_xy"] = xy[best].tolist()
                p["kp_conf"] = cf[best].tolist()

    def _attach_poses_from(self, persons: list, pose_result) -> None:
        """Shared path: attach keypoints from a pre-computed ModelHub pose result
        ({boxes, xy, conf} or None) instead of running pose here."""
        for p in persons:
            p["kp_xy"] = p["kp_conf"] = None
        if not pose_result or not persons:
            return
        self._match_poses(persons, pose_result["boxes"],
                          pose_result["xy"], pose_result["conf"])

    def _annotate_visibility(self, recs: list) -> None:
        """Set rec['visible'] = judgeable PPE categories, from the keypoints already
        attached to each person by _attach_poses (no extra pose pass). None when the
        person had no pose match (see PPE_VISIBILITY_STRICT)."""
        if not self._vis_gate:
            return
        for rec in recs:
            kc = rec["person"].get("kp_conf")
            rec["visible"] = visible_cats(kc) if kc is not None else None

    def _select_crop_persons(self, recs: list, frame_h: int) -> list[dict]:
        """Pick which people get the second, zoomed PPE pass. A person qualifies if
        they are SMALL in frame (distant → every item is a few px) OR a required
        category is still UNKNOWN after the full-frame pass (we couldn't judge it).
        Someone whose required items are already WORN/VIOLATION and who fills a good
        part of the frame is skipped — the zoom would tell us nothing new.

        Bounded to PPE_CROP_MAX_PER_FRAME with a round-robin cursor so a crowd walking
        in cannot spike the per-frame cost (the temporal confirm bridges the gap)."""
        min_h = float(getattr(cfg, "PPE_CROP_MIN_PERSON_H_FRAC", 0.45)) * frame_h
        cands = []
        for rec in recs:
            p = rec["person"]
            if p.get("track_id") is None:
                continue
            is_small = (p["y2"] - p["y1"]) < min_h
            has_unknown = any(rec["states"].get(c) == "UNKNOWN" for c in self._required)
            if is_small or has_unknown:
                cands.append(p)
        cap = max(1, int(getattr(cfg, "PPE_CROP_MAX_PER_FRAME", 6)))
        if len(cands) <= cap:
            self._crop_rr = 0
            return cands
        start = self._crop_rr % len(cands)
        sel = [cands[(start + i) % len(cands)] for i in range(cap)]
        self._crop_rr = (start + cap) % len(cands)
        return sel

    @staticmethod
    def _merge_items(base: list[dict], extra: list[dict]) -> list[dict]:
        """Fold crop-pass items into the full-frame items. A duplicate — same
        category, same place (IoU > 0.6) — keeps whichever scored higher, so the
        zoomed detection supersedes a weak full-frame one instead of double-counting.
        Association is duplicate-tolerant, but this keeps the item list and overlay
        clean."""
        if not extra:
            return base
        merged = list(base)
        for e in extra:
            for i, m in enumerate(merged):
                if m.get("taxo") == e.get("taxo") and _iou_box(m, e) > 0.6:
                    if e["conf"] > m["conf"]:
                        merged[i] = e
                    break
            else:
                merged.append(e)
        return merged

    def _confirmed_missing(self, tid, rec) -> list[str]:
        """Categories currently hit AND temporally confirmed for this person —
        the single source of truth shared by detect() (alarms) and _draw()
        (labels) so the box colour never disagrees with the alert."""
        if tid is None:
            return []
        # Same abstention evidence the decision used, or the box would turn red for
        # a category this camera just declined to judge — a label contradicting the
        # alert is exactly what this function exists to prevent.
        obs = next((o for o in self.last_observations if o["track_id"] == tid), None)
        return [c for c in CATEGORIES
                if self._cat_hit(rec, c, obs) and self.pconf.is_confirmed((tid, c))]

    def close_fall(self):
        """Release the pose backend (mediapipe leaks native graphs otherwise)."""
        if self._fall is not None:
            try:
                self._fall.close()
            except Exception:
                pass
        self._fall = None
        self._fallen.clear()
        self._fall_incidents.clear()

    def fall_step(self, frame, recs, now=None):
        """Runs from the pipeline's FIXED-CADENCE fall loop, not from detect():
        the classifier wants 30 evenly-spaced frames, and the detect loop skips
        frames. Every tracked person is evaluated — several people can fall in the
        same emergency, so each track gets its own confirm window and cooldown and
        emits its own event."""
        if not self.fall_ready or not recs:
            return []
        persons = [r["person"] for r in recs if r["person"].get("track_id") is not None]
        if not persons:
            return []
        try:
            results = self._fall.step(frame, persons, now=now)
        except Exception as e:
            print(f"[PPEEngine] fall_step: {e}")
            return []

        import time as _t
        h, w = frame.shape[:2]
        diag = (w * w + h * h) ** 0.5
        radius = float(getattr(cfg, "FALL_DEDUPE_RADIUS", 0.15)) * diag
        window = float(getattr(cfg, "FALL_DEDUPE_SEC", 20.0))
        tnow = _t.time()
        self._fall_incidents = [(t, x, y) for (t, x, y) in getattr(self, "_fall_incidents", [])
                                if tnow - t <= window]

        events, live = [], set()
        for tid, res in results.items():
            k = (tid,)
            live.add(k)
            confirmed = self.fconf.update(k, res.fallen)
            # Drive the RED box off the CONFIRMED state, not the instantaneous flag,
            # so a person on the floor doesn't flicker back to normal for a frame.
            if confirmed:
                self._fallen.add(tid)
            else:
                self._fallen.discard(tid)
            if not (confirmed and res.fallen and self.fcool.ready(k)):
                continue
            # De-duplicate by PLACE, not by track id. ByteTrack churns ids on some
            # cameras, and a per-id cooldown then lets each new id re-alarm for the
            # same person on the same patch of floor.
            p = next((r["person"] for r in recs if r["person"]["track_id"] == tid), None)
            if p is None:                      # track vanished between step() and here
                continue
            cx, cy = (p["x1"] + p["x2"]) / 2.0, (p["y1"] + p["y2"]) / 2.0
            if any(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 < radius
                   for (_, x, y) in self._fall_incidents):
                continue
            self._fall_incidents.append((tnow, cx, cy))
            events.append({"type": "fall", "track_id": tid, "key": k,
                           "level": cfg.ALERT_LEVEL_EMERGENCY,
                           "msg": "⚠️ ตรวจพบการล้ม"})
        self.fconf.gc(live)
        self._fallen &= {t for (t,) in live}
        return events

    def reset(self):
        # Shared mode: reset THIS camera's own tracker (never the hub's models,
        # which other cameras share). Single mode: reset the owned detectors.
        if self._hub is not None:
            if self._bytetrack is not None:
                self._bytetrack.reset()
        else:
            if self.person_detector is not None:
                self.person_detector.reset()
            if self.ppe_detector is not None:
                self.ppe_detector.reset()
        self._zone_inside.clear()

    def detect(self, frame, now=None):
        """Heavy step: track + associate + temporal confirm. Returns (recs, events).
        Call this from ONE worker thread (the tracker is stateful/persist=True).
        `now` (seconds) drives the WORN-hold expiry; defaults to wall-clock. Offline
        harnesses pass video time so the hold is measured in video seconds."""
        self._now = time.time() if now is None else now
        h, w = frame.shape[:2]
        self._frame_diag = (h * h + w * w) ** 0.5
        # People (+ track ids) from the COCO detector; PPE items from the fine-tune.
        # Skip the PPE pass entirely when the role is off — saves a full forward.
        persons = self.person_detector.track(frame)
        # One pose pass → keypoints on each person, shared by keypoint-anchored
        # association (crowd-safe item→person) and the visibility gate.
        if self._vis_gate or self._kp_assign:
            self._attach_poses(frame, persons)
        items = (self.ppe_detector.detect_items(frame)
                 if self.ppe_enabled and self.ppe_detector is not None else [])
        # HYBRID person-crop "zoom": the full-frame pass under-resolves small/distant
        # workers (a glove is a few px). Re-run the PPE model, batched, ONLY on the
        # people it could not fully judge, and merge those items back before
        # association. Bounded per frame + round-robin so a crowd can't spike latency.
        if (self.ppe_enabled and self.ppe_detector is not None and persons
                and getattr(cfg, "PPE_CROP_ENABLED", True)):
            crop_persons = self._select_crop_persons(
                associate(persons + items, frame_h=h), frame.shape[0])
            if crop_persons:
                items = self._merge_items(
                    items, self.ppe_detector.detect_items_crops(frame, crop_persons))
        return self._confirm_and_events(persons, items, w, h)

    def step(self, frame, persons_raw, items, pose_result, now=None):
        """SHARED-HUB path (multi-camera): detections come pre-computed + batched
        from the ModelHub; this camera only assigns its OWN track ids (standalone
        ByteTracker) and runs the exact same associate/confirm/event logic as
        detect(). Keeps per-camera state fully isolated.

        Kept as the one-call form (used by the cloud server and any caller that is
        not the batching scheduler). The scheduler instead calls
        step_prepare() → hub.detect_items_crops_multi() → step_finish() so every
        camera's crops share ONE forward; the two paths produce identical results."""
        state, crop_persons = self.step_prepare(frame, persons_raw, items,
                                                pose_result, now=now)
        crop_items = (self._hub.detect_items_crops(frame, crop_persons)
                      if crop_persons else [])
        return self.step_finish(state, crop_items)

    def step_prepare(self, frame, persons_raw, items, pose_result, now=None):
        """First half of step(): per-camera track ids + poses + choosing WHICH
        people need the crop "zoom" pass. Returns (state, crop_persons).

        The split exists so the caller can collect crop_persons from every camera
        and run ONE batched crop forward for all of them, instead of one forward
        per camera. `state` is opaque — hand it straight back to step_finish()."""
        self._now = time.time() if now is None else now
        h, w = frame.shape[:2]
        self._frame_diag = (h * h + w * w) ** 0.5
        persons = self._assign_track_ids(persons_raw)
        if self._vis_gate or self._kp_assign:
            self._attach_poses_from(persons, pose_result)
        else:
            for p in persons:
                p["kp_xy"] = p["kp_conf"] = None
        items = list(items or [])
        crop_persons = []
        # WHICH people need a crop is per-camera (it depends on this camera's
        # persons); running the crops is not, so only the selection happens here.
        # Bounded by PPE_CROP_MAX_PER_FRAME so the batch stays flat.
        if (self.ppe_enabled and self._hub is not None and self._hub.ppe is not None
                and persons and getattr(cfg, "PPE_CROP_ENABLED", True)):
            crop_persons = self._select_crop_persons(
                associate(persons + items, frame_h=h), h)
        return {"persons": persons, "items": items, "w": w, "h": h}, crop_persons

    def step_finish(self, state: dict, crop_items: list):
        """Second half of step(): merge this camera's slice of the batched crop
        results back in, then run the unchanged associate/confirm/event logic."""
        items = state["items"]
        if crop_items:
            items = self._merge_items(items, crop_items)
        return self._confirm_and_events(state["persons"], items,
                                        state["w"], state["h"])

    def _assign_track_ids(self, persons_raw: list) -> list:
        """Feed this camera's person detections into its OWN ByteTracker → stable
        per-camera track ids (never shared across cameras). Converts the normalized
        det dict (x1,y1,x2,y2) ↔ ByteTracker's centre/size dict, then writes the
        track's box + id back into a person dict the rest of the engine consumes."""
        bt_in = [{"x": p["cx"], "y": p["cy"],
                  "width":  p["x2"] - p["x1"], "height": p["y2"] - p["y1"],
                  "confidence": p["conf"], "class": "person"} for p in persons_raw]
        tracks = self._bytetrack.update(bt_in)
        out = []
        for t in tracks:
            x1, y1, x2, y2 = (float(v) for v in t.bbox)
            out.append({
                "track_id": int(t.track_id), "cls": "person",
                "taxo": cfg.ppe_taxo_index("person"),
                "conf": float(t.score),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                "foot": ((x1 + x2) / 2, y2), "is_person": True,
            })
        return out

    def _observe_ppe(self, recs, h: int) -> list[dict]:
        """OBSERVE half: this camera's graded PPE evidence for the current frame.

        Pure function of the frame — no confirmers, no cooldowns, no events, no
        state touched. That is the point: an observation says only "here is what I
        can see, and how well I can see it", which is the only form of evidence
        that several cameras can meaningfully combine. Deciding what it MEANS
        belongs to whoever holds all the observations.

        `w` per category is the product of four independent reasons to trust this
        camera about this person:
            visibility  — is the relevant body part in view at all
            orientation — front-worn items are barely evidence from behind
            scale       — a person 8% of frame height cannot be judged
            detection   — how confident the box that decided it was
        A weight of 0 means NO EVIDENCE, which is emphatically not "compliant" —
        the distinction the boolean pipeline cannot represent and the reason a
        camera looking at someone's back can currently vote them safe.

        Consumed by the decision path below, which refuses to raise a violation
        for a category this camera cannot actually see (PPE_ABSTAIN_W).
        """
        obs = []
        for rec in recs:
            p = rec["person"]
            tid = p.get("track_id")
            if tid is None:
                continue
            kp_conf = p.get("kp_conf")
            # No pose for this person (pose model off, or it found no match) means
            # we do not know how well we can see them — which is NOT the same as
            # knowing we cannot. Scoring it 0 would silence such a camera entirely,
            # and if pose were unavailable site-wide EVERY observation would weigh
            # 0 and nobody could ever be judged. Fall back to a neutral
            # weight and no orientation discount, mirroring the existing gate's
            # lenient policy (PPE_VISIBILITY_STRICT defaults False: no pose → the
            # charge still stands) while ranking below a camera that can see.
            has_pose = kp_conf is not None
            facing = facing_score(kp_conf) if has_pose else None
            nopose_w = float(getattr(cfg, "PPE_NOPOSE_WEIGHT", 0.4))
            h_frac = (p["y2"] - p["y1"]) / float(h) if h else 0.0
            w_scale = scale_weight(h_frac)
            # Strongest supporting detection per (category, state) on this person.
            support: dict = {}
            for it in rec.get("items", ()):
                cs = cat_state_of(it.get("taxo"))
                if cs is None:
                    continue
                key = cs
                c = float(it.get("conf", 0.0))
                if c > support.get(key, 0.0):
                    support[key] = c
            cats = {}
            for cat in CATEGORIES:
                state = rec["states"][cat]
                vis = cat_visibility(kp_conf, cat) if has_pose else nopose_w
                w_or = orientation_weight(cat, facing) if has_pose else 1.0
                # UNKNOWN carries no box, so there is no detection confidence to
                # fold in — the visibility IS the evidence, i.e. "I could have seen
                # it and did not". WORN/VIOLATION additionally scale by how sure
                # the supporting box was.
                conf = support.get((cat, state), 0.0) if state != "UNKNOWN" else 1.0
                cats[cat] = {
                    "state": state,
                    "w": round(vis * w_or * w_scale * conf, 4),
                    "vis": round(vis, 4),
                    "conf": round(conf, 4),
                }
            obs.append({
                "track_id": int(tid),
                # Which categories THIS camera was configured to enforce.
                "required": sorted(self._required),
                "cx": (p["x1"] + p["x2"]) / 2.0,
                "cy": (p["y1"] + p["y2"]) / 2.0,
                "foot": p.get("foot"),
                "kp_xy": p.get("kp_xy"),
                "h_frac": round(h_frac, 4),
                "facing": round(facing, 4) if facing is not None else None,
                "has_pose": has_pose,
                "cats": cats,
            })
        return obs

    def _confirm_and_events(self, persons, items, w, h):
        self.last_dets = persons + items
        recs = associate(persons + items, frame_h=h)
        # Visibility gate: mark which categories each person can be JUDGED for, so
        # _cat_hit won't accuse someone whose relevant body part is hidden.
        self._annotate_visibility(recs)
        # Graded "how well can I see this?" evidence per person (see _observe_ppe).
        self.last_observations = self._observe_ppe(recs, h) if self.ppe_enabled else []
        # Index by track id so the decision below can ask "how well can I actually
        # see this person's helmet/vest" without recomputing anything.
        obs_by_tid = {o["track_id"]: o for o in self.last_observations}
        events = []
        live_p, live_z = set(), set()

        for rec in recs:
            person = rec["person"]
            tid = person["track_id"]
            if tid is None:
                continue
            # Exclusion zones (operator booth) mask a person out of ALL detection —
            # a privacy/masking primitive, applied even when the Zone ROLE is off so
            # a PPE-only camera still won't alarm people standing inside the booth.
            if zone_geometry.in_any_exclusion(person, self.zones, w, h):
                continue
            if self.ppe_enabled:
                obs = obs_by_tid.get(tid)
                for cat in CATEGORIES:
                    k = (tid, cat); live_p.add(k)
                    # Latch "was CONFIRMED wearing it" (3-of-5, so a flickering
                    # false-WORN never latches) → hold that verdict through a later
                    # occlusion instead of falsely accusing them (see PPE_WORN_HOLD).
                    if self._worn_hold and cat in self._required:
                        if self.wconf.update(k, rec["states"][cat] == "WORN"):
                            self._worn_until[k] = self._now + self._worn_hold_sec
                            # id-independent mark: a new id at this spot inherits it
                            cx = (person["x1"] + person["x2"]) / 2.0
                            cy = (person["y1"] + person["y2"]) / 2.0
                            self._worn_places.setdefault(cat, []).append(
                                (cx, cy, self._now + self._worn_hold_sec))
                    hit = self._cat_hit(rec, cat, obs)   # required: absence counts
                    confirmed = self.pconf.update(k, hit)
                    if confirmed and hit and self.pcool.ready(k):
                        # De-dup by PLACE+CATEGORY, not just per id: ByteTrack churns
                        # ids so a per-id cooldown lets the same violation re-alert
                        # every few seconds. One alert per spot+category per window.
                        cx = (person["x1"] + person["x2"]) / 2.0
                        cy = (person["y1"] + person["y2"]) / 2.0
                        r = float(getattr(cfg, "PPE_DEDUPE_RADIUS", 0.15)) * self._frame_diag
                        if any(c2 == cat and (cx - x) ** 2 + (cy - y) ** 2 < r * r
                               for (_t, c2, x, y) in self._ppe_incidents):
                            continue
                        self._ppe_incidents.append((self._now, cat, cx, cy))
                        events.append({"type": "ppe", "track_id": tid, "key": k,
                                       "level": cfg.ALERT_LEVEL_WARNING,
                                       "msg": _CAT_TH.get(cat, "ไม่สวม " + cat)})
            if self.zone_enabled:
                # Record inside/outside for EVERY danger zone every frame (like PPE)
                # so the 3-of-5 window is real. Previously only inside-frames were
                # pushed and gc() reset the key the moment the person stepped out, so
                # a worker straddling the zone edge (foot-point jitter) could never
                # reach 3-of-5 and evade the intrusion alarm.
                hit_ids = {z.id for z in zone_geometry.danger_hits(person, self.zones, w, h)}
                for z in self.zones:
                    if z.type != "danger":
                        continue
                    zk = (tid, z.id); live_z.add(zk)
                    inside = z.id in hit_ids
                    confirmed = self.zconf.update(zk, inside)   # ≥3 of last 5 inside
                    if confirmed and inside:
                        # Rising edge (fresh entry) alerts immediately; a continuous
                        # stay re-alerts every cooldown. Tracking "already inside" per
                        # (track, zone) is what makes leave→re-enter fire again — the
                        # plain 20s cooldown alone suppressed re-entries and, once the
                        # person kept one track id, effectively alarmed only once.
                        if zk not in self._zone_inside:
                            self._zone_inside.add(zk)
                            self.zcool.mark(zk)
                            fire = True
                        else:
                            fire = self.zcool.ready(zk)
                        if fire:
                            events.append({"type": "zone", "track_id": tid, "key": zk,
                                           "level": cfg.ALERT_LEVEL_ALERT,
                                           "msg": f"บุกรุกพื้นที่อันตราย '{z.name}'"})
                    elif not confirmed:
                        self._zone_inside.discard(zk)   # debounced exit → next entry re-alerts
        self.pconf.gc(live_p); self.wconf.gc(live_p); self.zconf.gc(live_z)
        # Prune WORN-holds by TIME only — NOT by track presence. A track that walks
        # behind an object briefly drops out of live_p; pruning on absence would throw
        # the hold away at exactly the moment it's needed. The 4s expiry bounds it.
        self._worn_until = {k: t for k, t in self._worn_until.items() if t > self._now}
        self._worn_places = {c: [(x, y, e) for (x, y, e) in v if e > self._now]
                             for c, v in self._worn_places.items()}
        _dwin = float(getattr(cfg, "PPE_DEDUPE_SEC", 45.0))
        self._ppe_incidents = [(t, c, x, y) for (t, c, x, y) in self._ppe_incidents
                               if self._now - t <= _dwin]
        self._zone_inside &= live_z      # forget zones for tracks that left the frame
        return recs, events

    def draw_on(self, frame, recs):
        """Light step: draw the latest recs onto ANY frame (fast — no inference).
        Lets the display run at camera FPS while detection runs slower."""
        h, w = frame.shape[:2]
        return self._draw(frame, recs or [], w, h)

    def process(self, frame):
        """Convenience (offline harness): detect + draw in one call."""
        recs, events = self.detect(frame)
        return self.draw_on(frame, recs), events

    def _rec_style(self, tid, rec):
        """The single source of truth for a person's box COLOUR + LABEL (BGR).
        Shared by _draw() (local render) and draw_items() (remote render) so the
        edge's cloud-drawn boxes look identical to local ones."""
        # A confirmed fall outranks every PPE status — it's the emergency.
        if tid in self._fallen:
            return RED, "ล้ม!"
        # Show a PPE status only when PPE is on AND this camera actually checks at
        # least one item — otherwise a neutral box, never a premature "ปลอดภัย".
        if self.ppe_enabled and self._required:
            #   RED   = a required item is confirmed missing
            #   GREEN = every required item is positively worn (verified safe)
            #   CYAN  = still verifying (not yet seen worn, not confirmed-missing)
            cviol = self._confirmed_missing(tid, rec)
            required_ok = all(rec["states"][c] == "WORN" for c in self._required)
            if cviol:
                return RED, "ขาด: " + ",".join(_CAT_SHORT_TH.get(c, c) for c in cviol)
            if required_ok:
                return GREEN, "ปลอดภัย"
            return CYAN, "กำลังตรวจสอบ"
        return CYAN, "คน"

    def draw_items(self, recs, w, h) -> dict:
        """Serialisable drawing primitives for REMOTE rendering: person boxes
        (BGR colour + label + foot) and danger-zone polygons. The cloud returns
        these so the edge can overlay them on LIVE frames — smooth video at camera
        FPS with boxes that update as the cloud responds, instead of waiting for a
        fully-annotated image every frame (the source of the 'choppy + laggy'
        remote feel)."""
        boxes, zones = [], []
        if self.zone_enabled:
            for z in self.zones:
                poly = z.polygon_px(w, h)
                col = RED if z.type == "danger" else (140, 140, 140)
                zones.append({"points": [[int(pt[0]), int(pt[1])]
                                         for pt in poly.reshape(-1, 2)],
                              "color": [int(c) for c in col], "name": z.name})
        if self.ppe_enabled or self.zone_enabled or self.fall_enabled:
            for rec in recs:
                p = rec["person"]; tid = p["track_id"]
                color, label = self._rec_style(tid, rec)
                boxes.append({
                    "x1": int(p["x1"]), "y1": int(p["y1"]),
                    "x2": int(p["x2"]), "y2": int(p["y2"]),
                    "color": [int(c) for c in color], "label": label,
                    "foot": [int(p["foot"][0]), int(p["foot"][1])],
                })
        return {"boxes": boxes, "zones": zones}

    def _draw(self, frame, recs, w, h):
        # Draw shapes with cv2 (fast); collect text to render once with PIL so
        # Thai labels show correctly (cv2.putText → "???"). Colors below are BGR.
        #
        # Every size here SCALES with the frame. They used to be fixed pixel
        # counts tuned on a 960-wide test clip, but the frame drawn on is the
        # camera's NATIVE one — 2560 or 3072 wide on the 5-6MP IP cameras this
        # runs on — and the broadcaster then shrinks it to 960 for the browser.
        # An 18px label on a 2560-wide frame arrives 8px tall and a 2px box edge
        # arrives 0.6px, which is exactly the unreadable smear an operator sees.
        # Proportional sizes make the overlay look identical at any resolution.
        s = max(1.0, w / 960.0)
        thick = max(2, int(round(2 * s)))
        size = max(12, int(round(18 * s)))
        pad = int(round(22 * s))
        dot = max(3, int(round(4 * s)))

        texts = []   # (x, y_top, text, rgb)
        if self.zone_enabled:
            for z in self.zones:
                poly = z.polygon_px(w, h)
                col = RED if z.type == "danger" else (140, 140, 140)
                cv2.polylines(frame, [poly], True, col, thick)
                x0, y0 = int(poly[0][0]), int(poly[0][1])
                texts.append((x0, max(2, y0 - pad), z.name, (col[2], col[1], col[0])))
        # Person boxes are relevant to both PPE and Zone; skip entirely only when
        # neither role is on. PPE status/colour is shown only when PPE is enabled.
        if self.ppe_enabled or self.zone_enabled or self.fall_enabled:
            for rec in recs:
                p = rec["person"]; tid = p["track_id"]
                x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
                color, label = self._rec_style(tid, rec)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                texts.append((x1, max(2, y1 - pad), label, (color[2], color[1], color[0])))
                cv2.circle(frame, (int(p['foot'][0]), int(p['foot'][1])), dot, color, -1)
        return draw_texts_on(frame, texts, size)

    def _draw_texts(self, frame, items, size: int = 18):
        return draw_texts_on(frame, items, size)
