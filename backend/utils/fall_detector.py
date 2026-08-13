#!/usr/bin/env python3
# utils/fall_detector.py — per-track fall detection.
# ================================================================
# Pose sequence → pretrained Transformer (TFLite), plus a resolution-independent
# rule layer. Ported from punpayut/Fall-Detection (MIT):
#   https://github.com/punpayut/Fall-Detection
# The feature contract is copied VERBATIM from their deployment/raspberry_pi/
# fall-detector.py so the shipped weights see exactly what they were trained on:
#   17 keypoints, names SORTED ALPHABETICALLY → 51 features (x, y, conf) laid out
#   at [3i, 3i+1, 3i+2]; per-frame normalisation subtracts the hip midpoint and
#   divides x,y (never conf) by |mid_shoulder_y − mid_hip_y|; a frame with no pose
#   contributes zeros; a deque of 30 consecutive frames is the model input
#   (1, 30, 51) float32 → one fall probability.
#
# Two things upstream never had to solve, and we do:
#  * MULTI-PERSON. A real emergency can put two people on the floor. Every tracked
#    person gets their own deque / confirmation / cooldown, and we never cap, evict
#    or skip a live track.
#  * DISTANCE. Measured on this project's footage, MediaPipe finds landmarks for
#    ~88% of people ≥150px wide but only ~25% of people below that; YOLO-pose gets
#    ~75% of the distant ones. So pose alone silently blinds us to far-away
#    workers, and the rule layer (prone + motionless/sudden-drop) is mandatory,
#    not a nicety.
# ================================================================
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config as cfg

# ── Upstream feature contract (do not "clean up" — the weights depend on it) ──
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]
SORTED_NAMES = sorted(KEYPOINT_NAMES)
KP_INDEX = {n: i for i, n in enumerate(SORTED_NAMES)}
NUM_FEATURES = len(SORTED_NAMES) * 3          # 51
INPUT_TIMESTEPS = 30
MIN_KP_CONF = 0.3                              # for the normalisation reference only

# COCO-17 (ultralytics pose) happens to be exactly KEYPOINT_NAMES, in order.
_COCO_ORDER = KEYPOINT_NAMES


def _cols(name: str) -> tuple[int, int, int]:
    i = KP_INDEX[name]
    return 3 * i, 3 * i + 1, 3 * i + 2


def normalize_skeleton_frame(f: np.ndarray, min_confidence: float = MIN_KP_CONF) -> np.ndarray:
    """Verbatim port of upstream `normalize_skeleton_frame`.

    Hip-centred, torso-scaled → translation/scale invariant, which is also why a
    crop and a full frame produce the same vector, and why YOLO's pixel keypoints
    and MediaPipe's 0-1 keypoints both work."""
    out = np.copy(f)
    lsx, lsy, lsc = _cols("Left Shoulder")
    rsx, rsy, rsc = _cols("Right Shoulder")
    lhx, lhy, lhc = _cols("Left Hip")
    rhx, rhy, rhc = _cols("Right Hip")

    mid_sh_x = mid_sh_y = np.nan
    v_ls, v_rs = f[lsc] > min_confidence, f[rsc] > min_confidence
    if v_ls and v_rs:
        mid_sh_x, mid_sh_y = (f[lsx] + f[rsx]) / 2, (f[lsy] + f[rsy]) / 2
    elif v_ls:
        mid_sh_x, mid_sh_y = f[lsx], f[lsy]
    elif v_rs:
        mid_sh_x, mid_sh_y = f[rsx], f[rsy]

    mid_hip_x = mid_hip_y = np.nan
    v_lh, v_rh = f[lhc] > min_confidence, f[rhc] > min_confidence
    if v_lh and v_rh:
        mid_hip_x, mid_hip_y = (f[lhx] + f[rhx]) / 2, (f[lhy] + f[rhy]) / 2
    elif v_lh:
        mid_hip_x, mid_hip_y = f[lhx], f[lhy]
    elif v_rh:
        mid_hip_x, mid_hip_y = f[rhx], f[rhy]

    if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
        return f                                    # upstream returns the RAW vector

    ref_h = np.nan
    if not np.isnan(mid_sh_y) and not np.isnan(mid_hip_y):
        ref_h = np.abs(mid_sh_y - mid_hip_y)
    scale = not (np.isnan(ref_h) or ref_h < 1e-5)

    for n in SORTED_NAMES:
        xc, yc, _ = _cols(n)
        out[xc] -= mid_hip_x
        out[yc] -= mid_hip_y
        if scale:
            out[xc] /= ref_h
            out[yc] /= ref_h
    return out


def _has_pose(f: np.ndarray) -> bool:
    """A frame carried real landmarks (upstream signals 'no pose' with all-zeros)."""
    return bool(np.any(f))


def pose_is_normalizable(f: np.ndarray, min_confidence: float = MIN_KP_CONF) -> bool:
    """Can this skeleton actually be hip-centred and torso-scaled?

    Without a hip AND a shoulder, `normalize_skeleton_frame` falls through and
    returns the vector untouched — a different distribution from every frame the
    model was trained on. A close-up of someone's head produces exactly that, and
    it must count as "no pose", not as a usable observation."""
    _, _, lhc = _cols("Left Hip");      _, _, rhc = _cols("Right Hip")
    _, _, lsc = _cols("Left Shoulder"); _, _, rsc = _cols("Right Shoulder")
    hip = f[lhc] > min_confidence or f[rhc] > min_confidence
    sho = f[lsc] > min_confidence or f[rsc] > min_confidence
    return bool(hip and sho)


# ────────────────────── anatomy, in FRAME-PIXEL space ─────────────────────────
# The TFLite feature vector is hip-centred and torso-scaled on purpose: that makes
# it translation- and scale-invariant, which is exactly what the classifier wants
# and exactly what a physics rule cannot use. Absolute position (for a velocity)
# and absolute orientation (for a torso angle) are the two things that
# normalisation throws away.
#
# So the rule layer reads the RAW keypoint vector on a separate path and converts
# it to frame pixels. `st.seq` — the model's input — is untouched.

CROP_PAD = 0.08          # context padding around a person box, for the crop backends


def crop_box(p: dict, W: int, H: int) -> tuple[int, int, int, int]:
    """The padded person crop. Shared by _MediaPipeExtractor (which feeds pose this
    exact region) and `anatomy` (which must invert it) so the two cannot drift out
    of sync — if they did, every MediaPipe hip would land in the wrong place."""
    x1, y1 = max(0, int(p["x1"])), max(0, int(p["y1"]))
    x2, y2 = min(W, int(p["x2"])), min(H, int(p["y2"]))
    pw, ph = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
    return (max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph))


def _midpoint(f: np.ndarray, left: str, right: str,
              min_conf: float = MIN_KP_CONF) -> Optional[tuple[float, float]]:
    """Midpoint of a symmetric keypoint pair, tolerating one missing side (a person
    turned side-on shows only one hip). Same fallback ladder as upstream's
    normalisation, so the two agree on what counts as "visible"."""
    lx, ly, lc = _cols(left)
    rx, ry, rc = _cols(right)
    vl, vr = f[lc] > min_conf, f[rc] > min_conf
    if vl and vr:
        return (float(f[lx] + f[rx]) / 2, float(f[ly] + f[ry]) / 2)
    if vl:
        return (float(f[lx]), float(f[ly]))
    if vr:
        return (float(f[rx]), float(f[ry]))
    return None


class Anatomy:
    """Mid-hip, mid-shoulder and torso orientation, in frame pixels."""
    __slots__ = ("hip", "shoulder", "torso_deg", "torso_len")

    def __init__(self, hip, shoulder, torso_deg, torso_len):
        self.hip, self.shoulder = hip, shoulder
        self.torso_deg, self.torso_len = torso_deg, torso_len


def anatomy(f: np.ndarray, p: dict, W: int, H: int, space: str) -> Optional[Anatomy]:
    """Raw keypoints → mid-hip/mid-shoulder in frame pixels + torso angle.

    `space` says what the extractor's x,y mean, and getting it wrong is silent:
      "frame" (YOLO-pose) — already 0-1 of the whole frame.
      "crop"  (MediaPipe) — 0-1 of the PADDED PERSON CROP, which moves with the
              person. Read as frame coordinates, a falling worker's hip would look
              almost stationary (the crop falls with them) and the velocity would
              be ~0 for every fall. So the crop transform is inverted here.

    x and y are scaled back by W and H SEPARATELY. Skipping that would measure the
    angle in a space stretched by the frame's aspect ratio — on 16:9 a true 45°
    torso reads as ~28°, and the upright/prone thresholds would silently mean
    different things on different cameras.
    """
    hip = _midpoint(f, "Left Hip", "Right Hip")
    sho = _midpoint(f, "Left Shoulder", "Right Shoulder")
    if hip is None or sho is None:
        return None

    if space == "frame":
        sx, sy, ox, oy = float(W), float(H), 0.0, 0.0
    else:                                   # "crop"
        cx1, cy1, cx2, cy2 = crop_box(p, W, H)
        sx, sy, ox, oy = float(cx2 - cx1), float(cy2 - cy1), float(cx1), float(cy1)
        if sx < 1 or sy < 1:
            return None

    hx, hy = ox + hip[0] * sx, oy + hip[1] * sy
    shx, shy = ox + sho[0] * sx, oy + sho[1] * sy
    dx, dy = shx - hx, shy - hy
    torso_len = float(np.hypot(dx, dy))
    if torso_len < 1.0:
        return None                         # shoulders on top of hips → no direction

    # Angle of the torso away from VERTICAL, in degrees.
    # Image y grows DOWNWARD, so an upright person has their shoulder ABOVE their
    # hip (dy < 0) and -dy is positive → atan2(0, +len) = 0°.
    #   0°   perfectly upright
    #   90°  lying flat
    #   >90° head lower than hips (head-first fall — also a fall)
    torso_deg = float(np.degrees(np.arctan2(abs(dx), -dy)))
    return Anatomy((hx, hy), (shx, shy), torso_deg, torso_len)


# ────────────────────────── pose extractors ──────────────────────────
class _MediaPipeExtractor:
    """Faithful-to-upstream backend. MediaPipe Pose is single-person, so we run it
    on each tracked person's crop. Instances are NOT thread-safe → one per worker
    thread; sized so we still fit the loop budget with several people on screen
    (measured: 4 workers → 8 people in ~44 ms)."""

    name = "mediapipe"
    space = "crop"      # landmarks are 0-1 of the PERSON CROP, not of the frame

    def __init__(self, workers: int = 4, complexity: int = 1):
        import mediapipe as mp
        from concurrent.futures import ThreadPoolExecutor
        self._mp = mp
        self._complexity = complexity
        self._local = threading.local()
        self._workers = max(1, workers)
        # ONE long-lived pool. A fresh executor per call would spawn fresh threads,
        # and the thread-local below would then build a brand-new mediapipe graph
        # every frame and never close it — that leaks native resources until the
        # process aborts (observed).
        self._ex = ThreadPoolExecutor(max_workers=self._workers,
                                      thread_name_prefix="fall-pose")
        self._lm_index = {
            n: getattr(mp.solutions.pose.PoseLandmark, n.upper().replace(" ", "_")).value
            for n in KEYPOINT_NAMES
        }

    def _pose(self):
        p = getattr(self._local, "pose", None)
        if p is None:
            p = self._mp.solutions.pose.Pose(
                static_image_mode=True, model_complexity=self._complexity,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._local.pose = p
        return p

    def set_complexity(self, c: int):
        """Degrade under load (18.3ms → 13.7ms/person) rather than drop a person."""
        if c != self._complexity:
            self._complexity = c
            self._rebuild()

    def _rebuild(self):
        """Tear the pool down cleanly so the old graphs are released, then let the
        new worker threads build fresh Pose instances on first use."""
        from concurrent.futures import ThreadPoolExecutor
        self.close()
        self._local = threading.local()
        self._ex = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="fall-pose")

    def close(self):
        ex, self._ex = getattr(self, "_ex", None), None
        if ex is not None:
            ex.shutdown(wait=True)

    def _one(self, crop_bgr) -> np.ndarray:
        f = np.zeros(NUM_FEATURES, dtype=np.float32)
        if crop_bgr is None or crop_bgr.size == 0:
            return f
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose().process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            for n in KEYPOINT_NAMES:
                lm = lms[self._lm_index[n]]
                xc, yc, cc = _cols(n)
                f[xc], f[yc], f[cc] = lm.x, lm.y, lm.visibility
        return f

    def extract(self, frame_bgr, persons: list[dict]) -> dict[int, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        crops, tids = [], []
        for p in persons:
            # crop_box() is the single definition of this region — `anatomy` inverts
            # exactly this transform to recover frame pixels, so both must call it.
            cx1, cy1, cx2, cy2 = crop_box(p, w, h)
            crops.append(frame_bgr[cy1:cy2, cx1:cx2])
            tids.append(p["track_id"])
        if not crops:
            return {}
        if len(crops) == 1:                       # avoid pool hand-off for the common case
            return {tids[0]: self._one(crops[0])}
        return dict(zip(tids, self._ex.map(self._one, crops)))


class _YoloPoseExtractor:
    """One forward pass yields keypoints for EVERY person — multi-person for free,
    and measurably more robust on distant workers (75% vs MediaPipe's 25%).
    Ultralytics' COCO-17 order is identical to upstream's 17 names."""

    name = "yolo"
    space = "frame"     # keypoints are rescaled to 0-1 of the whole frame below

    def __init__(self, model_path: str, device: Optional[str] = None,
                 imgsz: Optional[int] = None):
        from ultralytics import YOLO
        from utils.detect_track import _device
        p = Path(model_path)
        self.model_path = str(p) if p.exists() else (p.name or "yolo11n-pose.pt")
        self.device = device or _device()
        self.model = YOLO(self.model_path)
        # 960 costs 396 ms/frame on a CPU box; 640 costs 263 ms and the fall loop
        # has a 100 ms budget. Make it tunable instead of hard-coding the slow one.
        self.imgsz = imgsz or int(getattr(cfg, "FALL_POSE_IMGSZ", 640))

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def extract(self, frame_bgr, persons: list[dict]) -> dict[int, np.ndarray]:
        out = {p["track_id"]: np.zeros(NUM_FEATURES, dtype=np.float32) for p in persons}
        if not persons:
            return out
        H, W = frame_bgr.shape[:2]
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=0.25,
                               device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return out
        # Ultralytics returns PIXELS; MediaPipe returns 0-1. normalize_skeleton_frame
        # has fallback branches (no valid hip, or shoulder_y == hip_y) that return the
        # vector UNSCALED — for pixels that is hundreds, far outside the ~[-3,3] the
        # Transformer was trained on. Convert here so both backends are identical in
        # every branch, not just the happy path.
        kxy = r.keypoints.xy.cpu().numpy().astype(np.float32)   # (N,17,2)
        kxy[..., 0] /= max(1, W)
        kxy[..., 1] /= max(1, H)
        kconf = (r.keypoints.conf.cpu().numpy()
                 if r.keypoints.conf is not None else np.ones(kxy.shape[:2], np.float32))
        pboxes = r.boxes.xyxy.cpu().numpy()
        used = set()
        for p in persons:                                   # assign each track its own skeleton
            box = (p["x1"], p["y1"], p["x2"], p["y2"])
            best, best_iou = -1, 0.3
            for j in range(len(pboxes)):
                if j in used:
                    continue
                s = self._iou(box, pboxes[j])
                if s > best_iou:
                    best, best_iou = j, s
            if best < 0:
                continue
            used.add(best)
            f = np.zeros(NUM_FEATURES, dtype=np.float32)
            for i, n in enumerate(_COCO_ORDER):
                xc, yc, cc = _cols(n)
                f[xc], f[yc], f[cc] = kxy[best, i, 0], kxy[best, i, 1], kconf[best, i]
            out[p["track_id"]] = f
        return out


# ────────────────────────── per-track state ──────────────────────────
class _TrackState:
    __slots__ = ("seq", "pose_hits", "hist", "anat", "prone_since", "transition_ok",
                 "p_fall", "last_box", "p_hist", "last_prone_t")

    def __init__(self):
        # (t, 51-vector) and (t, 1.0|0.0) — TIME-STAMPED, not a plain 30-frame
        # deque. The classifier is trained on 30 steps spanning ~2 s, and the
        # fall loop can only step when detection hands it a new frame, so its
        # real cadence is the shared scheduler's per-camera rate: measured at
        # 9.6 Hz with one camera and 4.6 Hz with two. As a bare 30-deque that
        # stretched the window from 2.0 s to 6.5 s, and a one-second fall then
        # occupies ~5 of the 30 steps instead of filling them — the motion the
        # model looks for is flattened away, so fall detection quietly stops
        # working as cameras are added. Sampling these onto a fixed time grid in
        # _predict() keeps the window 2 s wide at any arrival rate.
        # maxlen covers the window twice over even at the fastest cadence.
        self.seq: deque = deque(maxlen=INPUT_TIMESTEPS * 2)
        self.pose_hits: deque = deque(maxlen=INPUT_TIMESTEPS * 2)
        # (t, p_fall) for ticks where the window was pose-covered enough to make a
        # REAL prediction. The classifier's verdict outlives the pose that produced
        # it — see FALL_VERDICT_HOLD_SEC.
        self.p_hist: deque = deque(maxlen=200)
        # Last tick this person looked prone. Lets the "stayed down" timer survive a
        # dropped frame instead of restarting from zero (FALL_PRONE_GRACE_SEC).
        self.last_prone_t: Optional[float] = None
        # (t, aspect_ratio, cx, cy) — cx/cy normalized by frame WIDTH/HEIGHT.
        # Long enough to hold the upright baseline window.
        self.hist: deque = deque(maxlen=200)
        # (t, hip_y, torso_deg) — hip_y as a fraction of FRAME HEIGHT, so a velocity
        # derived from it is in frame-heights/sec and means the same thing on a 720p
        # camera and a 4K one. Only appended on frames where pose actually resolved,
        # so a gap in this deque is an honest "we could not see them", not a zero.
        self.anat: deque = deque(maxlen=200)
        self.prone_since: Optional[float] = None
        self.transition_ok: bool = False
        self.p_fall: float = 0.0
        self.last_box: Optional[tuple] = None      # for rebinding across an id switch


def body_span(p: dict, W: int, H: int) -> float:
    """How big is this person, as a fraction of the frame diagonal.

    Deliberately uses the box DIAGONAL: it is the same whether the person is
    standing or lying down. Box height is not — a fallen person's box is short by
    definition, which is why gating on height vetoed the very falls we exist to
    catch, and why gating on "height while upright" needs an upright history that a
    mid-fall id switch destroys. Measured: real falls span 0.04–0.51 of the
    diagonal; a face filling the lens spans 0.72."""
    w = float(p["x2"] - p["x1"])
    h = float(p["y2"] - p["y1"])
    diag = float(np.hypot(W, H)) or 1.0
    return float(np.hypot(w, h)) / diag


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ar_now(st: "_TrackState") -> float:
    """This person's current width/height. History-free by design."""
    return st.hist[-1][1] if st.hist else 0.0


def hip_velocity(st: "_TrackState", now: float, window: float,
                 min_dt: float = 0.15) -> float:
    """Steepest DOWNWARD speed of the mid-hip over the last `window` seconds, in
    FRAME HEIGHTS per second. Positive = falling. 0.0 when pose never resolved.

    This is the signal that separates a fall from sitting, kneeling and crouching.
    All four move the hip roughly the same DISTANCE — which is why the box-centre
    displacement the old rule used cannot tell them apart — but a fall covers that
    distance in a fraction of a second and the others take one or more.

    Taken as the steepest pair rather than the average so a fall that begins
    mid-window is not diluted by the flat frames before it. Pairs closer than
    `min_dt` apart are ignored: over one tick the hip moves a few pixels, and
    dividing normal keypoint jitter by a ~0.1 s dt manufactures a large fake
    velocity out of nothing.
    """
    pts = [(t, y) for (t, y, _deg) in st.anat if now - t <= window]
    if len(pts) < 2:
        return 0.0
    best = 0.0
    for i in range(len(pts) - 1):
        t0, y0 = pts[i]
        for j in range(i + 1, len(pts)):
            dt = pts[j][0] - t0
            if dt >= min_dt:
                best = max(best, (pts[j][1] - y0) / dt)
    return float(best)


def drop_rate(st: "_TrackState", now: float, window: float,
              min_dt: float = 0.15) -> float:
    """Steepest DOWNWARD speed of the person's BOX CENTRE, in frame heights/second.

    The same measurement as `hip_velocity`, taken from a witness that never blinks:
    `st.hist` is appended on every single tick whether or not pose resolved, so this
    is available at any frame rate and on any worker the pose model cannot read.

    It is coarser than the hip — the box centre also moves when a person merely bends
    — but it is still a RATE, and that is what matters: it separates a fall from
    someone lowering themselves to the floor, which a raw displacement cannot.
    """
    pts = [(t, cy) for (t, _ar, _cx, cy) in st.hist if now - t <= window]
    if len(pts) < 2:
        return 0.0
    best = 0.0
    for i in range(len(pts) - 1):
        t0, y0 = pts[i]
        for j in range(i + 1, len(pts)):
            dt = pts[j][0] - t0
            if dt >= min_dt:
                best = max(best, (pts[j][1] - y0) / dt)
    return float(best)


def ar_grew(st: "_TrackState", baseline: float) -> bool:
    """Has this person's box widened meaningfully versus their own upright shape?
    Corroboration for the model — a fall changes the silhouette; sitting at a desk
    with the camera in your face does not (there is no upright baseline at all)."""
    if not st.hist or not baseline:
        return False
    ar = st.hist[-1][1]
    return ar >= float(getattr(cfg, "FALL_AR_CORROBORATE", 1.25)) * baseline


class FallResult:
    __slots__ = ("p_fall", "prone", "prone_for", "pose_coverage", "drop", "body_ok",
                 "baseline", "transition", "fallen", "torso_deg", "hip_v",
                 "was_upright", "went_down_fast", "still_down", "transition_ok")

    def __init__(self, p_fall, prone, prone_for, pose_coverage, drop, body_ok,
                 baseline, transition, fallen, torso_deg=None, hip_v=0.0,
                 was_upright=False, went_down_fast=False, still_down=False,
                 transition_ok=False):
        self.p_fall, self.prone, self.prone_for = p_fall, prone, prone_for
        self.pose_coverage, self.drop, self.body_ok = pose_coverage, drop, body_ok
        self.baseline, self.transition, self.fallen = baseline, transition, fallen
        # `transition` is an AND of four independent facts. Reporting only the AND
        # tells an investigator that the fall was rejected but not BY WHICH TERM —
        # and the evaluation harness was left to GUESS between "never seen upright"
        # and "did not stay down long enough". Those two demand opposite fixes, so
        # the guess sent real debugging effort at the wrong threshold. Each term is
        # now observable on its own.
        self.was_upright = was_upright        # seen standing within FALL_BASELINE_SEC
        self.went_down_fast = went_down_fast  # hip OR box centre fell fast enough
        self.still_down = still_down          # down now (or within the grace window)
        # THE LATCH — `st.transition_ok`, and NOT the same thing as
        # (was_upright and went_down_fast). Those two are reported as "was it ever
        # true?"; the latch demands they be true SIMULTANEOUSLY, at a tick where the
        # person is already prone, and it is CLEARED whenever they stop looking prone
        # for longer than the grace window. Reporting only the two ingredients made
        # the harness blame FALL_MOTIONLESS_SEC on clips whose prone_for had in fact
        # reached 4.1 s — the timer was fine; the latch had never closed.
        self.transition_ok = transition_ok
        # None = pose could not resolve a torso this frame (too far / occluded), which
        # is different from 0° (resolved, and perfectly upright). The evaluation
        # harness needs to tell those two apart to explain a miss.
        self.torso_deg = torso_deg
        self.hip_v = hip_v              # downward frame-heights/sec, 0 if unknown


class FallDetector:
    """Per-track fall detection. `step()` is called from the pipeline's fixed-cadence
    fall loop so the 30-frame window always spans the duration the model was trained
    on — the PPE detect loop skips frames and would stretch a fall to ~2.3 s."""

    def __init__(self, backend: Optional[str] = None, tflite_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.threshold = float(getattr(cfg, "FALL_PROB_THRESHOLD", 0.90))
        self.min_coverage = float(getattr(cfg, "FALL_MIN_POSE_COVERAGE", 0.6))
        self.mode = getattr(cfg, "FALL_MODE", "hybrid")
        self._states: dict[int, _TrackState] = {}
        self._lost: dict[int, _TrackState] = {}    # recently-vanished tracks, for rebinding
        # The classifier's input window, in TIME. Anchored to the cadence the
        # fall loop aims for (FALL_LOOP_FPS), which is the cadence the model was
        # trained at — so the window is 30/15 = 2.0 s and stays 2.0 s whether the
        # box is feeding this detector 15 times a second or 4.
        self._seq_dt = 1.0 / max(1.0, float(getattr(cfg, "FALL_LOOP_FPS", 15)))
        self._seq_window = INPUT_TIMESTEPS * self._seq_dt
        self._seq_min_samples = int(getattr(cfg, "FALL_SEQ_MIN_SAMPLES", 6))
        self._seq_max_hold = float(getattr(cfg, "FALL_SEQ_MAX_HOLD_SEC", 0.4))

        backend = (backend or getattr(cfg, "FALL_POSE_BACKEND", "yolo")).lower()
        if backend == "mediapipe":
            self.extractor = _MediaPipeExtractor(
                workers=int(getattr(cfg, "FALL_POSE_WORKERS", 4)),
                complexity=int(getattr(cfg, "MEDIAPIPE_MODEL_COMPLEXITY", 1)))
        else:
            self.extractor = _YoloPoseExtractor(
                getattr(cfg, "FALL_POSE_MODEL", "yolo11n-pose.pt"), device=device)

        path = tflite_path or getattr(cfg, "FALL_TFLITE_PATH", "")
        from ai_edge_litert.interpreter import Interpreter
        self._it = Interpreter(model_path=str(path))
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        self._out = self._it.get_output_details()[0]
        got = tuple(int(v) for v in self._in["shape"])
        if got != (1, INPUT_TIMESTEPS, NUM_FEATURES):
            raise RuntimeError(f"fall model expects {got}, code assumes "
                               f"(1,{INPUT_TIMESTEPS},{NUM_FEATURES})")
        self._lock = threading.Lock()

    def close(self):
        """Release the pose backend's native graphs (mediapipe leaks without this)."""
        c = getattr(self.extractor, "close", None)
        if callable(c):
            c()

    # ── ID churn is worst exactly when it hurts most: during the fall ─────────
    def _state_for(self, tid: int, p: dict, now: float) -> _TrackState:
        """A falling body flips its box from tall to wide in a few frames; the IoU
        between consecutive boxes collapses and ByteTrack happily issues a fresh id
        (the live camera churned #515→#933 in minutes). A fresh id used to mean a
        fresh, empty history — losing the upright baseline, the pose sequence, and
        with them the transition that *is* the fall. So when an id we have never
        seen appears where an id we just lost disappeared, it is the same person:
        adopt the state instead of starting from nothing."""
        st = self._states.get(tid)
        if st is not None:
            return st
        box = (p["x1"], p["y1"], p["x2"], p["y2"])
        ttl = float(getattr(cfg, "FALL_REBIND_SEC", 3.0))
        for old in [k for k, s in self._lost.items()
                    if s.last_box is None or now - s.last_box[4] > ttl]:
            self._lost.pop(old, None)
        best, best_iou = None, float(getattr(cfg, "FALL_REBIND_IOU", 0.2))
        for old_tid, s in self._lost.items():
            score = iou(box, s.last_box[:4])
            if score > best_iou:
                best, best_iou = old_tid, score
        st = self._lost.pop(best) if best is not None else _TrackState()
        self._states[tid] = st
        return st

    # ── rule layer: a fall is a TRANSITION, not a posture ──────────────────────
    def _posture(self, st: _TrackState, p: dict, W: int, H: int, now: float,
                 anat: Optional[Anatomy]):
        """Returns (body_ok, prone, prone_for, drop, baseline, transition,
                    torso_deg, hip_v).

        Three signals, ordered by how much they can be trusted:

          1. TORSO ANGLE (needs pose) — the ONLY one that separates "sitting on the
             floor" from "lying on the floor". Both give a wide box with a low
             centre; only the person who is LYING has a horizontal torso. Gate 8
             demands zero false positives over five minutes of sitting on the floor,
             and no rule built on box shape can pass that test even in principle.
          2. HIP VELOCITY (needs pose) — a fall is FAST. Sitting, kneeling and
             crouching move the hip the same distance, over seconds. Only a rate
             tells them apart; a displacement cannot.
          3. ASPECT RATIO (box only) — the fallback for a worker too far away for
             pose to read, measured against that person's own upright baseline.

        Two dead ends are baked into this comment so they are not walked again.
        v1 tested a posture in isolation — `w/h > 0.72 and still` — so a seated
        person under a close-up lens alarmed forever (28 false emergencies). Signal
        1 exists precisely to make that class of error impossible.
        v2 over-corrected: EVERY gate hung off "this person's own upright
        baseline", which a fall destroys. The instant ByteTrack reassigns the id
        mid-fall (it does: the box flips from tall to wide, IoU collapses), the new
        track has never been seen upright, so `box_ok` was False for the rest of its
        life and the alarm could not fire at any probability. Measured: a track that
        starts already-prone reaches p_fall 0.988 and stays silent.

        So body plausibility is judged by the box DIAGONAL, which does not care
        which way up the person is and needs no history. The upright baseline is
        still computed — but only the *rule* layer, which detects a transition and
        therefore genuinely needs a "before", depends on it."""
        w_px = max(1.0, p["x2"] - p["x1"])
        h_px = max(1.0, p["y2"] - p["y1"])
        ar = w_px / h_px
        cx = (p["x1"] + p["x2"]) / 2.0 / max(1, W)       # was divided by H (bug)
        cy = (p["y1"] + p["y2"]) / 2.0 / max(1, H)

        st.hist.append((now, ar, cx, cy))
        base_sec = float(getattr(cfg, "FALL_BASELINE_SEC", 10.0))
        while st.hist and now - st.hist[0][0] > base_sec:
            st.hist.popleft()

        # Anatomy is appended ONLY when pose actually resolved a torso, so a gap in
        # this deque is an honest "we could not see them" — never a zero that would
        # read as "their hip was at the top of the frame".
        torso_deg = None
        if anat is not None:
            torso_deg = anat.torso_deg
            st.anat.append((now, anat.hip[1] / max(1, H), torso_deg))
        while st.anat and now - st.anat[0][0] > base_sec:
            st.anat.popleft()

        upright_ar = float(getattr(cfg, "FALL_UPRIGHT_AR", 0.8))
        upright_deg = float(getattr(cfg, "FALL_UPRIGHT_ANGLE", 35.0))
        prone_deg = float(getattr(cfg, "FALL_PRONE_ANGLE", 60.0))

        upright = [e for e in st.hist if e[1] <= upright_ar]
        baseline = float(np.median([e[1] for e in upright])) if upright else None

        # Is this a plausible human body, right now, in this frame? Orientation-free.
        span = body_span(p, W, H)
        body_ok = (float(getattr(cfg, "FALL_MIN_BODY_SPAN", 0.05)) <= span
                   <= float(getattr(cfg, "FALL_MAX_BODY_SPAN", 0.60)))

        # ── Prone: EITHER witness will do ────────────────────────────────────
        # Measured (URFD, 2026-07-14): a COCO-pretrained pose model — n, s AND m —
        # cannot find a person lying on the floor. yolo11n/m return no detection at
        # all; yolo11s returns a box whose shoulder and hip keypoints score 0.08 and
        # 0.17, far under MIN_KP_CONF. This is not a model-size problem, it is what
        # COCO keypoints are made of: upright people.
        #
        # So the torso goes blind at exactly the moment the person is on the ground.
        # An earlier version of this made `prone` mean "torso says so, and if we
        # can't see the torso, the box" — which read as NOT prone the instant pose
        # died, i.e. throughout the entire time the person lay there.
        #
        # Take EITHER witness. The box stays wide on the ground, where pose is blind;
        # the torso rotates even when the box does not, for someone falling towards
        # or away from the lens. Sitting on the floor also gives a wide box — that is
        # what the VELOCITY gate below is for, and it is the gate that does the
        # discriminating, not this one.
        spike = float(getattr(cfg, "FALL_AR_SPIKE", 1.8))
        abs_min = float(getattr(cfg, "FALL_AR_ABS_MIN", 0.9))
        thr = max(abs_min, spike * baseline) if baseline else abs_min
        prone = (ar > thr) or (torso_deg is not None and torso_deg > prone_deg)

        # Velocity is measured over the TRANSITION window, not a short one, and this
        # is load-bearing. A person only becomes `prone` once they have LANDED — and
        # by then pose has been blind for several ticks, so a 0.5 s lookback contains
        # no hip samples at all and reports 0.00 for every real fall. Measured on
        # fall-02: the hip genuinely dropped at 0.35 h/s during the descent, then the
        # gate sampled 0.00 at touchdown and threw the fall away. The descent is the
        # evidence; look back far enough to still see it.
        tw = float(getattr(cfg, "FALL_TRANSITION_SEC", 1.5))
        hip_v = hip_velocity(st, now, tw)

        # The two facts the latch below is made of, evaluated EVERY tick rather than
        # only inside the latch — so they can be reported even on a track that never
        # latches. That is the whole point: a miss must be able to name the term that
        # rejected it. (Both are O(window²) in pure Python over ~100 samples ≈ a few
        # ms; the fall loop's real cost is the 263 ms pose pass, so this is noise.)
        look = float(getattr(cfg, "FALL_BASELINE_SEC", 10.0))
        was_upright = bool(
            any(d <= upright_deg for (t_, _y, d) in st.anat if now - t_ <= look)
            or any(e[1] <= upright_ar for e in st.hist if now - e[0] <= look))
        peak_v = hip_velocity(st, now, look)     # 0.0 if pose never landed
        box_v = drop_rate(st, now, look)         # always available (box, not pose)
        went_down_fast = bool(
            peak_v >= float(getattr(cfg, "FALL_HIP_VELOCITY", 0.35))
            or box_v >= float(getattr(cfg, "FALL_BOX_DROP_RATE", 0.15)))

        grace = float(getattr(cfg, "FALL_PRONE_GRACE_SEC", 0.6))
        if prone:
            st.last_prone_t = now
            if st.prone_since is None:
                st.prone_since = now
            # Re-asked EVERY tick the person is down, and LATCHED once true.
            #
            # It used to be asked once, at the instant they first went prone, and
            # whatever it answered was frozen for the rest of the track's life. One
            # unlucky instant therefore disqualified a real fall permanently. That
            # instant is easy to hit: in the live app the models take ~10 s to load,
            # so the track often BEGINS with the person already on the floor — no
            # upright history yet, transition_ok=False, and it could never recover
            # even after the person stood up and fell again. Observed in the running
            # pipeline: trans_ok=False for 12 seconds straight while the subject lay
            # in plain view.
            #
            # The three facts a fall is made of — was upright, dropped fast, stayed
            # down — do not have to be true in the same instant. Accumulate them.
            if not st.transition_ok:
                # "Was this person ever standing?" — asked over the WHOLE baseline
                # window, and answered by EITHER witness (torso or box).
                #
                # It used to be asked only of the 1.5 s transition window, and only
                # of whichever witness happened to be available. That is the worst
                # possible place to ask: the 1.5 s before touchdown is exactly when
                # the body is rotating and motion-blurred, so pose is least reliable
                # there. If no torso reading in that narrow window happened to land
                # under FALL_UPRIGHT_ANGLE, `transition_ok` was set False — and since
                # it is only computed once, at prone onset, it stayed False for the
                # REST OF THE TRACK'S LIFE.
                #
                # Measured on URFD: fall-09, fall-15 and fall-22 lay on the floor for
                # 3.4-4.1 s with the person detected on 94-100% of ticks, and never
                # alarmed — for this reason alone. A person who is now on the ground
                # was standing at SOME point in the last ten seconds; that is the
                # fact the rule actually needs, and it does not require the answer to
                # come from the one second where the camera can see least.
                # (computed above, every tick, so a miss can name the term that
                # rejected it — see FallResult.was_upright / went_down_fast.)
                #
                # DID THEY GO DOWN FAST? Two independent witnesses; either will do.
                #
                # Both are asked over the same window as `was_upright`, and for the
                # same reason: the reported hip_v only looks back FALL_TRANSITION_SEC,
                # so a few seconds after touchdown it reads 0.00, and a gate that is
                # re-checked every tick against a value which decays to zero could
                # only ever pass in the first second.
                #
                # The hip is the better witness — but it needs POSE, and pose needs
                # frames. Measured in the live app on this CPU box: the fall loop
                # manages ~1.5 ticks/second, so a 0.5 s fall lands between two samples
                # ~4 s apart and every velocity is smeared to roughly an eighth of its
                # true value. hip_v read 0.00 through an entire fall that the offline
                # evaluator (resampled to a clean 10 fps) measured at 0.73.
                #
                # `st.hist` is appended on EVERY tick, pose or no pose, so the box
                # centre's fall RATE is always available. It is coarser than the hip,
                # but it is a rate — so it still separates a fall from someone
                # lowering themselves to the floor, which a displacement cannot.
                st.transition_ok = bool(was_upright and went_down_fast)
        elif st.last_prone_t is None or (now - st.last_prone_t) > grace:
            # Only NOW has this person genuinely stopped looking prone. A single
            # dropped frame must not restart the "stayed down" timer: pose and the
            # person detector both blink on a body lying on the floor, so `prone`
            # arrives as P P . P P . P, and resetting on the first dot meant
            # prone_for restarted from zero forever and could never reach
            # FALL_MOTIONLESS_SEC. Measured on URFD: 9 of 30 falls cleared prone AND
            # a strong hip velocity, and were lost to precisely this.
            st.prone_since = None
            st.transition_ok = False

        # `still_down` keeps counting through the blinks, for the same reason.
        still_down = prone or (st.last_prone_t is not None
                               and (now - st.last_prone_t) <= grace)
        prone_for = (now - st.prone_since) if st.prone_since else 0.0
        drop = 0.0
        if len(st.hist) >= 2:
            drop = cy - min(e[3] for e in st.hist)

        # A fall the rule layer will vouch for: plausible body, seen upright, went
        # prone fast, and STAYED down.
        transition = bool(body_ok and still_down and st.transition_ok and
                          prone_for >= float(getattr(cfg, "FALL_MOTIONLESS_SEC", 2.0)))
        return (body_ok, prone, prone_for, drop, baseline, transition, torso_deg,
                hip_v, was_upright, went_down_fast, still_down, st.transition_ok)

    @staticmethod
    def _resample(buf, now: float, dt: float, fill, max_hold: float = 0.4):
        """Sample a (t, value) buffer onto INPUT_TIMESTEPS points spaced `dt`
        apart, ending at `now`. Zero-order hold: each grid point takes the most
        recent sample at or before it.

        Each grid point takes the NEAREST sample in time, not the last one at or
        before it. Real tick timestamps jitter by a millisecond or two, and with
        a hold a sample landing just after its grid point is pushed into the
        next slot — duplicating one step and dropping another for no reason.
        Nearest is stable under that jitter and still reproduces the plain
        30-frame deque exactly when samples arrive one per dt, so the model sees
        an identical input whenever the box is keeping up. Below that cadence
        samples repeat, which is honest: we did not observe more. What matters
        is that the window still SPANS the same 2 s either way.
        """
        samples = list(buf)          # deque indexing is O(n); walk it once
        n = len(samples)
        out, j = [], 0
        for k in range(INPUT_TIMESTEPS - 1, -1, -1):
            t_k = now - k * dt       # oldest grid point first, so t_k increases
            while (j + 1 < n
                   and abs(samples[j + 1][0] - t_k) <= abs(samples[j][0] - t_k)):
                j += 1
            # Reuse a sample only while it is still recent. The tolerance is a
            # TIME, not one grid step: at 4.6 Hz real samples sit 217 ms apart,
            # so a one-step tolerance would mark two thirds of the grid as gaps,
            # drive pose coverage under FALL_MIN_POSE_COVERAGE and switch the
            # classifier off entirely — precisely the multi-camera case this is
            # meant to rescue. Beyond max_hold there genuinely was no view of
            # this person, and a gap is the honest input.
            out.append(samples[j][1] if abs(samples[j][0] - t_k) <= max_hold
                       else fill)
        return out

    def _predict(self, st: _TrackState, now: float) -> float:
        # Enough real observations to describe the window. The old test was
        # "30 frames buffered", which at a slow cadence meant 30 frames spanning
        # six seconds; this asks the question that actually matters — do we have
        # a few genuine samples inside the last 2 s.
        if len(st.seq) < self._seq_min_samples:
            return 0.0
        span = st.seq[-1][0] - st.seq[0][0]
        if span < self._seq_window * 0.5:
            return 0.0          # track too young: the window is mostly padding
        grid = self._resample(st.seq, now, self._seq_dt,
                              np.zeros(NUM_FEATURES, np.float32),
                              self._seq_max_hold)
        x = np.expand_dims(np.array(grid, dtype=np.float32), 0)
        with self._lock:                                # one interpreter, many tracks
            self._it.set_tensor(self._in["index"], x)
            self._it.invoke()
            return float(self._it.get_tensor(self._out["index"])[0][0])

    def step(self, frame_bgr, persons: list[dict], now: Optional[float] = None) -> dict[int, FallResult]:
        """persons: dicts with track_id + x1,y1,x2,y2. Every live track is processed —
        no cap, no eviction. Returns per-track state; the engine decides on events."""
        now = time.time() if now is None else now
        persons = [p for p in persons if p.get("track_id") is not None]
        H, W = frame_bgr.shape[:2]
        feats = self.extractor.extract(frame_bgr, persons) if persons else {}

        alive, results = set(), {}
        for p in persons:
            tid = p["track_id"]
            alive.add(tid)
            st = self._state_for(tid, p, now)

            f = feats.get(tid, np.zeros(NUM_FEATURES, np.float32))
            # A skeleton without a hip+shoulder cannot be normalised, so it is NOT
            # an observation — it is a gap, exactly like "no pose" upstream.
            ok = _has_pose(f) and pose_is_normalizable(f)
            st.seq.append((now, normalize_skeleton_frame(f) if ok
                                else np.zeros(NUM_FEATURES, np.float32)))
            st.pose_hits.append((now, 1.0 if ok else 0.0))

            # Physics, read from the RAW keypoints — normalisation above deliberately
            # destroys absolute position and orientation, which is what a torso angle
            # and a hip velocity are made of. The model's input path is untouched.
            anat = anatomy(f, p, W, H, getattr(self.extractor, "space", "frame")) if ok else None

            # Coverage is measured over the SAME resampled window the classifier
            # sees, so "how much of the model's input is real pose" stays the
            # honest question it was when the two were the same 30 frames.
            coverage = (float(np.mean(self._resample(st.pose_hits, now,
                                                     self._seq_dt, 0.0,
                                                     self._seq_max_hold)))
                        if st.pose_hits else 0.0)
            # A mostly-zero window makes p_fall meaningless — judge those tracks by rules.
            if coverage >= self.min_coverage:
                st.p_fall = self._predict(st, now)
                st.p_hist.append((now, st.p_fall))      # a real observation
            else:
                st.p_fall = 0.0                          # not an observation: "we can't say"

            (body_ok, prone, prone_for, drop, baseline, transition, torso_deg,
             hip_v, was_upright, went_down_fast, still_down,
             transition_ok) = self._posture(st, p, W, H, now, anat)
            st.last_box = (p["x1"], p["y1"], p["x2"], p["y2"], now)

            # ── The classifier's verdict outlives the pose that produced it ──────
            # p_fall answers "was the motion over the last ~1-3 s a fall?". Once it
            # has said yes at 0.986, that is a statement about motion that ALREADY
            # HAPPENED. It does not become false because the person is now lying
            # still and the pose model — which cannot see people on the floor — has
            # stopped reporting them.
            #
            # Reading only the instantaneous p_fall erased that verdict the moment
            # coverage decayed, which for a fallen person is immediately and forever.
            # Measured on URFD: 7 of 30 falls were recognised by the classifier (up
            # to p=0.986) and then forgotten before the confirm window could fill.
            #
            # This is not a free pass. `corroborated` below still demands, on THIS
            # frame, that the person is down. A stale verdict alone never fires.
            hold = float(getattr(cfg, "FALL_VERDICT_HOLD_SEC", 3.0))
            while st.p_hist and now - st.p_hist[0][0] > hold:
                st.p_hist.popleft()
            p_recent = max((pv for _t, pv in st.p_hist), default=0.0)
            pose_hit = p_recent >= self.threshold
            # The model is the evidence; posture only corroborates. Every term here
            # is answerable from the CURRENT frame, so a person who is already on the
            # floor when their track begins is still detectable — that is the whole
            # point.
            # When the torso is visible it is a strictly better witness than the box:
            # `lying_now` (a wide box) is ALSO true of someone sitting on the floor,
            # so corroborating the model with it re-admits the very false positive
            # the torso angle exists to kill. `ar_grew` / `lying_now` survive as the
            # fallback for a body too small for pose — and for someone lying towards
            # the lens, whose box is foreshortened and never gets wide.
            if torso_deg is not None:
                corroborated = body_ok and torso_deg > float(
                    getattr(cfg, "FALL_PRONE_ANGLE", 60.0))
            else:
                lying_now = ar_now(st) > float(getattr(cfg, "FALL_AR_ABS_MIN", 0.9))
                corroborated = body_ok and (lying_now or ar_grew(st, baseline))
            pose_fall = pose_hit and corroborated

            # ── THE FIX ──────────────────────────────────────────────────────
            # This used to read:
            #     rule_fall = (coverage < self.min_coverage) and transition
            # The rule layer was only allowed to speak when pose had FAILED. So a
            # worker plainly on the floor — seen upright a second ago, torso
            # horizontal, hip dropped fast, motionless for two seconds — raised NO
            # alarm whenever the model happened to score 0.85 against its 0.90
            # threshold. The rules held the evidence and were gagged by a condition
            # that had nothing to do with whether they were right.
            #
            # A torso-backed `transition` is now strong enough to stand alone: it
            # demands was-upright + torso past FALL_PRONE_ANGLE + a real downward hip
            # RATE + stayed down. That is four independent facts, not a posture.
            #
            # Without a torso it is NOT strong enough and must not stand alone: to a
            # bounding box, sitting on the floor and lying on the floor are the same
            # picture, and nothing in that picture can separate them. So for a worker
            # too far away for pose, keep the old deliberately narrow contract — the
            # rules may only speak when the model was blind anyway. That is a real
            # limitation of monocular far-field vision, not an oversight.
            rule_fall = transition if torso_deg is not None else (
                (coverage < self.min_coverage) and transition)

            if self.mode == "pose":
                fallen = pose_fall
            elif self.mode == "yolo":
                fallen = transition
            else:                                        # hybrid (default)
                fallen = pose_fall or rule_fall

            results[tid] = FallResult(st.p_fall, prone, prone_for, coverage, drop,
                                      body_ok, baseline, transition, fallen,
                                      torso_deg=torso_deg, hip_v=hip_v,
                                      was_upright=was_upright,
                                      went_down_fast=went_down_fast,
                                      still_down=still_down,
                                      transition_ok=transition_ok)

        # A track that left is not forgotten immediately — it is parked for
        # FALL_REBIND_SEC so the same person returning under a new id keeps their
        # history. Anything older is dropped by `_state_for`.
        for tid in list(self._states):
            if tid not in alive:
                st = self._states.pop(tid)
                if st.last_box is not None:
                    self._lost[tid] = st
        ttl = float(getattr(cfg, "FALL_REBIND_SEC", 3.0))
        for tid, st in list(self._lost.items()):         # bound the parked set
            if now - st.last_box[4] > ttl:
                self._lost.pop(tid, None)
        return results
