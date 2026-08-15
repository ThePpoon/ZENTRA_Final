# config.py — ZENTRA System Configuration
# Zone Environment Network Thermal Risk Analysis
# Windows 11 + NVIDIA GPU | Python 3.11
# อ้างอิง Slide: CEDT Innovation Summit 2026 (อันดับ 1)
# ================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load this project's .env explicitly so it works no matter what the
# current working directory is (the desktop app imports config from a
# different cwd). override=False keeps real environment vars authoritative.
load_dotenv(Path(__file__).parent / ".env", override=False)

# ================================================================
# BASE PATHS
# ================================================================
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
LOGS_DIR      = BASE_DIR / "logs"
COLLECTED_DIR = DATA_DIR / "collected"

for _d in [
    DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    COLLECTED_DIR / "ppe_violations",
    COLLECTED_DIR / "zone_intrusions",
    COLLECTED_DIR / "fall_events",
    COLLECTED_DIR / "normal",
]:
    _d.mkdir(parents=True, exist_ok=True)

# ================================================================
# MODEL WEIGHTS (all inference is in-process ultralytics — see utils/detect_track)
# ================================================================
# There is no inference server. The Roboflow-on-Docker-:9001 architecture is gone
# (removed 2026-07-14 with the Docker files); every model below is a local .pt run
# through ultralytics on MPS/CUDA/CPU. Nothing listens on :9001.
PPE_LOCAL_MODEL  = str(MODELS_DIR / "ppe_finetuned.pt")
FALL_LOCAL_MODEL = str(MODELS_DIR / "fall_finetuned.pt")
# Roboflow model id for the fall project — training/eval only; runtime inference uses
# the vendored FALL_TFLITE_PATH transformer, never this.
FALL_MODEL_ID    = os.getenv("FALL_MODEL_ID", "fall-detection-ovjqo/5")

# ── Roboflow SDK — used ONLY by training (dataset download / label upload) ──
# SECURITY: no hardcoded default — the previously committed key is leaked in git
# history and MUST be rotated. Set ROBOFLOW_API_KEY in .env/env. Empty is fine:
# inference never touches Roboflow, so only the training commands need it.
ROBOFLOW_API_KEY      = os.getenv("ROBOFLOW_API_KEY",  "")
ROBOFLOW_WORKSPACE    = os.getenv("ROBOFLOW_WORKSPACE", "pholawats-workspace")
ROBOFLOW_PPE_PROJECT  = os.getenv("ROBOFLOW_PPE_PROJECT",  "zentra-ppe")
ROBOFLOW_FALL_PROJECT = os.getenv("ROBOFLOW_FALL_PROJECT", "zentra-fall")

# ================================================================
# CAMERA  (Windows 11: USE_DSHOW=true เร็วกว่า CAP_FFMPEG)
# ================================================================
CAMERA_SOURCE   = os.getenv("CAMERA_SOURCE", "webcam")
WEBCAM_INDEX    = int(os.getenv("WEBCAM_INDEX", "0"))
RTSP_URL        = os.getenv("RTSP_URL", "")   # set in .env or Settings; no baked-in credentials
VIDEO_FILE_PATH = os.getenv("VIDEO_FILE_PATH", "")
USE_DSHOW       = os.getenv("USE_DSHOW", "true").lower() == "true"

# ================================================================
# INFERENCE
# ================================================================
INFERENCE_CONFIDENCE = float(os.getenv("INFERENCE_CONFIDENCE", "0.45"))
INFERENCE_IOU        = float(os.getenv("INFERENCE_IOU",        "0.45"))
# Decouple TRACKING confidence from PPE confidence. The ultralytics Detector runs
# detect+track at THIS low floor so ByteTrack receives the low-score boxes it
# needs for second-association (→ stable ids through a confidence dip). PPE items
# are then filtered separately by INFERENCE_CONFIDENCE in ppe_association, so the
# low track floor does NOT flood the violation logic with junk PPE boxes.
# MUST be <= track_low_thresh in the tracker yaml or those boxes never arrive.
PPE_TRACK_CONF       = float(os.getenv("PPE_TRACK_CONF", "0.10"))

# ── PERSON DETECTOR (decoupled from the PPE model) ──────────────────────────
# "Who is a person" needs high recall + stable tracking; "what PPE item is this"
# needs precision. Those were conflated into one PPE fine-tune whose `person`
# class is only a byproduct (recall ~0.63) — the real reason crowds went
# undetected. A COCO-pretrained YOLO detects people far better, and it now owns
# the tracker so track IDs follow the STRONG detector. The PPE model no longer
# emits persons and no longer tracks (see utils/detect_track.py).
# yolo11m, not 11s: measured on 7 real site photos (2026-07-14, ~18 people).
# Both find every real person, but 11s hallucinated a person on a CONCRETE PILLAR
# at conf 0.37 — over the tracker's new_track_thresh, so it would earn a track id,
# get PPE-checked, and (under the required-absence policy) raise a phantom
# violation forever. 11m scored that same pillar 0.13 and correctly dropped it.
# 11m also holds more margin on the hard cases (half-occluded worker 0.79 vs 0.69).
# Cost: full-engine 17.2 → 13.1 fps on MPS, which is still far more than safety
# monitoring needs. Set PERSON_MODEL=<path> to override (yolo11s.pt is still vendored).
PERSON_MODEL  = os.getenv("PERSON_MODEL", str(MODELS_DIR / "yolo11m.pt"))
# Model-call floor for the person detector. Low on purpose: ByteTrack needs the
# low-score boxes for second-association. Gating happens in the tracker yaml
# (new_track_thresh). MUST be <= track_low_thresh.
PERSON_CONF   = float(os.getenv("PERSON_CONF", "0.10"))
# 640 = yolo11s's own train size. This was 960 on the theory that small/distant
# people need the resolution — MEASURED FALSE on real site footage (2026-07-14):
# on IMG_5674 (worker half-occluded behind a precast box) the SAME model scored
# 0.691 at 640 but collapsed to 0.248 at 960 — just under the tracker's
# new_track_thresh (0.25), so the worker got no track id and PPEEngine skipped him
# entirely. The system was blind to a man standing in the middle of the frame.
# Swept 12 sizes, deterministic: 960 was the WORST of all of them (896→0.690,
# 1024→0.589). Inference far from a model's train scale is a real risk; don't
# raise this without re-measuring on real footage.
PERSON_IMGSZ  = int(os.getenv("PERSON_IMGSZ", "640"))
# Inference device for the PPE model. Empty → detect_track auto-picks
# MPS → CUDA → CPU. Set explicitly (e.g. "cpu") to FORCE a device.
PPE_INFER_DEVICE     = os.getenv("PPE_INFER_DEVICE", "") or None
INFER_EVERY_N_FRAMES = int(os.getenv("INFER_EVERY_N_FRAMES",   "2"))   # PPE is ~11ms on GPU → infer more often (less flicker)
# Anti-flicker: keep the last PPE boxes this long to bridge single-frame misses
# (detector occasionally drops a frame → boxes blink). 0 = no hold.
PPE_HOLD_SEC         = float(os.getenv("PPE_HOLD_SEC", "0.5"))
# Display smoothing: EMA on box coordinates so boxes glide instead of jittering
# frame-to-frame ("ดิ้น"). Lower ALPHA = smoother but a touch laggier.
# Clean display: draw ONE box per person + a PPE status label instead of many
# overlapping class boxes (Protex-style, far less cluttered).
PPE_CLEAN_DISPLAY    = os.getenv("PPE_CLEAN_DISPLAY", "true").lower() == "true"
PPE_SMOOTH           = os.getenv("PPE_SMOOTH", "true").lower() == "true"
PPE_SMOOTH_ALPHA     = float(os.getenv("PPE_SMOOTH_ALPHA", "0.25"))
PPE_SMOOTH_IOU       = float(os.getenv("PPE_SMOOTH_IOU", "0.30"))
# Deadband (px): if a box moved less than this, freeze it (rock-steady when a
# person stands still; still follows real movement above the threshold).
PPE_SMOOTH_DEADBAND  = float(os.getenv("PPE_SMOOTH_DEADBAND", "3.0"))
# De-duplicate overlapping boxes of the SAME class (one object → one box).
# Predicting at a low floor returns extra overlapping candidates; suppress any
# same-class box overlapping an already-kept one above this IoU. 0.70 only
# removes near-identical duplicates (keeps two distinct-but-close people).
PPE_NMS_IOU          = float(os.getenv("PPE_NMS_IOU", "0.45"))
# Inference resolution for the LOCAL PPE model. 640 = the size best2 was trained
# at, and 1.6x faster than 960 on MPS (98 vs 156 ms/frame) with no loss on the
# classes that matter — the previous 960 default was a recall crutch for the weak
# old fine-tune. run_native.sh already pinned 640; this only aligns the default
# so the offline tools (autolabel, evaluate) match what deploy actually runs.
PPE_IMGSZ = int(os.getenv("PPE_IMGSZ", "640"))
# Per-CATEGORY confidence floor, applied in utils/ppe_association.associate().
# Keyed by ppe_association CATEGORY: helmet / vest / gloves / glasses / boots.
# A class not listed uses the global INFERENCE_CONFIDENCE slider.
#
# EMPTY on purpose. glasses/gloves used to sit at 0.25 — a recall crutch for the
# old fine-tune, whose glasses/gloves boxes scored a median 0.16-0.18 and could
# never clear the 0.45 slider. best2 scores those same classes at 0.59-0.74 on
# real val images, so the lowered floor no longer buys recall; it only lets junk
# boxes through, and a junk WORN box is the dangerous kind (WORN beats VIOLATION
# in associate(), so a false glove can mask a real bare hand).
# Re-add an entry ONLY if a val sweep shows a class is being starved by 0.45. (The
# safety-critical false-WORN case for VEST is handled separately, size-aware, below —
# a scalar floor there fails: measured on real footage a plain shirt reads as a WORN
# vest up to conf 0.84 while a REAL vest at distance drops to 0.42, so the two
# distributions overlap and NO single floor separates them.)
PPE_CLASS_CONF: dict = {}

# ── VEST false-WORN, size-aware ─────────────────────────────────────────────
# A plain coloured shirt on a NEAR worker gets read as a worn vest with high conf
# (median 0.63, up to 0.84) — a false-WORN that MASKS a real violation. But a REAL
# vest on a FAR worker scores only 0.42-0.69. So the confidence a vest box must clear
# to count as WORN is scaled by how big the person is: a near/well-resolved worker
# must clear PPE_VEST_STRICT_CONF (few pixels of doubt → demand confidence, kill the
# shirt-as-vest); a far/small worker keeps the base INFERENCE_CONFIDENCE floor (their
# real vest is genuinely faint — don't drop it and falsely accuse them). Applied in
# ppe_association.associate() using the person's box height / frame height.
PPE_VEST_STRICT_CONF   = float(os.getenv("PPE_VEST_STRICT_CONF", "0.72"))
PPE_VEST_STRICT_H_FRAC = float(os.getenv("PPE_VEST_STRICT_H_FRAC", "0.40"))  # taller than this = "near"

# ── PERSON-CROP PPE (two-stage "zoom") ──────────────────────────────────────
# A worker 120px tall in a 1080p frame gives a glove ~8px — under the PPE model's
# effective floor. Cropping that person and re-running the PPE model at CROP_IMGSZ
# gives the glove ~40px, recovering small-item recall the full-frame pass misses.
# HYBRID: the full-frame pass still runs (context + large/near workers); crops are
# an ADDITIONAL pass spent ONLY on the people who need it, so latency stays bounded.
# Measured on this MPS box: full-frame @640 = 28.6ms; 5 crops @320 BATCHED = 35.1ms
# (vs 65ms un-batched) — so batch the crops and keep CROP_IMGSZ small.
PPE_CROP_ENABLED     = os.getenv("PPE_CROP_ENABLED", "true").lower() == "true"
PPE_CROP_IMGSZ       = int(os.getenv("PPE_CROP_IMGSZ", "320"))   # 640 is ~4x slower per crop
# Who gets a crop: a person shorter than this fraction of the frame (distant/small,
# where every item is tiny) OR one with a required category still UNKNOWN after the
# full-frame pass (couldn't judge it — zoom in). A near worker filling the frame is
# already high-res and is skipped.
PPE_CROP_MIN_PERSON_H_FRAC = float(os.getenv("PPE_CROP_MIN_PERSON_H_FRAC", "0.45"))
# Cost is ∝ crop count. Cap per frame and ROUND-ROBIN the rest across later frames
# (the 3-of-5 temporal confirm absorbs the gap) so a sudden crowd can't spike the
# frame time — this is the "delay when the scene changes" guard.
PPE_CROP_MAX_PER_FRAME = int(os.getenv("PPE_CROP_MAX_PER_FRAME", "6"))
# Pad each crop so an item at the body edge (helmet crown, boot sole) isn't clipped.
PPE_CROP_PAD_FRAC    = float(os.getenv("PPE_CROP_PAD_FRAC", "0.08"))

# ── VISIBILITY GATE (pose) — don't accuse a body part you cannot see ─────────
# required-absence charges "not-WORN" as a violation. It cannot tell "not wearing a
# vest" from "torso hidden behind a machine", so an occluded compliant worker gets
# accused — the fastest way to make people ignore the alarms. Gate each category on
# whether the body part needed to judge it is visible in yolo11n-pose keypoints
# (helmet↔head, vest↔shoulders, gloves↔wrists, glasses↔face, boots↔ankles).
# Validated on 7 real photos / 19 people: charges 35→34, killed the ONE false
# accusation (IMG_5674, torso behind concrete), lost ZERO real violations.
PPE_VISIBILITY_GATE  = os.getenv("PPE_VISIBILITY_GATE", "true").lower() == "true"
PPE_KP_CONF          = float(os.getenv("PPE_KP_CONF", "0.5"))   # keypoint "visible" floor
# How strict "torso visible" is for VEST: strict = both shoulders + ≥1 hip (default —
# shoulders alone can show while the chest, where a vest sits, is hidden: IMG_5674 has
# both shoulders at 0.7 but the torso behind concrete, so only strict withholds the
# vest charge); medium = both shoulders; loose = either shoulder.
PPE_VEST_VISIBILITY  = os.getenv("PPE_VEST_VISIBILITY", "strict").lower()
# When pose finds NO match for a person at all (not merely an occluded part):
# strict=True → don't accuse (can't confirm visibility); False → keep charging by
# required-absence (a safety system shouldn't let a violation walk free just because
# the light pose model missed a distant worker — the crop pass still judges items).
PPE_VISIBILITY_STRICT = os.getenv("PPE_VISIBILITY_STRICT", "false").lower() == "true"
# Evidence weight for a person with NO pose match, used by the graded evidence
# path. Not knowing how well you can see someone is not the same as knowing you
# cannot: scoring it 0 would mute that camera, and with pose unavailable site-wide
# every observation would weigh 0 and nobody could ever be judged.
# Mirrors PPE_VISIBILITY_STRICT=false above (no pose → still judge), but ranks
# below a camera that actually has the body part in view.
PPE_NOPOSE_WEIGHT    = float(os.getenv("PPE_NOPOSE_WEIGHT", "0.40"))

# ── ABSTENTION: a camera that cannot see it does not get to accuse ──────────
# The boolean visibility gate answers yes/no per body part, which is too coarse:
# a worker's shoulders stay visible from behind, so the gate says "you may judge
# the vest" for a view that cannot actually see one. _observe_ppe() already
# scores HOW WELL this camera can judge each category (visibility x orientation x
# scale x detection confidence); this is the floor that score must clear before a
# violation may be raised. Below it the camera abstains — recording neither a
# violation nor a clean bill of health.
#
# 0.35 comes from measurement, not taste. On real site footage (377 frames,
# back-view vs front-view of the same worker):
#     vest     0.122 / 0.705      glasses  0.016 / 0.800     helmet 0.632 / 0.757
# so 0.35 blocks the rear views of vest and glasses, passes their front views,
# and passes helmet from BOTH — which is correct, a helmet looks the same from
# any angle (measured ratio 1.2x, versus 5.8x for vest and 49.6x for glasses).
PPE_ABSTAIN_ENABLED  = os.getenv("PPE_ABSTAIN_ENABLED", "true").lower() == "true"
PPE_ABSTAIN_W        = float(os.getenv("PPE_ABSTAIN_W", "0.35"))
# Per-category override, for sites that want e.g. a stricter vest rule. Anything
# absent falls back to PPE_ABSTAIN_W. Kept empty so the single slider governs.
PPE_ABSTAIN_W_BY_CAT: dict = {}
PPE_POSE_MODEL       = os.getenv("PPE_POSE_MODEL", str(MODELS_DIR / "yolo11n-pose.pt"))
PPE_POSE_IMGSZ       = int(os.getenv("PPE_POSE_IMGSZ", "640"))
# ── Keypoint-anchored item→person assignment (Ext 2) ────────────────────────
# Containment alone mis-assigns in crowds: a helmet between two overlapping workers
# lands on whoever's box contains 30% of it, which can be the wrong person. When >1
# person could own an item, pick the one whose category-relevant keypoint (helmet↔
# head, gloves↔wrist, boots↔ankle, vest↔torso) actually sits inside/nearest the item.
# Reuses the same pose pass as the visibility gate (no extra cost). Falls back to
# containment when pose is unavailable.
PPE_KP_ASSIGN        = os.getenv("PPE_KP_ASSIGN", "true").lower() == "true"
PPE_KP_ANCHOR_CONF   = float(os.getenv("PPE_KP_ANCHOR_CONF", "0.30"))  # anchor kp must clear this

# ── WORN hold — "I just saw them wearing it" (foreground-occlusion guard) ────
# A worker CONFIRMED wearing a vest/helmet, then walks behind a machine/culvert:
# pose keypoints keep firing (they're estimated through the object) so the
# visibility gate can't help, the partially-hidden vest scores below the WORN floor,
# and a spurious `no_vest` box flips the state → the compliant worker is falsely
# accused. Measured on 160539@21s: #4 was WORN for 3s, then occluded and charged.
# Fix: once a category is WORN-CONFIRMED (same 3-of-5 rigor as a violation, so a
# flickering shirt→vest false-WORN never latches), remember it per track and do NOT
# raise a violation for PPE_WORN_HOLD_SEC afterwards. Time-based → FPS-independent.
# Trade-off: a vest removed WHILE occluded is held stale until the window expires.
PPE_WORN_HOLD        = os.getenv("PPE_WORN_HOLD", "true").lower() == "true"
PPE_WORN_HOLD_SEC    = float(os.getenv("PPE_WORN_HOLD_SEC", "4.0"))
# ByteTrack churns the id every time a worker disappears behind a culvert and
# reappears (measured on 160539: one worker became #1→#2→#3→#4→#5→#9…), and each
# FRESH id briefly reads not-worn before WORN re-confirms → a per-id hold is lost.
# So the hold is ALSO remembered SPATIALLY: "a vest was confirmed WORN near here
# recently" suppresses a violation for whoever is at that spot, regardless of id.
# Radius as a fraction of the frame diagonal. A genuinely bare worker never confirms
# WORN, leaves no mark, and still fires (verified on the no-PPE clip).
PPE_WORN_HOLD_RADIUS = float(os.getenv("PPE_WORN_HOLD_RADIUS", "0.18"))

# ── PPE alert de-dup — stop the spam ────────────────────────────────────────
# The per-(track,category) cooldown lets a violation re-alert every few seconds
# because ByteTrack churns the id (each fresh id = fresh cooldown) and the key is gc'd
# whenever a worker briefly leaves the frame. So PPE alerts are ALSO de-duplicated by
# PLACE+CATEGORY+TIME (exactly like fall): the same missing-helmet at the same spot
# fires ONCE per window, no matter how the id churns. A persistent violation still
# re-alerts after the window so it doesn't fall off the radar.
PPE_DEDUPE_SEC       = float(os.getenv("PPE_DEDUPE_SEC", "45.0"))
PPE_DEDUPE_RADIUS    = float(os.getenv("PPE_DEDUPE_RADIUS", "0.15"))   # fraction of frame diagonal

# ================================================================
# PPE CLASSES
# Slide Module 1: Helmet / Vest / Goggles / Gloves / Safety Boots
# ================================================================
PPE_CLASSES: dict[str, dict] = {
    "helmet":          {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    "vest":            {"label": "Vest",      "label_th": "สวมเสื้อกั๊ก",      "color": (0, 210, 0),   "violation": False},
    # The fine-tuned local model (zentra-ppe v3) emits "Vest" capitalised — index
    # 0 in the deployed taxonomy. Without this key it fell through to the gray
    # default. _info() also normalises case, but keep it explicit for clarity.
    "Vest":            {"label": "Vest",      "label_th": "สวมเสื้อกั๊ก",      "color": (0, 210, 0),   "violation": False},
    "goggles":         {"label": "Goggles",   "label_th": "สวมแว่นตานิรภัย",   "color": (0, 210, 0),   "violation": False},
    "gloves":          {"label": "Gloves",    "label_th": "สวมถุงมือ",          "color": (0, 210, 0),   "violation": False},
    "safety_boots":    {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
    "glasses":         {"label": "Glasses",   "label_th": "สวมแว่นตา",         "color": (0, 210, 0),   "violation": False},
    "boots":           {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
    "no_helmet":       {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no_vest":         {"label": "No Vest",   "label_th": "ไม่สวมเสื้อกั๊ก",   "color": (0, 0, 220),   "violation": True},
    "no_goggles":      {"label": "No Goggles","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    # zentra-ppe v3 emits "no_glasses" (underscore, index 6). Previously only the
    # spaced "no glasses" and "no_goggles" existed, so eye-protection violations
    # from the local model were SILENTLY dropped (fell through to violation:False).
    "no_glasses":      {"label": "No Glasses","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    "no_gloves":       {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "no_safety_boots": {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    "no helmet":       {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no vest":         {"label": "No Vest",   "label_th": "ไม่สวมเสื้อกั๊ก",   "color": (0, 0, 220),   "violation": True},
    "no glasses":      {"label": "No Glasses","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    "no gloves":       {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "no boots":        {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    # ppe-tqmos/1 emits these extra class names — map them so violations/worn
    # states register correctly (singular "no glove" + redundant "helmet on").
    "no glove":        {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "helmet on":       {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    # ppe-vum8g/2 emits these class names — "hardhat" == helmet, "no_boots" not
    # covered by the safety_boots mapping above. Map them so worn/violation
    # states register with the same labels as the rest of the system.
    "hardhat":         {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    "no_hardhat":      {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no_boots":        {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    "person":          {"label": "Person",    "label_th": "บุคคล",              "color": (255, 190, 0), "violation": False},
    # ppe-vum8g/2 quirk: the runtime class strings differ from the Roboflow UI
    # labels. WORN items come back prefixed with "a " (a person / a hardhat /
    # a vest / a gloves / a boots) while VIOLATIONS use underscores (no_hardhat
    # …). Map the "a " variants so worn/person boxes draw correctly AND get
    # written into the collected YOLO training labels.
    "a person":        {"label": "Person",    "label_th": "บุคคล",              "color": (255, 190, 0), "violation": False},
    "a hardhat":       {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    "a vest":          {"label": "Vest",      "label_th": "สวมเสื้อกั๊ก",      "color": (0, 210, 0),   "violation": False},
    "a gloves":        {"label": "Gloves",    "label_th": "สวมถุงมือ",          "color": (0, 210, 0),   "violation": False},
    "a boots":         {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
}
# NOTE: PPE violation detection does NOT compute "person without helmet" — it
# relies on the model emitting explicit negative classes (e.g. "no helmet",
# "no vest") which modules/ppe.py flags via PPE_CLASSES[...]["violation"].
# Both current models (cloud ppe-cpxsz/2, local ppe_finetuned.pt) provide these
# plus "person" for Zone tracking. The pipeline logs a warning at runtime if a
# loaded model is missing those classes (see Pipeline._validate_ppe_classes).

# ================================================================
# PPE TAXONOMY — the SACRED 11-class order of OUR DATASET / label space.
# This is the single source of truth for training/collection class INDICES.
# It MUST match backend/data/train_dataset/roboflow_dl/data.yaml and
# training/merge_external.py:DEPLOYED. Do NOT reorder — every collected label
# .txt on disk is written in THIS index space (a past bug wrote sorted(display-
# label) indices, a different class space entirely).
#
# NOTE: a deployed .pt does NOT have to share this internal order. The current
# ppe_finetuned.pt (best2, trained on Colab from yolo11m) orders its head
# boots/glasses/gloves/helmet/... — different from the list below. That is fine:
# nothing reads raw model indices. Detections carry the model's CLASS STRING and
# ppe_taxo_index() maps it here by name. Swapping weights only breaks if the new
# model emits a class name that has no taxonomy match (→ taxo=None, silently
# dropped by ppe_association) — check with ppe_taxo_index() before deploying one.
# ================================================================
PPE_TAXONOMY: list[str] = [
    "Vest", "boots", "glasses", "gloves", "helmet",
    "no_boots", "no_glasses", "no_gloves", "no_helmet", "no_vest", "person",
]

# Map a RAW model class string (whatever the active model emits) → PPE_TAXONOMY
# index, or None to drop. Handles the local fine-tuned model (emits the taxonomy
# names verbatim) AND cloud-model variants (hardhat, a vest, goggles, …).
_PPE_TAXO_ALIAS: dict[str, str] = {
    "hardhat": "helmet", "a_hardhat": "helmet", "a_helmet": "helmet", "helmet_on": "helmet",
    "no_hardhat": "no_helmet",
    "a_vest": "Vest", "safety_vest": "Vest", "no_safety_vest": "no_vest",
    "goggles": "glasses", "a_goggles": "glasses", "safety_glasses": "glasses",
    "no_goggles": "no_glasses", "no_goggle": "no_glasses",
    "safety_boots": "boots", "a_boots": "boots", "a_shoes": "boots",
    "no_safety_boots": "no_boots",
    "a_gloves": "gloves", "a_glove": "gloves", "no_glove": "no_gloves",
    "a_person": "person", "worker": "person",
}

def ppe_taxo_index(raw: str) -> "int | None":
    """Raw model class string → PPE_TAXONOMY index (or None if unmapped)."""
    norm = " ".join(str(raw).lower().split()).replace("-", "_").replace(" ", "_")
    # direct match against the taxonomy itself (local model emits these)
    for i, name in enumerate(PPE_TAXONOMY):
        if norm == name.lower().replace(" ", "_"):
            return i
    tgt = _PPE_TAXO_ALIAS.get(norm)
    return PPE_TAXONOMY.index(tgt) if tgt else None

# ================================================================
# MEDIAPIPE — Slide Module 3: 33 Keypoints, 3 Detection Methods
# ================================================================
MEDIAPIPE_MODEL_COMPLEXITY    = 1
FALL_KEYPOINT_VELOCITY_THRESH = 0.30
FALL_BBOX_RATIO_THRESH        = 0.72
FALL_CONFIRM_FRAMES           = 6
GAIT_HISTORY_FRAMES           = 30
GAIT_ANOMALY_THRESH           = 0.20

# ── Fall fusion (Hybrid = YOLO primary + MediaPipe cross-check) ──
# mode: hybrid | yolo | pose
FALL_MODE                 = os.getenv("FALL_MODE", "hybrid")
FALL_YOLO_CONFIDENCE      = float(os.getenv("FALL_YOLO_CONFIDENCE", "0.50"))
FALL_YOLO_CONFIRM_FRAMES  = int(os.getenv("FALL_YOLO_CONFIRM_FRAMES", "4"))   # need N fall-frames per track
FALL_CONFIRM_WINDOW       = int(os.getenv("FALL_CONFIRM_WINDOW", "6"))        # ...within the last M frames
FALL_ASSOC_OVERLAP        = float(os.getenv("FALL_ASSOC_OVERLAP", "0.30"))    # min overlap of a fall box inside a person box
# MediaPipe Pose is heavy. Run it only every Nth frame (and never in 'yolo'
# mode) so it doesn't cap the live pipeline FPS / freeze the Live view.
FALL_POSE_EVERY_N         = int(os.getenv("FALL_POSE_EVERY_N", "3"))

# ── FALL: pose-sequence Transformer (punpayut/Fall-Detection, MIT) ──
# The classifier wants 30 CONSECUTIVE frames (~1 s at 30 fps). The PPE detect loop
# skips frames at ~13 fps, which would stretch a fall to ~2.3 s and wreck the
# temporal signal, so fall runs in its own fixed-cadence loop at FALL_LOOP_FPS.
FALL_TFLITE_PATH      = os.getenv(
    "FALL_TFLITE_PATH", str(BASE_DIR / "assets" / "models" / "fall_detection_transformer.tflite"))
# Pose backend: "yolo" (one pass, every person, far better on distant workers —
# measured 75% vs 25% landmark hit-rate below 150px) or "mediapipe" (faithful to
# upstream; single-person, so it runs per-crop on a thread pool).
FALL_POSE_BACKEND     = os.getenv("FALL_POSE_BACKEND", "yolo")
FALL_POSE_MODEL       = os.getenv("FALL_POSE_MODEL", str(MODELS_DIR / "yolo11n-pose.pt"))
FALL_POSE_WORKERS     = int(os.getenv("FALL_POSE_WORKERS", "4"))   # mediapipe backend only
# Pose inference resolution. 960 costs ~396 ms/frame on a CPU box; 640 costs ~263 ms
# and the fall loop's budget at FALL_LOOP_FPS is tight — tunable, not hard-coded.
FALL_POSE_IMGSZ       = int(os.getenv("FALL_POSE_IMGSZ", "640"))
FALL_LOOP_FPS         = int(os.getenv("FALL_LOOP_FPS", "15"))
# ── RTSP connect budget ─────────────────────────────────────────────────────
# How long cv2.VideoCapture may spend opening (and reading from) an IP camera
# before giving up. FFMPEG's default is ~30 s with a retry, so ONE unreachable
# camera held a /api/pipeline/start request for 48 s; three of them saturated
# the browser's per-origin connection pool and the whole UI stopped responding.
# A camera on the same LAN answers in well under a second.
RTSP_OPEN_TIMEOUT_MS  = int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "6000"))
# The classifier's 30-step input is resampled onto a fixed 1/FALL_LOOP_FPS grid,
# so its window stays 2 s wide however fast frames actually arrive (they arrive
# slower with every camera added). This is the minimum number of REAL samples
# that window must contain before the probability is trusted; below it the track
# is judged by the rule layer alone. Six covers ~1.3 s at the 4.6 Hz measured
# with two cameras, and rises to a full 30 when the box is keeping up.
FALL_SEQ_MIN_SAMPLES  = int(os.getenv("FALL_SEQ_MIN_SAMPLES", "6"))
# How long one observation may stand in for a grid point it did not land on.
# Must exceed the real inter-sample gap at the slowest cadence worth supporting
# (217 ms at the 4.6 Hz measured with two cameras) or the window fills with
# gaps; well under it there genuinely was no view of the person, and a gap is
# the honest input rather than a stale pose repeated.
FALL_SEQ_MAX_HOLD_SEC = float(os.getenv("FALL_SEQ_MAX_HOLD_SEC", "0.4"))
FALL_PROB_THRESHOLD   = float(os.getenv("FALL_PROB_THRESHOLD", "0.90"))  # upstream's value
# A window that is mostly no-pose zeros makes the probability meaningless; below
# this coverage the track is judged by the rule layer alone.
FALL_MIN_POSE_COVERAGE = float(os.getenv("FALL_MIN_POSE_COVERAGE", "0.6"))
# ── Torso angle + hip velocity: the two pose signals that box shape cannot give ──
# Signal 1 (torso angle) is the ONLY thing that separates "sitting on the floor" from
# "lying on the floor": both give a wide, low box; only the person lying has a
# horizontal torso. Signal 2 (hip velocity) is what makes a fall a fall — sitting,
# kneeling and crouching move the hip the same DISTANCE, only a RATE tells them apart.
FALL_UPRIGHT_ANGLE   = float(os.getenv("FALL_UPRIGHT_ANGLE",   "35.0"))  # deg from vertical: at/below = upright
FALL_PRONE_ANGLE     = float(os.getenv("FALL_PRONE_ANGLE",     "60.0"))  # deg from vertical: above = lying down
FALL_HIP_VELOCITY    = float(os.getenv("FALL_HIP_VELOCITY",    "0.35"))  # downward, in FRAME HEIGHTS per second
FALL_VELOCITY_WINDOW = float(os.getenv("FALL_VELOCITY_WINDOW", "0.5"))   # (informational; runtime uses TRANSITION/BASELINE windows)
# Box-centre fall RATE — the witness that never blinks (st.hist is appended every
# tick, pose or not), so a fall is still caught when pose can't read the person.
FALL_BOX_DROP_RATE   = float(os.getenv("FALL_BOX_DROP_RATE", "0.15"))
# The classifier's verdict outlives the pose that produced it: once p_fall says a
# fall happened, that stays true for this long even after the person lies still and
# the pose model (blind to people on the floor) stops reporting them.
FALL_VERDICT_HOLD_SEC = float(os.getenv("FALL_VERDICT_HOLD_SEC", "3.0"))
# ── Rule layer: a fall is a TRANSITION (upright → wide + dropped → stays down) ──
# The first version tested a posture in isolation (w/h > 0.72 AND still), which on a
# close-up or overhead camera is true forever: a seated person alarmed every cooldown.
# Everything below is measured against each PERSON'S OWN upright baseline instead.
FALL_MOTIONLESS_SEC   = float(os.getenv("FALL_MOTIONLESS_SEC", "2.0"))   # must STAY down this long
# A single dropped frame must not restart the "stayed down" timer: pose and the
# person detector both blink on a body on the floor (prone arrives as P P . P P . P).
# Keep counting as long as they looked prone within this grace window.
FALL_PRONE_GRACE_SEC  = float(os.getenv("FALL_PRONE_GRACE_SEC", "0.6"))
FALL_UPRIGHT_AR       = float(os.getenv("FALL_UPRIGHT_AR", "0.8"))       # w/h at or below this = upright
FALL_BASELINE_SEC     = float(os.getenv("FALL_BASELINE_SEC", "10.0"))    # window for the upright baseline
FALL_AR_SPIKE         = float(os.getenv("FALL_AR_SPIKE", "1.8"))         # prone = w/h > spike x baseline
FALL_AR_ABS_MIN       = float(os.getenv("FALL_AR_ABS_MIN", "0.9"))       # ...with an absolute floor
FALL_AR_CORROBORATE   = float(os.getenv("FALL_AR_CORROBORATE", "1.25"))  # posture backing for the model
FALL_TRANSITION_SEC   = float(os.getenv("FALL_TRANSITION_SEC", "1.5"))   # upright→prone must happen within
FALL_DROP_MIN         = float(os.getenv("FALL_DROP_MIN", "0.02"))        # box centre must fall this much
# A fall needs a plausible body. Reject a head-and-shoulders close-up that fills the
# frame, and specks in the distance — their aspect ratio says nothing about posture.
# Body plausibility, measured on the box DIAGONAL so it does not change when the
# person goes from standing to lying. Real falls span 0.04-0.51 of the frame
# diagonal; a face filling the lens spans 0.72 and is rejected.
FALL_MIN_BODY_SPAN    = float(os.getenv("FALL_MIN_BODY_SPAN", "0.05"))
FALL_MAX_BODY_SPAN    = float(os.getenv("FALL_MAX_BODY_SPAN", "0.60"))
# A fall makes ByteTrack drop the id (tall box -> wide box, IoU collapses). Carry
# the fall history over to the replacement id instead of starting from zero.
FALL_REBIND_SEC       = float(os.getenv("FALL_REBIND_SEC", "3.0"))
FALL_REBIND_IOU       = float(os.getenv("FALL_REBIND_IOU", "0.2"))
# Alert de-duplication is SPATIAL, not per-track: ByteTrack ids churn badly on some
# cameras (observed #515 → #933 in minutes), and a per-id cooldown lets every new id
# re-alarm. Two falls near the same place within this window are one incident.
FALL_DEDUPE_SEC       = float(os.getenv("FALL_DEDUPE_SEC", "20.0"))
FALL_DEDUPE_RADIUS    = float(os.getenv("FALL_DEDUPE_RADIUS", "0.15"))   # fraction of frame diagonal

# ================================================================
# BYTETRACK — Slide Module 2: Multi-Object Tracking
# ================================================================
BYTETRACK_TRACK_THRESH = 0.50
BYTETRACK_TRACK_BUFFER = 30
# IoU gate for matching a track to a detection. Our ByteTracker has NO motion
# model (no Kalman), so at 10 fps a walking person can drop below a high IoU
# between frames and get a NEW id (id churn breaks per-track logic). 0.50
# tolerates normal frame-to-frame motion while still rejecting unrelated boxes.
BYTETRACK_MATCH_THRESH = float(os.getenv("BYTETRACK_MATCH_THRESH", "0.50"))
# Low-score detection floor for the second matching pass (ByteTrack's "rescue" of
# tracks whose detection confidence dipped). Named to match the three above so the
# standalone tracker reads a complete, consistently-named set.
BYTETRACK_LOW_THRESH   = float(os.getenv("BYTETRACK_LOW_THRESH", "0.10"))

# Tracker config actually used by the in-process ultralytics Detector
# (utils/detect_track.py). Points at our tuned bytetrack yaml; falls back to
# ultralytics' packaged "bytetrack.yaml" if the file is missing.
#
# WHICH TRACKER RUNS WHEN — the BYTETRACK_* scalars above are NOT dead:
#   single-camera  → ultralytics native ByteTrack, driven by this yaml
#   shared multi-camera (the deployed multi-camera path) → the standalone
#     utils/tracker.py ByteTracker, driven by the BYTETRACK_* scalars above
#     (see utils/ppe_engine.py, the `hub is not None` branch)
# So a change to the scalars affects real multi-camera behaviour. It only ever
# looked otherwise because the engine read misspelled key names and silently got
# its own fallback literals instead of these values.
_ZENTRA_TRACKER = BASE_DIR / "assets" / "trackers" / "bytetrack_zentra.yaml"
PPE_TRACKER_CONFIG = os.getenv(
    "PPE_TRACKER_CONFIG",
    str(_ZENTRA_TRACKER) if _ZENTRA_TRACKER.exists() else "bytetrack.yaml",
)

# ================================================================
# SAFETY ZONE — Slide Module 2
# ================================================================
ZONE_POLYGON_FILE = str(DATA_DIR / "zones.json")
MAX_ZONES         = 10
ZONE_USE_FOOT_POINT = os.getenv("ZONE_USE_FOOT_POINT", "true").lower() == "true"  # test feet, not bbox centre
ZONE_TRACK_MIN_HITS = int(os.getenv("ZONE_TRACK_MIN_HITS", "3"))   # ignore unstable tracks
ZONE_CONFIRM_FRAMES = int(os.getenv("ZONE_CONFIRM_FRAMES", "3"))   # (legacy frame-count — no longer drives zone)
ZONE_CONFIRM_WINDOW = int(os.getenv("ZONE_CONFIRM_WINDOW", "5"))   # (legacy frame-count — kept for back-compat)
# FPS-INDEPENDENT zone confirm (time-based, replaces the frame-count above for zone):
# confirmed when the foot point was inside for >= MIN_RATIO of the last WINDOW_SEC
# seconds, with an absolute floor of MIN_HITS inside observations. A fixed N-of-M
# frame window's time span swings with FPS (5 frames = 0.17s @30fps vs ~1.3s @4fps),
# so a fast walk-through at low FPS could miss 3-of-5. Seconds behave the same at any
# FPS. Lower WINDOW_SEC / MIN_RATIO = more sensitive (catches quick crossings, more FP).
ZONE_CONFIRM_WINDOW_SEC = float(os.getenv("ZONE_CONFIRM_WINDOW_SEC", "0.8"))
ZONE_CONFIRM_MIN_RATIO  = float(os.getenv("ZONE_CONFIRM_MIN_RATIO",  "0.5"))
ZONE_CONFIRM_MIN_HITS   = int(os.getenv("ZONE_CONFIRM_MIN_HITS",   "2"))

# ================================================================
# PPE — accuracy / debounce
# ================================================================
PPE_CONFIRM_FRAMES = int(os.getenv("PPE_CONFIRM_FRAMES", "3"))     # need N violation frames per track
PPE_CONFIRM_WINDOW = int(os.getenv("PPE_CONFIRM_WINDOW", "5"))     # ...within the last M frames (per track)
PPE_ASSOC_OVERLAP  = float(os.getenv("PPE_ASSOC_OVERLAP", "0.30")) # min fraction of a PPE box inside a person box to associate

# REQUIRED-PPE ABSENCE POLICY (fixes "no helmet → shows ปลอดภัย").
# For these categories, a person is in VIOLATION when the item is NOT positively
# detected as worn — i.e. absence counts, we do NOT wait for the model to emit an
# explicit "no_helmet" box (those negative classes are unreliable). Detecting a
# WORN helmet is far more reliable than detecting its absence, so we invert:
#   required cat  → violation = (state != WORN)     [missing OR explicit no_*]
#   other cat     → violation = (state == VIOLATION) [only an explicit no_* box]
# The 3-of-5 temporal confirm still debounces flicker, so a worn helmet that
# blinks out for a frame won't false-alarm. Narrow this list (e.g. just "helmet")
# if the model can't reliably see a class. Values are ppe_association CATEGORIES:
#   helmet, vest, gloves, glasses, boots
PPE_REQUIRED = [c.strip().lower() for c in
                os.getenv("PPE_REQUIRED", "helmet,vest").split(",") if c.strip()]

# ================================================================
# ALERT COOLDOWN — Slide: 3 ระดับ
# ================================================================
VIOLATION_COOLDOWN_SECONDS = 30
ZONE_COOLDOWN_SECONDS      = 20
# Env-tunable (the other two are not, by design). A second genuine fall within this
# window of the first — same person gets up and falls again — is suppressed as one
# ongoing incident; lower it if every re-fall must raise its own alert. Works with
# FALL_DEDUPE_SEC (spatial de-dup), the tighter of the two wins.
FALL_COOLDOWN_SECONDS      = int(os.getenv("FALL_COOLDOWN_SECONDS", "15"))

ALERT_LEVEL_WARNING   = "warning"
ALERT_LEVEL_ALERT     = "alert"
ALERT_LEVEL_EMERGENCY = "emergency"

# ================================================================
# LINE OA — Slide: ส่งถึง หัวหน้างาน / Safety / Emergency
# ================================================================
LINE_OA_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_OA_CHANNEL_ACCESS_TOKEN", "")
LINE_OA_GROUP_SUPERVISOR     = os.getenv("LINE_OA_GROUP_SUPERVISOR", "")
LINE_OA_GROUP_SAFETY         = os.getenv("LINE_OA_GROUP_SAFETY",     "")
LINE_OA_GROUP_EMERGENCY      = os.getenv("LINE_OA_GROUP_EMERGENCY",  "")

_FB = os.getenv("LINE_OA_GROUP_ID", "")
if not LINE_OA_GROUP_SUPERVISOR: LINE_OA_GROUP_SUPERVISOR = _FB
if not LINE_OA_GROUP_SAFETY:     LINE_OA_GROUP_SAFETY     = _FB
if not LINE_OA_GROUP_EMERGENCY:  LINE_OA_GROUP_EMERGENCY  = _FB

ALERT_RECIPIENTS: dict[str, list[str]] = {
    ALERT_LEVEL_WARNING:   [LINE_OA_GROUP_SUPERVISOR],
    ALERT_LEVEL_ALERT:     [LINE_OA_GROUP_SAFETY, LINE_OA_GROUP_SUPERVISOR],
    ALERT_LEVEL_EMERGENCY: [LINE_OA_GROUP_EMERGENCY, LINE_OA_GROUP_SAFETY, LINE_OA_GROUP_SUPERVISOR],
}

# ── "All groups" routing (Settings → LINE group list) ───────────────────────
# The Settings page now manages a flat list of LINE groups; EVERY alert (any
# level) goes to EVERY enabled group — there is no per-level routing. These two
# are seeded from the legacy per-level ids here and REBUILT at runtime by
# pipeline.apply_settings() from the saved group list. LINE_GROUP_COOLDOWN is a
# per-group throttle keyed by group id (seconds); line_notify._dispatch reads it.
LINE_ALL_GROUPS: list[str] = [g for g in dict.fromkeys(
    [LINE_OA_GROUP_SUPERVISOR, LINE_OA_GROUP_SAFETY, LINE_OA_GROUP_EMERGENCY]) if g]

LINE_GROUP_COOLDOWN: dict[str, int] = {}

DAILY_REPORT_TIME = os.getenv("DAILY_REPORT_TIME", "20:00")
# External public image host used ONLY when LINE_UPLOAD_IMAGES is opted-in.
# LINE push image messages require a public HTTPS URL we must host ourselves —
# LINE provides no upload-and-get-URL endpoint — so attaching a photo to a LINE
# alert unavoidably sends it through an external host.
IMAGE_UPLOAD_URL  = "https://catbox.moe/user/api.php"
# PDPA: default OFF so no person image ever leaves the device. Evidence photos
# are still kept locally (History). Turn on only with explicit consent — then
# LINE alerts attach the photo via IMAGE_UPLOAD_URL above.
LINE_UPLOAD_IMAGES = os.getenv("LINE_UPLOAD_IMAGES", "false").lower() == "true"

# ================================================================
# DATA COLLECTION
# ================================================================
AUTO_COLLECT_FRAMES      = os.getenv("AUTO_COLLECT_FRAMES", "true").lower() == "true"
COLLECT_VIOLATION_FRAMES = True
COLLECT_NORMAL_INTERVAL  = 300
COLLECT_MAX_PER_CLASS    = 2000
COLLECT_JPEG_QUALITY     = 90
# Dataset quality: keep DIVERSE frames, not 30 near-identical ones per moment
COLLECT_MIN_INTERVAL_SEC = float(os.getenv("COLLECT_MIN_INTERVAL_SEC", "2.0"))  # min gap between saves / category
COLLECT_DEDUP_DIFF       = float(os.getenv("COLLECT_DEDUP_DIFF", "8.0"))        # skip if 32x32 gray diff < this
# Background sampling of the SITE itself (no event): the frames that teach the
# model this camera's lighting, angle and clutter. Seconds, not frames — the
# detect loop's fps swings with load, so a frame count is not a wall-clock rate.
COLLECT_NORMAL_INTERVAL_SEC = float(os.getenv("COLLECT_NORMAL_INTERVAL_SEC", "10.0"))
# Pseudo-label floors. The engine TRACKS at PPE_TRACK_CONF (~0.10) to hold ids
# through confidence dips; labels written at that floor are mostly junk to delete
# in Roboflow. Only boxes this confident become pseudo-labels.
COLLECT_LABEL_CONF       = float(os.getenv("COLLECT_LABEL_CONF", "0.40"))
# Persons get their own, lower floor. A missing person box is the WORST label
# error we can write (it teaches the next model that a visible worker is
# background), and person boxes come from the strong COCO detector + ByteTrack —
# a 0.30 person is still a person. A spurious PPE box, by contrast, is one click
# to delete in Roboflow.
COLLECT_PERSON_CONF      = float(os.getenv("COLLECT_PERSON_CONF", "0.25"))

# ================================================================
# TRAINING
# ================================================================
TRAIN_EPOCHS        = int(os.getenv("TRAIN_EPOCHS",     "80"))
TRAIN_PATIENCE      = int(os.getenv("TRAIN_PATIENCE",   "15"))   # early stop if no gain for N epochs
TRAIN_BATCH_SIZE    = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
TRAIN_IMG_SIZE      = int(os.getenv("TRAIN_IMG_SIZE", "768"))  # 768 helps small PPE (glasses/gloves) during training; inference runs at PPE_IMGSZ=640 (best2's train size)
TRAIN_DEVICE        = os.getenv("TRAIN_DEVICE", "0")
TRAIN_WORKERS       = int(os.getenv("TRAIN_WORKERS", "8"))
TRAIN_LR0           = 0.001
TRAIN_LRF           = 0.01
TRAIN_MOMENTUM      = 0.937
TRAIN_WEIGHT_DECAY  = 0.0005
TRAIN_WARMUP_EPOCHS = 5
TRAIN_VAL_SPLIT     = 0.15
TRAIN_AUG           = os.getenv("TRAIN_AUG", "true").lower() == "true"  # used by training/trainer.py
YOLO_BASE_MODEL     = os.getenv("YOLO_BASE_MODEL", "yolov8m.pt")

# ================================================================
# DISPLAY — Windows 11
# ================================================================
WINDOW_TITLE   = "ZENTRA Smart Detection"
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
FONT_SCALE     = 0.5
FONT_THICKNESS = 1
OSD_COLOR      = (255, 255, 255)
OSD_BG_COLOR   = (20, 20, 20)

# ================================================================
# PERFORMANCE
# ================================================================
TARGET_FPS        = 60
FRAME_BUFFER_SIZE = 4
ENABLE_THREADING  = True

