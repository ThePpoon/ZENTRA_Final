#!/usr/bin/env python3
# utils/pose_visibility.py — decide which PPE categories can be JUDGED for a person,
# from their pose keypoints. Lifts the logic validated in scripts/proto_ppe_visibility.py.
# ================================================================
# required-absence flags a category as a violation when it is not positively WORN.
# That is right for a model that sees worn items better than their absence — but it
# cannot tell "not wearing a vest" from "torso hidden". So before charging a required
# category, ask the direct question: is the body part needed to judge it visible?
#   helmet ← head (nose/eyes/ears)   vest ← shoulders (+hip, tunable)
#   gloves ← wrists                  glasses ← face (nose/eyes)   boots ← ankles
# A category whose body part is NOT visible is left UN-JUDGED (no charge), instead of
# accusing a worker we simply cannot see.
# ================================================================
from __future__ import annotations

import config as cfg

# COCO-17 keypoint indices per PPE category. "any" = at least one of these visible.
_NEEDS_ANY = {
    "helmet":  (0, 1, 2, 3, 4),     # nose / eyes / ears → head visible
    "gloves":  (9, 10),             # wrists
    "glasses": (0, 1, 2),           # nose / eyes (face toward camera)
    "boots":   (15, 16),            # ankles
}
_L_SH, _R_SH, _L_HIP, _R_HIP = 5, 6, 11, 12


def visible_cats(kp_conf, kp_conf_floor: float | None = None,
                 vest_mode: str | None = None) -> set:
    """Which PPE categories can be judged, given one person's 17 keypoint confidences.

    vest_mode: 'strict' (both shoulders + ≥1 hip) / 'medium' (both shoulders) /
    'loose' (either shoulder). Defaults to cfg.PPE_VEST_VISIBILITY."""
    floor = cfg.PPE_KP_CONF if kp_conf_floor is None else kp_conf_floor
    vest_mode = (cfg.PPE_VEST_VISIBILITY if vest_mode is None else vest_mode).lower()
    vis = {i for i, c in enumerate(kp_conf) if c >= floor}
    out = set()
    for cat, idxs in _NEEDS_ANY.items():
        if vis & set(idxs):
            out.add(cat)
    both_sh = _L_SH in vis and _R_SH in vis
    either_sh = _L_SH in vis or _R_SH in vis
    any_hip = _L_HIP in vis or _R_HIP in vis
    if vest_mode == "strict":
        vest_ok = both_sh and any_hip
    elif vest_mode == "loose":
        vest_ok = either_sh
    else:                                    # medium (recommended)
        vest_ok = both_sh
    if vest_ok:
        out.add("vest")
    return out


# ── GRADED versions of the above ─────────────────────────────────────────────
# visible_cats() answers a yes/no question ("may this camera judge X?") by
# thresholding every keypoint confidence at PPE_KP_CONF. That is too coarse: a
# worker's shoulders stay visible from behind, so the boolean gate says "you may
# judge the vest" for a view that cannot actually see one. These functions keep
# the magnitude, which is what the abstention rule (PPE_ABSTAIN_W) thresholds on.
# They are additive; nothing that already calls visible_cats() changes behaviour.

def _soft_or(kp_conf, idxs) -> float:
    """P(at least one of these keypoints is really visible), treating the model's
    per-keypoint confidences as independent probabilities. Reduces to the old
    boolean "any of them above the floor" at the extremes, but stays smooth in
    between."""
    p = 1.0
    for i in idxs:
        if i < len(kp_conf):
            p *= (1.0 - float(kp_conf[i]))
    return 1.0 - p


def cat_visibility(kp_conf, cat: str, vest_mode: str | None = None) -> float:
    """How well can this camera judge `cat` for this person? 0 = not at all,
    1 = fully. Same body-part logic as visible_cats(), kept continuous.

    A 0 here means "no evidence", which is NOT the same as "compliant" — the
    distinction the boolean gate cannot express.
    """
    if kp_conf is None:
        return 0.0
    idxs = _NEEDS_ANY.get(cat)
    if idxs is not None:
        return _soft_or(kp_conf, idxs)
    if cat != "vest":
        return 0.0
    mode = (cfg.PPE_VEST_VISIBILITY if vest_mode is None else vest_mode).lower()
    sh_l = float(kp_conf[_L_SH]) if _L_SH < len(kp_conf) else 0.0
    sh_r = float(kp_conf[_R_SH]) if _R_SH < len(kp_conf) else 0.0
    hip = max(float(kp_conf[_L_HIP]) if _L_HIP < len(kp_conf) else 0.0,
              float(kp_conf[_R_HIP]) if _R_HIP < len(kp_conf) else 0.0)
    if mode == "loose":
        return max(sh_l, sh_r)
    if mode == "strict":
        return min(sh_l, sh_r) * hip        # both shoulders AND a hip
    return min(sh_l, sh_r)                  # medium: both shoulders


def facing_score(kp_conf) -> float:
    """0 = the person's back is to this camera, 1 = they face it.

    Nose + both eyes: those keypoints disappear when someone turns away, while the
    ears survive. MEASURED on real site footage (151 frames, verified by eye):
    0.011-0.04 on every frame where the subject's back was to the camera, and
    0.98-0.99 on every frame where they faced it, with profile views landing
    around 0.65. This is the signal that lets the system work out which camera to
    believe about front-worn PPE, without anyone configuring camera angles."""
    if kp_conf is None or len(kp_conf) < 3:
        return 0.0
    return float((kp_conf[0] + kp_conf[1] + kp_conf[2]) / 3.0)


# How much a category's evidence survives being seen from behind. A helmet or a
# boot looks the same from any angle; a vest is mostly (not entirely — hi-vis
# vests carry rear reflective panels) a frontal observation; glasses are visible
# from the front and essentially nowhere else. The floor is what a fully
# rear-facing view is still worth.
_ORIENT_FLOOR = {"vest": 0.25, "glasses": 0.10}


def orientation_weight(cat: str, facing: float) -> float:
    """Discount `cat`'s evidence for a person who is not facing this camera."""
    floor = _ORIENT_FLOOR.get(cat)
    if floor is None:
        return 1.0                          # helmet / gloves / boots: angle-agnostic
    return floor + (1.0 - floor) * max(0.0, min(1.0, facing))


def scale_weight(h_frac: float) -> float:
    """A person occupying 8% of the frame gives PPE evidence you cannot trust; the
    same person at 40% gives evidence you can. Ramps to full trust at
    PPE_CROP_MIN_PERSON_H_FRAC, the height the crop "zoom" pass already treats as
    'big enough to judge without help'."""
    full = float(getattr(cfg, "PPE_CROP_MIN_PERSON_H_FRAC", 0.45))
    if full <= 0:
        return 1.0
    return max(0.0, min(1.0, float(h_frac) / full))


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_pose_to_person(person: dict, poses: list, min_iou: float = 0.3):
    """Pick the pose whose box best overlaps this person box. `poses` is a list of
    (box_xyxy, kp_conf). Returns kp_conf of the best match, or None."""
    pbox = (person["x1"], person["y1"], person["x2"], person["y2"])
    best, best_iou = None, min_iou
    for box, kp_conf in poses:
        s = _iou(pbox, box)
        if s > best_iou:
            best, best_iou = kp_conf, s
    return best
