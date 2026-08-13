#!/usr/bin/env python3
# utils/ppe_association.py — assign PPE / no_* boxes to person tracks by
# CONTAINMENT, then derive each person's per-category compliance state.
# ================================================================
# Containment (not IoU): frac = area(item ∩ person) / area(item). A helmet box is
# tiny vs a person box, so IoU is ~0 and useless; containment answers "is this
# helmet ON this person?". Assign each item to the person with the highest frac
# above PPE_ASSOC_OVERLAP (0.30).
#
# Per-category state ∈ {WORN, VIOLATION, UNKNOWN}. WORN wins over VIOLATION for
# the same category on the same person (a positive detection of the item means
# they have it → avoids false alarms from a spurious no_* box). UNKNOWN (no box
# either way) does NOT alarm by default — see the plan's absence policy.
# ================================================================
from __future__ import annotations

import config as cfg

CATEGORIES = ["helmet", "vest", "gloves", "glasses", "boots"]

# COCO-17 keypoints that ANCHOR each category to a person — a helmet belongs to the
# head under it, gloves to the wrists, etc. Used to break a crowd tie that pure box
# containment gets wrong.
_CAT_ANCHOR_KP = {
    "helmet":  (0, 1, 2, 3, 4),   # nose / eyes / ears
    "glasses": (0, 1, 2),         # nose / eyes
    "vest":    (5, 6, 11, 12),    # shoulders / hips (torso)
    "gloves":  (9, 10),           # wrists
    "boots":   (15, 16),          # ankles
}


def _pick_person(it: dict, cat: str, persons: list[dict], cands: list) -> int:
    """Choose which candidate person owns this item. `cands` = [(idx, containment)]
    with containment ≥ threshold. One candidate → it. Several (overlapping boxes in a
    crowd) → the person whose category anchor keypoint sits INSIDE the item, else
    whose anchor keypoint is NEAREST the item centre; fall back to max containment
    when no pose is available."""
    if len(cands) == 1:
        return cands[0][0]
    anchors = _CAT_ANCHOR_KP.get(cat)
    kp_floor = float(getattr(cfg, "PPE_KP_ANCHOR_CONF", 0.30))
    icx, icy = it["cx"], it["cy"]
    best_j, best_score = None, None
    if anchors:
        for j, _frac in cands:
            p = persons[j]
            kp_xy, kp_cf = p.get("kp_xy"), p.get("kp_conf")
            if kp_xy is None or kp_cf is None:
                continue
            for idx in anchors:
                if idx < len(kp_cf) and kp_cf[idx] >= kp_floor:
                    kx, ky = kp_xy[idx]
                    inside = (it["x1"] <= kx <= it["x2"] and it["y1"] <= ky <= it["y2"])
                    dist = ((kx - icx) ** 2 + (ky - icy) ** 2) ** 0.5
                    score = (0 if inside else 1, dist)  # prefer inside, then nearest
                    if best_score is None or score < best_score:
                        best_score, best_j = score, j
    if best_j is not None:
        return best_j
    return max(cands, key=lambda c: c[1])[0]      # no pose → box containment wins

# taxonomy index → (category, state)
_TAXO_TO_CAT_STATE = {
    0: ("vest", "WORN"), 1: ("boots", "WORN"), 2: ("glasses", "WORN"),
    3: ("gloves", "WORN"), 4: ("helmet", "WORN"),
    5: ("boots", "VIOLATION"), 6: ("glasses", "VIOLATION"),
    7: ("gloves", "VIOLATION"), 8: ("helmet", "VIOLATION"), 9: ("vest", "VIOLATION"),
}


def _containment(item: dict, person: dict) -> float:
    """area(item ∩ person) / area(item)."""
    ix1 = max(item["x1"], person["x1"]); iy1 = max(item["y1"], person["y1"])
    ix2 = min(item["x2"], person["x2"]); iy2 = min(item["y2"], person["y2"])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a = max(1e-6, (item["x2"] - item["x1"]) * (item["y2"] - item["y1"]))
    return inter / a


def _vest_worn_ok(conf: float, person: dict, frame_h: int | None) -> bool:
    """Should this vest box count as WORN for this person? Size-aware: a NEAR worker
    (box taller than PPE_VEST_STRICT_H_FRAC of the frame) must clear a strict conf
    floor, because a plain shirt on a well-resolved worker reads as a vest at high
    conf and a false-WORN masks a real violation. A FAR/small worker keeps the base
    floor — their real vest is genuinely faint and dropping it falsely accuses them.
    frame_h None → size-unaware (accept; the base floor in `_keep` already applied)."""
    if frame_h is None:
        return True
    h_frac = (person["y2"] - person["y1"]) / max(1, frame_h)
    if h_frac >= float(getattr(cfg, "PPE_VEST_STRICT_H_FRAC", 0.40)):
        return conf >= float(getattr(cfg, "PPE_VEST_STRICT_CONF", 0.72))
    return True


def associate(dets: list[dict], assoc_overlap: float | None = None,
              min_item_conf: float | None = None,
              frame_h: int | None = None) -> list[dict]:
    """Return one record per person:
       {person: <det>, states: {cat: WORN|VIOLATION|UNKNOWN}, items: [dets]}.

    Detection now runs at a LOW tracking floor (PPE_TRACK_CONF) so ByteTrack ids
    stay stable, which means low-confidence PPE boxes also arrive here. Filter
    those out at INFERENCE_CONFIDENCE (the PPE slider) BEFORE associating, so
    stable tracking doesn't cost PPE precision. Persons are NOT filtered — the
    tracker's own new_track_thresh already gates which persons get an id.

    `frame_h` (frame height in px) enables the size-aware VEST false-WORN guard
    (`_vest_worn_ok`); omit it and vest behaves size-unaware (back-compat)."""
    thr = cfg.PPE_ASSOC_OVERLAP if assoc_overlap is None else assoc_overlap
    base_conf = cfg.INFERENCE_CONFIDENCE if min_item_conf is None else min_item_conf
    # Per-CATEGORY confidence floor (cfg.PPE_CLASS_CONF), falling back to the global
    # PPE slider. Small classes (glasses/gloves) score lower than a helmet, so a
    # slightly lower floor for them recovers recall without dropping the whole
    # slider. Keyed by ppe_association CATEGORY (helmet/vest/gloves/glasses/boots).
    class_conf = getattr(cfg, "PPE_CLASS_CONF", {}) or {}
    persons = [d for d in dets if d.get("is_person")]

    def _keep(d) -> bool:
        taxo = d.get("taxo")
        if taxo not in _TAXO_TO_CAT_STATE:
            return False
        cat = _TAXO_TO_CAT_STATE[taxo][0]
        return d.get("conf", 0.0) >= class_conf.get(cat, base_conf)

    items = [d for d in dets if not d.get("is_person") and _keep(d)]

    recs = [{"person": p, "states": {c: "UNKNOWN" for c in CATEGORIES}, "items": []}
            for p in persons]

    for it in items:
        cat, state = _TAXO_TO_CAT_STATE[it["taxo"]]
        # All persons that contain enough of the item, then pick the true owner by
        # keypoint anchor (crowd-safe) — not merely the highest box overlap.
        cands = [(j, f) for j, p in enumerate(persons)
                 if (f := _containment(it, p)) >= thr]
        if not cands:
            continue
        best_j = _pick_person(it, cat, persons, cands)
        rec = recs[best_j]
        cur = rec["states"][cat]
        if state == "WORN":
            # A near worker's low-conf vest is likely a shirt misread as a vest — do
            # NOT let it count as WORN and mask a violation; leave the category
            # UNKNOWN so required-absence decides. A far worker's faint vest is kept.
            if cat == "vest" and not _vest_worn_ok(it.get("conf", 0.0),
                                                   persons[best_j], frame_h):
                continue
            rec["items"].append(it)
            rec["states"][cat] = "WORN"           # WORN wins
        else:
            rec["items"].append(it)
            if state == "VIOLATION" and cur != "WORN":
                rec["states"][cat] = "VIOLATION"
    return recs


def violations_of(rec: dict) -> list[str]:
    """Categories currently in VIOLATION state for one person record."""
    return [c for c, s in rec["states"].items() if s == "VIOLATION"]


def cat_state_of(taxo):
    """(category, state) for a taxonomy index, or None if it is not a PPE item.

    Public accessor for _TAXO_TO_CAT_STATE so consumers that need to know WHICH
    detection supported a category's verdict (the evidence-weighting layer) can
    ask instead of duplicating the mapping."""
    return _TAXO_TO_CAT_STATE.get(taxo)
