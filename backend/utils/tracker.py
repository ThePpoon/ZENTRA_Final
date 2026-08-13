# utils/tracker.py — Lightweight ByteTrack-inspired Multi-Object Tracker
# ติดตาม ID บุคคลต่อเนื่องข้ามเฟรม, รองรับ Occlusion
# ไม่ต้องพึ่ง external ByteTrack library (pure numpy + scipy)

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Track State
# ──────────────────────────────────────────────────────────────
class TrackState:
    NEW        = 0
    TRACKED    = 1
    LOST       = 2
    REMOVED    = 3


@dataclass
class Track:
    track_id:   int
    bbox:       np.ndarray      # [x1, y1, x2, y2]
    score:      float
    cls:        str
    state:      int = TrackState.NEW
    age:        int = 0         # frames since created
    hits:       int = 1         # times matched
    time_lost:  int = 0         # frames since last match
    history:    list = field(default_factory=list)   # center history (COG trajectory)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    def update(self, bbox: np.ndarray, score: float):
        self.bbox      = bbox
        self.score     = score
        self.hits     += 1
        self.age      += 1
        self.time_lost = 0
        self.state     = TrackState.TRACKED
        cx, cy = self.center
        self.history.append((cx, cy))
        if len(self.history) > 60:
            self.history.pop(0)

    def mark_lost(self):
        self.time_lost += 1
        self.age       += 1
        if self.time_lost > 0:
            self.state = TrackState.LOST

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox.tolist()
        return {
            "track_id": self.track_id,
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width":  x2 - x1,
            "height": y2 - y1,
            "score": self.score,
            "class": self.cls,
            "history": self.history[-10:],
        }


# ──────────────────────────────────────────────────────────────
# IoU Utilities
# ──────────────────────────────────────────────────────────────
def _iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """คำนวณ IoU matrix (N×M) ระหว่าง 2 ชุด bbox [x1,y1,x2,y2]"""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    ax1, ay1, ax2, ay2 = bboxes_a[:, 0], bboxes_a[:, 1], bboxes_a[:, 2], bboxes_a[:, 3]
    bx1, by1, bx2, by2 = bboxes_b[:, 0], bboxes_b[:, 1], bboxes_b[:, 2], bboxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter + 1e-6

    return inter / union


def _greedy_match(cost_matrix: np.ndarray, thresh: float) -> tuple[list, list, list]:
    """
    Greedy matching — จับคู่ track ↔ detection ด้วย IoU > thresh
    คืน: (matched_pairs, unmatched_track_idx, unmatched_det_idx)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    matched, unmatched_t, unmatched_d = [], [], []
    used_t, used_d = set(), set()

    # เรียงจากคู่ที่ IoU สูงสุด
    rows, cols = np.where(cost_matrix >= thresh)
    if len(rows) == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    scores = cost_matrix[rows, cols]
    order  = np.argsort(-scores)
    for idx in order:
        r, c = rows[idx], cols[idx]
        if r in used_t or c in used_d:
            continue
        matched.append((r, c))
        used_t.add(r)
        used_d.add(c)

    unmatched_t = [i for i in range(cost_matrix.shape[0]) if i not in used_t]
    unmatched_d = [i for i in range(cost_matrix.shape[1]) if i not in used_d]
    return matched, unmatched_t, unmatched_d


# ──────────────────────────────────────────────────────────────
# ByteTracker (Simplified)
# ──────────────────────────────────────────────────────────────
class ByteTracker:
    """
    Simplified ByteTrack:
    1. High-score detections → match tracked tracks (IoU)
    2. Low-score  detections → rescue lost tracks
    3. Unmatched tracked → mark lost
    4. Lost too long → remove
    """

    def __init__(
        self,
        track_thresh:  float = 0.5,
        track_buffer:  int   = 30,
        match_thresh:  float = 0.8,
        low_thresh:    float = 0.1,
        revive_thresh: float = 0.3,
    ):
        self.track_thresh  = track_thresh
        self.track_buffer  = track_buffer   # max frames a track survives without match
        self.match_thresh  = match_thresh
        self.low_thresh    = low_thresh
        # IoU needed to hand a LOST track's id back to a high-score detection.
        # Looser than match_thresh on purpose: a person who was occluded for a
        # few frames has moved, so their new box overlaps the remembered one
        # less than a frame-to-frame match would. Still tight enough that two
        # different people never inherit each other's id — boxes overlapping by
        # a third are essentially the same place.
        self.revive_thresh = revive_thresh

        self._tracks:  list[Track] = []
        self._next_id: int = 1

    # ── Public ──────────────────────────────────────────────
    def update(self, detections: list[dict]) -> list[Track]:
        """
        detections: list of dicts with keys x,y,width,height,confidence,class
        returns: list of active Track objects
        """
        if not detections:
            for t in self._tracks:
                t.mark_lost()
            self._remove_dead()
            return self._active_tracks()

        # Convert to bbox arrays
        det_bboxes = self._dets_to_bboxes(detections)
        det_scores = np.array([d.get("confidence", 1.0) for d in detections])
        det_cls    = [d.get("class", "person") for d in detections]

        # Separate high / low score detections
        high_mask = det_scores >= self.track_thresh
        low_mask  = (~high_mask) & (det_scores >= self.low_thresh)

        high_bboxes = det_bboxes[high_mask]
        high_scores = det_scores[high_mask]
        high_cls    = [c for c, m in zip(det_cls, high_mask) if m]

        low_bboxes  = det_bboxes[low_mask]
        low_scores  = det_scores[low_mask]
        low_cls     = [c for c, m in zip(det_cls, low_mask) if m]

        tracked   = [t for t in self._tracks if t.state == TrackState.TRACKED]
        lost      = [t for t in self._tracks if t.state == TrackState.LOST]

        # Step 1: Match high-score dets with tracked tracks
        t_bboxes = np.array([t.bbox for t in tracked]) if tracked else np.empty((0, 4))
        iou_mat  = _iou_matrix(t_bboxes, high_bboxes)
        matched, unmatched_t, unmatched_d = _greedy_match(iou_mat, self.match_thresh)

        for ti, di in matched:
            tracked[ti].update(high_bboxes[di], high_scores[di])

        # Step 2: REVIVE lost tracks from the high-score detections that no
        # tracked track claimed.
        #
        # Without this the second stage only ever offered LOW-score detections to
        # lost tracks, so the ordinary case — a worker walks behind a machine for
        # a moment and reappears clearly — could not be matched: the clear
        # reappearance is a HIGH-score detection, step 1 never considers lost
        # tracks, and step 4 below immediately spent it on a brand-new id. Every
        # brief occlusion therefore renamed the person, which resets that track's
        # PPE confirm window, its WORN-hold and its zone dwell state, and is the
        # "ByteTrack churns ids" behaviour the engine works around elsewhere.
        revived = set()
        if lost and unmatched_d:
            l_bboxes = np.array([t.bbox for t in lost])
            sub      = high_bboxes[unmatched_d]
            m2, _, _ = _greedy_match(_iou_matrix(l_bboxes, sub), self.revive_thresh)
            claimed  = set()
            for li, si in m2:
                di = unmatched_d[si]
                lost[li].update(high_bboxes[di], high_scores[di])
                revived.add(li)
                claimed.add(di)
            unmatched_d = [d for d in unmatched_d if d not in claimed]

        # Step 3: Try to rescue the STILL-lost tracks with low-score dets
        remaining_lost = [t for i, t in enumerate(lost) if i not in revived]
        if remaining_lost and len(low_bboxes) > 0:
            l_bboxes  = np.array([t.bbox for t in remaining_lost])
            iou_mat2  = _iou_matrix(l_bboxes, low_bboxes)
            matched2, _, _ = _greedy_match(iou_mat2, 0.5)
            rescued = set()
            for li, di in matched2:
                remaining_lost[li].update(low_bboxes[di], low_scores[di])
                rescued.add(li)
            for li, t in enumerate(remaining_lost):
                if li not in rescued:
                    t.mark_lost()
        else:
            for t in remaining_lost:
                t.mark_lost()

        # Step 4: Unmatched tracked → mark lost
        for ti in unmatched_t:
            tracked[ti].mark_lost()

        # Step 5: New tracks from the high-score dets nobody could claim
        for di in unmatched_d:
            score = high_scores[di]
            if score >= self.track_thresh:
                new_t = Track(
                    track_id = self._next_id,
                    bbox     = high_bboxes[di],
                    score    = score,
                    cls      = high_cls[di],
                    state    = TrackState.TRACKED,
                )
                self._tracks.append(new_t)
                self._next_id += 1

        self._remove_dead()
        return self._active_tracks()

    def reset(self):
        self._tracks  = []
        self._next_id = 1

    # ── Private ─────────────────────────────────────────────
    def _remove_dead(self):
        self._tracks = [
            t for t in self._tracks
            if not (t.state == TrackState.LOST and t.time_lost > self.track_buffer)
        ]

    def _active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state in (TrackState.NEW, TrackState.TRACKED)]

    @staticmethod
    def _dets_to_bboxes(dets: list[dict]) -> np.ndarray:
        bboxes = []
        for d in dets:
            cx, cy = d.get("x", 0), d.get("y", 0)
            w, h   = d.get("width", 0), d.get("height", 0)
            bboxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0, 4))
