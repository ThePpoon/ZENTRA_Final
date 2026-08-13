# utils/collector.py — ZENTRA Auto Data Collector for Training
# เก็บ frame + annotation อัตโนมัติเมื่อเกิด event
# รองรับ format: YOLO (.txt), COCO (.json), Roboflow upload

from __future__ import annotations
import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


def _cfg():
    import config as c
    return c


class DataCollector:
    """
    Auto-collect frames + YOLO-format annotations สำหรับ fine-tuning

    โครงสร้าง:
    data/collected/
    ├── ppe_violations/
    │   ├── 20260330_143012_frame0001.jpg
    │   └── 20260330_143012_frame0001.txt  (YOLO format)
    ├── zone_intrusions/
    ├── fall_events/
    └── normal/
    """

    def __init__(self):
        self.cfg      = _cfg()
        self.lock     = threading.Lock()
        self._counts: dict[str, int] = {}   # category → count
        self._frame_counter = 0
        self._last_save_ts: dict[str, float] = {}      # category → last save time
        self._last_thumb:  dict[str, np.ndarray] = {}  # category → 32x32 gray
        self._load_counts()

    @staticmethod
    def _thumb(frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (32, 32)).astype(np.float32)

    def _is_diverse(self, category: str, frame: np.ndarray) -> bool:
        """True if this frame is far enough (in time + appearance) from the last
        saved one — avoids storing 20 near-identical frames of the same moment."""
        cfg = self.cfg
        now = time.time()
        min_gap = getattr(cfg, "COLLECT_MIN_INTERVAL_SEC", 2.0)
        if now - self._last_save_ts.get(category, 0.0) < min_gap:
            return False
        th = self._thumb(frame)
        prev = self._last_thumb.get(category)
        if prev is not None:
            diff = float(np.mean(np.abs(th - prev)))   # mean abs pixel diff (0–255)
            if diff < getattr(cfg, "COLLECT_DEDUP_DIFF", 8.0):
                return False                            # too similar → skip
        return True

    # ── Public API ────────────────────────────────────────────
    def collect(
        self,
        frame: np.ndarray,
        predictions: list[dict],
        category: str = "normal",
        force: bool   = False,
    ) -> bool:
        """
        บันทึก frame + annotation

        Parameters
        ----------
        frame       : BGR numpy array
        predictions : list of Roboflow prediction dicts
        category    : 'ppe_violations' | 'zone_intrusions' | 'fall_events' | 'normal'
        force       : True = บันทึกทันที, False = ตรวจ quota ก่อน
        """
        if not self.cfg.AUTO_COLLECT_FRAMES:
            return False

        # Diversity gate: skip near-duplicate / too-frequent frames (better dataset)
        if not force and not self._is_diverse(category, frame):
            return False

        # Check quota — but do NOT charge it yet. A slot spent on a write that
        # then fails (disk full, read-only dir) is a slot lost forever: the count
        # would climb to the cap with nothing on disk and collection would stop
        # for a category that never collected anything.
        max_q = self.cfg.COLLECT_MAX_PER_CLASS
        with self.lock:
            if not force and self._counts.get(category, 0) >= max_q:
                return False

        # Remember this frame as the reference for diversity checks
        self._last_save_ts[category] = time.time()
        self._last_thumb[category]   = self._thumb(frame)

        save_dir = Path(self.cfg.COLLECTED_DIR) / category
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{ts}_f{self._frame_counter:06d}"
        img_path = save_dir / f"{stem}.jpg"
        lbl_path = save_dir / f"{stem}.txt"

        # Save image. imwrite REPORTS failure by returning False — it does not
        # raise — so an unwritable dir would otherwise look like a clean save.
        ok = cv2.imwrite(
            str(img_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.cfg.COLLECT_JPEG_QUALITY],
        )
        if not ok:
            print(f"[Collector] ⚠️ เขียนภาพไม่สำเร็จ: {img_path}")
            return False

        # Save YOLO annotation
        h, w = frame.shape[:2]
        yolo_lines = self._to_yolo(predictions, w, h)
        lbl_path.write_text("\n".join(yolo_lines))

        with self.lock:
            self._counts[category] = self._counts.get(category, 0) + 1
        self._frame_counter += 1
        return True

    def collect_dets(
        self,
        frame: np.ndarray,
        dets: list[dict],
        category: str = "normal",
        force: bool = False,
    ) -> bool:
        """collect() for the live engine, whose detections are corner-boxes
        (utils/detect_track shape: cls / conf / x1,y1,x2,y2) — not the centre-box
        Roboflow prediction dicts the InferencePipeline path produced."""
        return self.collect(frame, self._dets_to_preds(dets), category, force=force)

    @staticmethod
    def dets_from_recs(recs: list[dict]) -> list[dict]:
        """Engine records (one per person, with the PPE items associated to them)
        → the flat det list collect_dets() wants. Use this when the caller holds
        recs rather than the engine's last_dets — e.g. the fall loop, which runs
        on an older frame/recs pair than the newest detection."""
        dets: list[dict] = []
        for rec in recs or []:
            person = rec.get("person")
            if person:
                dets.append(person)
            dets.extend(rec.get("items") or [])
        return dets

    @staticmethod
    def _dets_to_preds(dets: list[dict]) -> list[dict]:
        """Engine dets → Roboflow-style prediction dicts, dropping boxes too weak
        to be worth a pseudo-label. The engine tracks at a LOW conf floor
        (PPE_TRACK_CONF ≈ 0.10) to keep ByteTrack ids alive through dips; writing
        those into the dataset would fill Roboflow with junk boxes to delete."""
        cfg = _cfg()
        item_floor   = getattr(cfg, "COLLECT_LABEL_CONF", 0.40)
        person_floor = getattr(cfg, "COLLECT_PERSON_CONF", 0.25)
        preds = []
        for d in dets or []:
            floor = person_floor if d.get("is_person") else item_floor
            if float(d.get("conf", 0.0)) < floor:
                continue
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            preds.append({
                "class":  d.get("cls", ""),
                "x":      (x1 + x2) / 2,
                "y":      (y1 + y2) / 2,
                "width":  x2 - x1,
                "height": y2 - y1,
            })
        return preds

    def collect_normal(self, frame: np.ndarray, predictions: list[dict], frame_id: int):
        """เก็บ normal frame ทุก N frames"""
        if frame_id % self.cfg.COLLECT_NORMAL_INTERVAL == 0:
            self.collect(frame, predictions, "normal")

    def get_stats(self) -> dict:
        with self.lock:
            return dict(self._counts)

    def reload_counts(self):
        """Re-read the per-category counts from disk.

        The quota lives in memory, so anything that deletes collected files behind
        the collector's back (the app's "ลบข้อมูลสะสม" button) leaves it thinking
        the quota is still full — collection then stops for good, silently, until
        the process restarts. Whoever deletes the files must call this."""
        with self.lock:
            self._counts.clear()
            self._last_save_ts.clear()
            self._last_thumb.clear()
            self._load_counts()

    def export_dataset_yaml(self, output_path: str | None = None) -> str:
        """สร้าง dataset.yaml สำหรับ YOLOv8 training"""
        cfg    = self.cfg
        base   = Path(cfg.COLLECTED_DIR)
        yaml_p = Path(output_path or cfg.DATA_DIR / "dataset.yaml")

        # Deployed 11-class taxonomy (matches the YOLO label indices _to_yolo writes)
        class_names = list(cfg.PPE_TAXONOMY)

        content = f"""# ZENTRA Auto-Collected Dataset
path: {base.resolve()}
train: .
val: .

nc: {len(class_names)}
names: {class_names}
"""
        yaml_p.write_text(content)
        print(f"[Collector] dataset.yaml → {yaml_p}")
        return str(yaml_p)

    def export_coco_json(self, category: str = "ppe_violations") -> str:
        """Export COCO format JSON (สำหรับ upload Roboflow)"""
        cfg      = self.cfg
        save_dir = Path(cfg.COLLECTED_DIR) / category
        images, annotations, ann_id = [], [], 1

        # Deployed 11-class taxonomy (same index space as the .txt labels)
        class_names = list(cfg.PPE_TAXONOMY)
        categories  = [{"id": i, "name": n} for i, n in enumerate(class_names)]

        for img_id, img_path in enumerate(sorted(save_dir.glob("*.jpg"))):
            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                continue

            img    = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w   = img.shape[:2]

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w, "height": h,
            })

            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                bw_abs = bw * w
                bh_abs = bh * h
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x1, y1, bw_abs, bh_abs],
                    "area": bw_abs * bh_abs,
                    "iscrowd": 0,
                })
                ann_id += 1

        out = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        out_path = Path(cfg.COLLECTED_DIR) / f"{category}_coco.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[Collector] COCO JSON → {out_path} ({len(images)} imgs, {len(annotations)} anns)")
        return str(out_path)

    # ── Private ───────────────────────────────────────────────
    def _to_yolo(self, predictions: list[dict], img_w: int, img_h: int) -> list[str]:
        """แปลง predictions → YOLO lines in the DEPLOYED 11-class index space
        (cfg.PPE_TAXONOMY). Must match ppe_finetuned.pt / roboflow_dl/data.yaml so
        collected frames are directly trainable — NOT sorted(display-label)."""
        cfg = _cfg()
        lines = []
        for pred in predictions:
            cls_id = cfg.ppe_taxo_index(pred.get("class", ""))
            if cls_id is None:
                continue
            cx = pred.get("x", 0) / img_w
            cy = pred.get("y", 0) / img_h
            bw = pred.get("width",  0) / img_w
            bh = pred.get("height", 0) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    def _load_counts(self):
        """โหลด count จาก disk เพื่อไม่ให้ quota reset ทุกครั้ง"""
        cfg = self.cfg
        for cat in ["ppe_violations", "zone_intrusions", "fall_events", "normal"]:
            d = Path(cfg.COLLECTED_DIR) / cat
            if d.exists():
                self._counts[cat] = len(list(d.glob("*.jpg")))


# Singleton
_collector_instance: Optional[DataCollector] = None


def get_collector() -> DataCollector:
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = DataCollector()
    return _collector_instance
