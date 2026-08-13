"""
server/report.py — ZENTRA safety report (local PDF via matplotlib)

Builds an A4 PDF safety report from the local event store, following common
occupational-safety report structure (header/identity, KPI summary, trend,
severity breakdown, detailed event log with evidence + a corrective-action
column, and a signature block). No external service, no new dependency.

Thai text renders with the bundled OFL font (backend/assets/fonts) so it works
in the Linux container; falls back to a system Thai font if present.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")                      # headless backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from server import store

_REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"
_BUNDLED_FONT = Path(__file__).parent.parent / "backend" / "assets" / "fonts" / "Sarabun-SemiBold.ttf"

_FONT_READY = False

_LVL_COLOR = {"warning": "#ea580c", "alert": "#2563eb", "emergency": "#dc2626"}
_LVL_TH    = {"warning": "เตือน (PPE)", "alert": "อันตราย (เขต)", "emergency": "ฉุกเฉิน"}
_TYPE_TH   = {"ppe": "PPE", "zone": "เขตหวงห้าม", "fall": "การล้ม", "heat": "ความร้อน"}


def _ensure_thai_font():
    global _FONT_READY
    if _FONT_READY:
        return
    # Prefer the bundled OFL font so Thai always renders (esp. in the container).
    if _BUNDLED_FONT.exists():
        try:
            fm.fontManager.addfont(str(_BUNDLED_FONT))
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=str(_BUNDLED_FONT)).get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            _FONT_READY = True
            return
        except Exception:
            pass
    for name in ("Sarabun", "Tahoma", "Leelawadee UI", "Leelawadee", "TH Sarabun New"):
        try:
            path = fm.findfont(name, fallback_to_default=False)
            if path and Path(path).exists():
                fm.fontManager.addfont(path)
                matplotlib.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
                break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def _safety_index(total: int) -> int:
    """Estimated safety index 0–100 (fewer confirmed incidents → higher)."""
    return max(0, 100 - total * 4)


def build_daily_pdf(day: Optional[str] = None, start: Optional[str] = None,
                    end: Optional[str] = None, org: Optional[dict] = None) -> Path:
    """Render the safety report PDF and return its local path.
    Single day (`day`) or an inclusive range (`start`,`end`)."""
    _ensure_thai_font()
    org = org or {}
    is_range = bool(start and end)
    if not is_range:
        day = day or date.today().strftime("%Y-%m-%d")
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stats  = store.today_stats(day=day, start=start, end=end)
    events = store.list_events(limit=100000, offset=0, day=day, start=start, end=end)["events"]
    period = (f"ช่วงวันที่ {start} ถึง {end}" if is_range else f"วันที่ {day}")

    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
    fig.patch.set_facecolor("white")

    # ── Header / identity ────────────────────────────────────
    company  = org.get("company") or "ZENTRA Industrial Safety"
    site     = org.get("site") or "-"
    preparer = org.get("preparer") or "-"
    fig.text(0.06, 0.960, "รายงานความปลอดภัย (PPE & พื้นที่หวงห้าม)",
             fontsize=19, fontweight="bold", color="#0f172a")
    fig.text(0.06, 0.940, company, fontsize=12, color="#1e293b")
    fig.text(0.06, 0.924, f"สถานที่/ไลน์: {site}    ·    {period}", fontsize=10, color="#475569")
    fig.text(0.06, 0.909,
             f"ผู้จัดทำ: {preparer}    ·    ออกรายงาน: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
             f"    ·    ระบบ ZENTRA v1.0",
             fontsize=8.5, color="#94a3b8")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.899, 0.899], color="#cdd6e3", lw=1))

    # ── KPI row ──────────────────────────────────────────────
    idx = _safety_index(stats["total"])
    kpis = [
        ("ดัชนีความปลอดภัย", f"{idx}", "#16a34a" if idx >= 80 else ("#d97706" if idx >= 60 else "#dc2626")),
        ("เหตุการณ์รวม", str(stats["total"]),          "#2563eb"),
        ("PPE", str(stats["ppe_violations"]),          "#ea580c"),
        ("เข้าเขต", str(stats["zone_intrusions"]),      "#2563eb"),
        ("ฉุกเฉิน/ล้ม", str(stats["emergency"]),        "#dc2626"),
    ]
    n = len(kpis)
    for i, (label, val, color) in enumerate(kpis):
        x = 0.06 + i * (0.88 / n)
        fig.text(x + 0.015, 0.858, val, fontsize=25, fontweight="bold", color=color)
        fig.text(x + 0.015, 0.838, label, fontsize=9.5, color="#475569")

    # ── Trend chart (hourly for a day, per-day for a range) ──
    ax = fig.add_axes([0.08, 0.545, 0.86, 0.235])
    if is_range:
        counts = store.daily_counts(start=start, end=end)
        labels = list(counts.keys())
        values = [counts[k] for k in labels]
        ax.bar(range(len(labels)), values, color="#2563eb", width=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l[5:] for l in labels], fontsize=7, color="#64748b", rotation=45, ha="right")
        ax.set_title("จำนวนเหตุการณ์รายวัน", fontsize=12, color="#0f172a", loc="left")
    else:
        hourly = store.hourly(day)
        hours  = [f"{h:02d}" for h in range(24)]
        values = [hourly.get(h, 0) for h in hours]
        ax.bar(range(24), values, color="#2563eb", width=0.7)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=8, color="#64748b")
        ax.set_title("จำนวนเหตุการณ์รายชั่วโมง", fontsize=12, color="#0f172a", loc="left")
    ax.tick_params(axis="y", labelsize=8, colors="#64748b")
    if max(values, default=0) <= 5:
        ax.set_ylim(0, 5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cdd6e3")
    ax.spines["bottom"].set_color("#cdd6e3")
    ax.grid(axis="y", color="#e2e8f0", lw=0.6)

    # ── Severity breakdown ───────────────────────────────────
    sev = {"warning": 0, "alert": 0, "emergency": 0}
    for e in events:
        if e["level"] in sev:
            sev[e["level"]] += 1
    fig.text(0.06, 0.505, "สรุปตามระดับความรุนแรง", fontsize=11.5, fontweight="bold", color="#0f172a")
    yb = 0.478
    for lvl in ("emergency", "alert", "warning"):
        c = _LVL_COLOR[lvl]
        fig.text(0.08, yb, _LVL_TH[lvl], fontsize=9.5, color="#334155")
        total_sev = max(1, len(events))
        w = 0.42 * (sev[lvl] / total_sev)
        fig.add_artist(plt.Rectangle((0.30, yb - 0.004), 0.42, 0.013, color="#eef2f7"))
        fig.add_artist(plt.Rectangle((0.30, yb - 0.004), max(0.004, w), 0.013, color=c))
        fig.text(0.74, yb, str(sev[lvl]), fontsize=9.5, fontweight="bold", color=c)
        yb -= 0.026

    # ── Evidence gallery (up to 5 thumbnails) ────────────────
    snaps = [e for e in events if e.get("has_snapshot")][:5]
    if snaps:
        fig.text(0.06, 0.388, "ตัวอย่างภาพหลักฐาน", fontsize=11.5, fontweight="bold", color="#0f172a")
        for i, e in enumerate(snaps):
            p = store.snapshot_path(e["id"])
            if not p:
                continue
            try:
                img = plt.imread(str(p))
            except Exception:
                continue
            axx = fig.add_axes([0.06 + i * 0.178, 0.300, 0.16, 0.075])
            axx.imshow(img)
            axx.set_xticks([]); axx.set_yticks([])
            for sp in axx.spines.values():
                sp.set_color("#cdd6e3")
            axx.set_title(e.get("time", ""), fontsize=6.5, color="#64748b")

    # ── Event log (with a manual corrective-action column) ──
    fig.text(0.06, 0.262, "บันทึกเหตุการณ์ (ล่าสุด)", fontsize=11.5, fontweight="bold", color="#0f172a")
    fig.text(0.06, 0.245, "เวลา", fontsize=8, fontweight="bold", color="#64748b")
    fig.text(0.17, 0.245, "ประเภท", fontsize=8, fontweight="bold", color="#64748b")
    fig.text(0.28, 0.245, "รายละเอียด", fontsize=8, fontweight="bold", color="#64748b")
    fig.text(0.70, 0.245, "การแก้ไข/ผู้รับผิดชอบ", fontsize=8, fontweight="bold", color="#64748b")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.238, 0.238], color="#cdd6e3", lw=0.8))
    y = 0.222
    shown = 8
    if not events:
        fig.text(0.08, y, "— ไม่มีเหตุการณ์ในช่วงนี้ —", fontsize=10, color="#94a3b8")
    else:
        for e in events[:shown]:
            c = _LVL_COLOR.get(e["level"], "#475569")
            fig.text(0.06, y, f"{e.get('date','')[5:]} {e['time']}", fontsize=8, color="#64748b")
            fig.text(0.17, y, _TYPE_TH.get(e["type"], e["type"]), fontsize=8, fontweight="bold", color=c)
            fig.text(0.28, y, (e["message"] or "")[:44], fontsize=8, color="#0f172a")
            fig.text(0.70, y, "[ภาพ] " if e["has_snapshot"] else "", fontsize=7, color="#94a3b8")
            fig.add_artist(plt.Line2D([0.70, 0.93], [y - 0.006, y - 0.006], color="#e2e8f0", lw=0.6))
            y -= 0.0150
    if len(events) > shown:
        fig.text(0.06, y - 0.002, f"… และอีก {len(events) - shown} เหตุการณ์ (ดูไฟล์ CSV)",
                 fontsize=8, color="#94a3b8")

    # ── Signature block ──────────────────────────────────────
    ys = 0.085
    for i, role in enumerate(("ผู้จัดทำรายงาน", "ผู้ตรวจสอบ (จป.)", "ผู้อนุมัติ")):
        x = 0.06 + i * 0.30
        fig.add_artist(plt.Line2D([x, x + 0.24], [ys, ys], color="#94a3b8", lw=0.8))
        fig.text(x, ys - 0.016, role, fontsize=8.5, color="#475569")
        fig.text(x, ys - 0.030, "วันที่ ......../......../........", fontsize=7.5, color="#94a3b8")

    fig.text(0.06, 0.028,
             "ข้อมูลจัดเก็บภายในเครื่อง (on-device) ตามหลัก PDPA · อ้างอิงแนวทาง OHS/ISO 45001 · สร้างโดย ZENTRA",
             fontsize=7.5, color="#94a3b8")

    tag = f"{start}_{end}" if is_range else day
    out = _REPORTS_DIR / f"zentra_report_{tag}.pdf"
    fig.savefig(str(out), format="pdf")
    plt.close(fig)
    return out


def daily_stats_for_line(day: Optional[str] = None) -> dict:
    """Stats dict shaped for alerts.line_notify.send_daily_report (text only)."""
    s = store.today_stats(day)
    return {
        "ppe_violations":  s["ppe_violations"],
        "zone_intrusions": s["zone_intrusions"],
        "fall_events":     s["falls"],
    }
