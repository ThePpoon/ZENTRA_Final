"""
server/power.py — host power state (AC / battery / Windows power plan).

Why this exists: it is the single largest performance variable measured on this
app, larger than any software optimisation in the pipeline. The SAME inference
code, same GPU, same models, measured:

    on AC power            229 ms per 4-camera tick   (~4.0 detections/s/camera)
    while power-limited   1051 ms per 4-camera tick   (~0.9 detections/s/camera)

a 4.6x swing. During the slow run the dGPU drew 19 W at 77% utilisation — a
laptop GPU held far below its power budget. Nothing in the logs said so, so the
symptom presented as "the AI is slow and the cameras freeze", which sends you
hunting through the wrong layer of the stack entirely.

An edge box left on battery or on a power-saver plan is therefore a
misconfiguration that looks exactly like a software defect. Detect it, say so
loudly at startup, and expose it in /api/status so it can never be the silent
explanation for a slow site again.

Read-only: this module never changes the machine's power settings. Switching a
Windows power plan is a system-wide change affecting everything the operator
runs, so it stays their decision — we only report and recommend.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys

# powercfg scheme GUIDs (stable across Windows versions)
_SCHEMES = {
    "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
    "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate performance",
}

_AC_OFFLINE, _AC_ONLINE = 0, 1


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),        # 0 battery, 1 AC, 255 unknown
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),  # 0..100, 255 unknown
        ("SystemStatusFlag", ctypes.c_ubyte),    # 1 = battery saver active
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def _active_scheme() -> str | None:
    """Active Windows power plan name, or None if it cannot be determined.

    Shells out to powercfg because there is no stable ctypes-only way to get the
    plan NAME. Kept short-timeout and fully non-fatal: a missing or slow powercfg
    must never delay or break startup.
    """
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True, text=True, timeout=3,
        ).stdout.lower()
    except Exception:
        return None
    for guid, name in _SCHEMES.items():
        if guid in out:
            return name
    return None


def power_status() -> dict:
    """{on_ac, battery_percent, battery_saver, plan, ok, warnings}.

    `ok` is False when the host is in a state known to throttle inference. Every
    field is None-tolerant: on a non-Windows host, or if the API is unavailable,
    the result reports "unknown" rather than guessing — an unknown power state
    must not masquerade as a good one.
    """
    st: dict = {
        "on_ac": None, "battery_percent": None, "battery_saver": None,
        "plan": None, "ok": True, "warnings": [],
    }
    if sys.platform == "win32":
        sps = _SYSTEM_POWER_STATUS()
        try:
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)):
                if sps.ACLineStatus in (_AC_OFFLINE, _AC_ONLINE):
                    st["on_ac"] = bool(sps.ACLineStatus == _AC_ONLINE)
                if sps.BatteryLifePercent <= 100:
                    st["battery_percent"] = int(sps.BatteryLifePercent)
                st["battery_saver"] = bool(sps.SystemStatusFlag == 1)
        except Exception:
            pass
        st["plan"] = _active_scheme()

    if st["on_ac"] is False:
        pct = st["battery_percent"]
        st["warnings"].append(
            "กำลังใช้ไฟจากแบตเตอรี่"
            + (f" ({pct}%)" if pct is not None else "")
            + " — GPU จะถูกจำกัดกำลังไฟ ตรวจจับช้าลงได้ถึง 4 เท่า กรุณาต่อสายไฟ"
        )
    if st["battery_saver"]:
        st["warnings"].append(
            "เปิดโหมดประหยัดพลังงาน (Battery saver) อยู่ — ปิดเพื่อให้ AI ทำงานเต็มความเร็ว"
        )
    if st["plan"] in ("Power saver", "Balanced"):
        st["warnings"].append(
            f"Windows power plan = '{st['plan']}' — แนะนำ 'High performance' "
            "สำหรับเครื่องที่รัน AI ตลอดเวลา"
        )
    st["ok"] = not st["warnings"]
    return st


def log_power_status() -> dict:
    """Print the power state at startup. Returns the status dict for reuse."""
    st = power_status()
    if st["on_ac"] is None and st["plan"] is None:
        return st                       # nothing detectable (non-Windows host)
    src = "AC" if st["on_ac"] else ("battery" if st["on_ac"] is False else "unknown")
    print(f"[Power] source={src} plan={st['plan'] or 'unknown'} "
          f"battery={st['battery_percent'] if st['battery_percent'] is not None else '-'}%")
    for w in st["warnings"]:
        print(f"[Power] ⚠️  {w}")
    return st
