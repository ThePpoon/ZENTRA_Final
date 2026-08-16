"""ZENTRA install smoke test — run by INSTALL.bat as the last step.

This is a real .py file rather than an inline `python -c` in the .bat on
purpose: quoting Python through CMD is a reliability trap, and this is the one
piece of the installer that has to give a clear answer on a machine nobody can
log into.

It proves the things that actually break in the field:
  - torch imports and reports which device it will use
  - cv2 imports (the DLL load is the usual Windows failure)
  - all three YOLO weights load
  - the TFLite fall model loads AND has the shape the code assumes
  - pywebview imports (pulls pythonnet → the WebView2 bridge)

Failures are grouped by CAUSE, not listed as symptoms. One missing system
library makes five of these tests fail at once, and a list of five identical
"DLL initialization routine failed" tracebacks tells the person at the machine
nothing they can act on. See `diagnose()`.

Exit code 0 = good, 1 = something is wrong. Messages are Thai; the console is
run with chcp 65001 by INSTALL.bat.
"""
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILED: list[tuple[str, BaseException]] = []


def step(label: str, fn):
    sys.stdout.write(f"      - {label} ... ")
    sys.stdout.flush()
    try:
        detail = fn()
        print(f"ok{f'  ({detail})' if detail else ''}")
    except Exception as e:
        print("ล้มเหลว")
        FAILED.append((label, e))


# ── the tests ────────────────────────────────────────────────────────────
def _torch():
    import torch
    if torch.cuda.is_available():
        return f"{torch.__version__} — ใช้การ์ดจอ {torch.cuda.get_device_name(0)}"
    return f"{torch.__version__} — ใช้ CPU"


def _cv2():
    import cv2
    return f"OpenCV {cv2.__version__}"


def _yolo(name: str):
    def run():
        from ultralytics import YOLO
        p = HERE / "backend" / "models" / name
        if not p.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ {p}")
        YOLO(str(p))
        return f"{p.stat().st_size // (1024 * 1024)} MB"
    return run


def _tflite():
    from ai_edge_litert.interpreter import Interpreter
    p = HERE / "backend" / "assets" / "models" / "fall_detection_transformer.tflite"
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {p}")
    it = Interpreter(model_path=str(p))
    it.allocate_tensors()
    shape = tuple(int(v) for v in it.get_input_details()[0]["shape"])
    # backend/utils/fall_detector.py builds this same check from its own
    # INPUT_TIMESTEPS (30) and NUM_FEATURES (17 landmarks x 3 = 51) and raises
    # at runtime if the model disagrees. Verified against the shipped model.
    # Catch it here, at install time, rather than mid-shift.
    if shape != (1, 30, 51):
        raise RuntimeError(f"รูปทรง input ผิด: {shape} (ต้องเป็น (1, 30, 51))")
    return f"input {shape}"


def _webview():
    # The import is the real test: on Windows it drags in pythonnet/clr-loader,
    # which is the WebView2 bridge and the usual failure. Note pywebview exposes
    # no __version__ attribute — ask the package metadata instead.
    import importlib.metadata as md

    import webview  # noqa: F401
    return f"pywebview {md.version('pywebview')}"


# ── turning symptoms into a cause ────────────────────────────────────────
VC_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def _vcruntime_missing() -> list[str]:
    """The C++ runtime torch and LiteRT are built against. Python ships
    vcruntime140.dll beside python.exe but NOT msvcp140.dll or
    vcruntime140_1.dll, so a machine that never had Visual Studio or a C++
    app installed is missing exactly the two that matter."""
    sysdir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return [d for d in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
            if not (sysdir / d).exists()]


def _is_dll_failure(e: BaseException) -> bool:
    if getattr(e, "winerror", None) == 1114:
        return True
    text = str(e).lower()
    return ("dll load failed" in text
            or "dynamic link library (dll) initialization routine failed" in text)


def _broken_stdlib_file(e: BaseException) -> str | None:
    """A SyntaxError inside Python's own Lib/ means the interpreter's files are
    damaged — truncated or half-written — not that our code is wrong."""
    if not isinstance(e, SyntaxError) or not e.filename:
        return None
    f = Path(e.filename)
    try:
        stdlib = Path(sys.base_prefix).resolve() / "Lib"
        f.resolve().relative_to(stdlib)
    except (ValueError, OSError):
        return None
    return str(f)


def diagnose() -> None:
    """Print one block per distinct cause, most actionable first."""
    dll_hits = [lbl for lbl, e in FAILED if _is_dll_failure(e)]
    broken = [(lbl, p) for lbl, e in FAILED if (p := _broken_stdlib_file(e))]
    missing_files = [lbl for lbl, e in FAILED if isinstance(e, FileNotFoundError)]
    no_module = [lbl for lbl, e in FAILED
                 if isinstance(e, ModuleNotFoundError) and not _is_dll_failure(e)]

    print()
    print("  " + "=" * 62)
    print(f"  พบปัญหา {len(FAILED)} รายการ — สาเหตุที่แท้จริงมีดังนี้")
    print("  " + "=" * 62)

    n = 0

    if dll_hits:
        n += 1
        absent = _vcruntime_missing()
        print()
        print(f"  [{n}] เครื่องนี้ยังไม่มี Visual C++ Redistributable")
        print(f"      ทำให้ล้มพร้อมกัน {len(dll_hits)} รายการ:")
        for lbl in dll_hits:
            print(f"        - {lbl}")
        print()
        if absent:
            print("      ไฟล์ระบบที่ขาด: " + ", ".join(absent))
        else:
            print("      (ไฟล์อยู่ครบแล้ว แต่อาจเป็นรุ่นเก่าเกินไป)")
        print("      PyTorch กับตัวตรวจจับการล้มถูกคอมไพล์ด้วย Microsoft C++")
        print("      จึงต้องใช้ไลบรารีตัวนี้ ตัวติดตั้ง Python ไม่ได้ลงให้")
        print()
        print("      วิธีแก้:")
        print(f"        1. โหลด {VC_URL}")
        print("        2. ดับเบิลคลิกติดตั้ง (จะมีหน้าต่างขออนุญาต ให้กด Yes)")
        print("        3. ดับเบิลคลิก ZENTRA.bat อีกครั้ง")

    if broken:
        n += 1
        print()
        print(f"  [{n}] ไฟล์ของ Python เองเสียหาย (ติดตั้งไม่สมบูรณ์)")
        for lbl, p in broken:
            print(f"        - {lbl}")
            print(f"          ไฟล์ที่เสีย: {p}")
        print()
        print("      ไฟล์นี้เป็นของ Python ไม่ใช่ของ ZENTRA การที่มันอ่านไม่ออก")
        print("      แปลว่าตอนติดตั้ง Python เขียนไฟล์ไม่ครบ")
        print("      (ดิสก์เต็มระหว่างติดตั้ง หรือโปรแกรมป้องกันไวรัสขัดจังหวะ)")
        print()
        print("      วิธีแก้:")
        print("        1. เปิด Settings -> Apps -> Installed apps")
        print("        2. หา Python 3.11.9 -> กด ... -> Modify -> Repair")
        print("        3. ลบโฟลเดอร์ .venv ในโฟลเดอร์ ZENTRA ทิ้ง")
        print("        4. ดับเบิลคลิก ZENTRA.bat อีกครั้ง")

    if missing_files:
        n += 1
        print()
        print(f"  [{n}] ไฟล์โมเดล AI ไม่ครบ")
        for lbl in missing_files:
            print(f"        - {lbl}")
        print()
        print("      วิธีแก้: แตกไฟล์ ZIP ที่โหลดจาก GitHub ใหม่ทั้งหมด")
        print("      (คลิกขวา -> Extract All ห้ามลากไฟล์ออกมาทีละอัน)")

    if no_module:
        n += 1
        print()
        print(f"  [{n}] ไลบรารีติดตั้งไม่ครบ")
        for lbl in no_module:
            print(f"        - {lbl}")
        print()
        print("      วิธีแก้: ดับเบิลคลิก ZENTRA.bat อีกครั้ง")
        print("      (มักเกิดจากอินเทอร์เน็ตหลุดกลางทาง ของเดิมจะไม่โหลดซ้ำ)")

    classified = set(dll_hits) | {l for l, _ in broken} | set(missing_files) | set(no_module)
    rest = [(lbl, e) for lbl, e in FAILED if lbl not in classified]
    if rest:
        n += 1
        print()
        print(f"  [{n}] ปัญหาอื่นที่ยังระบุสาเหตุอัตโนมัติไม่ได้")
        for lbl, e in rest:
            print(f"        - {lbl}: {e.__class__.__name__}: {e}")
        print()
        print("      กรุณาถ่ายภาพหน้าจอนี้ส่งให้ผู้ดูแลระบบ")

    print()
    print("  " + "-" * 62)
    print("  รายละเอียดทางเทคนิค (สำหรับผู้ดูแลระบบ):")
    for lbl, e in FAILED:
        print(f"    * {lbl}: {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    print()
    step("PyTorch", _torch)
    step("OpenCV", _cv2)
    step("โมเดลตรวจจับคน (yolo11m.pt)", _yolo("yolo11m.pt"))
    step("โมเดลตรวจจับอุปกรณ์นิรภัย (ppe_finetuned.pt)", _yolo("ppe_finetuned.pt"))
    step("โมเดลท่าทาง (yolo11n-pose.pt)", _yolo("yolo11n-pose.pt"))
    step("โมเดลตรวจจับการล้ม (tflite)", _tflite)
    step("หน้าต่างโปรแกรม (pywebview)", _webview)
    print()

    if FAILED:
        diagnose()
        sys.exit(1)

    print("  ทดสอบผ่านทั้งหมด")
    sys.exit(0)
