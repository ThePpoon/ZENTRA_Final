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

Exit code 0 = good, 1 = something is wrong. Messages are Thai; the console is
run with chcp 65001 by INSTALL.bat.
"""
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILED: list[str] = []


def step(label: str, fn):
    sys.stdout.write(f"      - {label} ... ")
    sys.stdout.flush()
    try:
        detail = fn()
        print(f"ok{f'  ({detail})' if detail else ''}")
    except Exception as e:
        print("ล้มเหลว")
        FAILED.append(f"{label}: {e.__class__.__name__}: {e}")
        traceback.print_exc(limit=2)


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
        print("  พบปัญหา %d รายการ:" % len(FAILED))
        for f in FAILED:
            print(f"    * {f}")
        sys.exit(1)

    print("  ทดสอบผ่านทั้งหมด")
    sys.exit(0)
