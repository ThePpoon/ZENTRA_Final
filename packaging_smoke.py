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


# ── forensics for a native-import failure ────────────────────────────────
# A missing C++ runtime is only the most COMMON cause of WinError 1114, not
# the only one, and asserting it when the DLLs are demonstrably present sends
# the person at the machine off to install something they already have. When
# the easy explanation does not hold, collect the facts that separate the
# remaining ones instead of guessing again.

def _file_version(p: Path) -> str:
    """Read a DLL's version resource. Old system DLLs load but export too
    little for a binary built against a newer toolset, and that shows up here
    and nowhere else."""
    try:
        import ctypes
        ver = ctypes.windll.version
        size = ver.GetFileVersionInfoSizeW(str(p), None)
        if not size:
            return "?"
        buf = ctypes.create_string_buffer(size)
        ver.GetFileVersionInfoW(str(p), 0, size, buf)
        block, length = ctypes.c_void_p(), ctypes.c_uint()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(block),
                                  ctypes.byref(length)):
            return "?"
        # VS_FIXEDFILEINFO: [0]=signature [1]=strucVersion [2]=fileVersionMS
        # [3]=fileVersionLS
        ffi = ctypes.cast(block, ctypes.POINTER(ctypes.c_uint * 4)).contents
        ms, ls = ffi[2], ffi[3]
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception:
        return "?"


def _torch_lib_dir() -> Path | None:
    """Locate torch/lib without importing torch — importing is what fails."""
    import site
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for r in roots:
        d = Path(r) / "torch" / "lib"
        if d.is_dir():
            return d
    return None


def _dll_forensics() -> list[str]:
    import ctypes

    out: list[str] = []
    sysdir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"

    out.append("ไลบรารี C++ ของระบบ (System32):")
    for d in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
              "ucrtbase.dll"):
        p = sysdir / d
        if p.exists():
            out.append(f"     มี    {d:<20} v{_file_version(p)}"
                       f"  {p.stat().st_size:,} ไบต์")
        else:
            out.append(f"     ไม่มี  {d}")

    try:
        k32 = ctypes.windll.kernel32
        avx = bool(k32.IsProcessorFeaturePresent(39))   # PF_AVX_INSTRUCTIONS
        avx2 = bool(k32.IsProcessorFeaturePresent(40))  # PF_AVX2_INSTRUCTIONS
        out.append(f"ชุดคำสั่งของ CPU: AVX={'มี' if avx else 'ไม่มี'}"
                   f"  AVX2={'มี' if avx2 else 'ไม่มี'}")
        if not avx2:
            out.append("     ** PyTorch สำหรับ Windows ต้องใช้ CPU ที่มี AVX2 **")
            out.append("     ** ถ้า CPU ไม่มี จะพังแบบนี้เสมอ แก้ด้วยการลง "
                       "ไลบรารีเพิ่มไม่ได้ **")
    except Exception as e:
        out.append(f"ชุดคำสั่งของ CPU: อ่านไม่ได้ ({e})")

    try:
        import importlib.metadata as md
        out.append(f"torch ที่ติดตั้งไว้: {md.version('torch')}")
    except Exception:
        out.append("torch ที่ติดตั้งไว้: อ่านรุ่นไม่ได้")

    libdir = _torch_lib_dir()
    if libdir is None:
        out.append("ไม่พบโฟลเดอร์ torch\\lib")
        return out

    dlls = sorted(libdir.glob("*.dll"))
    total = sum(d.stat().st_size for d in dlls)
    out.append(f"torch\\lib: {len(dlls)} ไฟล์ รวม {total / 1048576:.0f} MB")
    empty = [d.name for d in dlls if d.stat().st_size == 0]
    if empty:
        out.append("     ** ไฟล์ขนาด 0 ไบต์ (เขียนไม่ครบ): "
                   + ", ".join(empty[:6]) + " **")

    # Load the core DLLs one at a time. Whichever fails first is the real
    # subject of the error message, which only ever names c10.dll.
    try:
        with os.add_dll_directory(str(libdir)):
            for name in ("c10.dll", "c10_cuda.dll"):
                p = libdir / name
                if not p.exists():
                    continue
                try:
                    ctypes.WinDLL(str(p))
                    out.append(f"     โหลด {name} ... ผ่าน "
                               f"({p.stat().st_size:,} ไบต์)")
                except OSError as e:
                    out.append(f"     โหลด {name} ... ล้มเหลว "
                               f"WinError {getattr(e, 'winerror', '?')} "
                               f"({p.stat().st_size:,} ไบต์)")
    except Exception as e:
        out.append(f"     ทดสอบโหลด DLL ไม่ได้: {e}")

    return out


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
        if absent:
            print(f"  [{n}] เครื่องนี้ยังไม่มี Visual C++ Redistributable")
        else:
            print(f"  [{n}] โหลดไลบรารีของ AI ไม่ได้ (ไฟล์ .dll เริ่มทำงานไม่สำเร็จ)")
        print(f"      ทำให้ล้มพร้อมกัน {len(dll_hits)} รายการ:")
        for lbl in dll_hits:
            print(f"        - {lbl}")
        print()

        if absent:
            print("      ไฟล์ระบบที่ขาด: " + ", ".join(absent))
            print("      PyTorch กับตัวตรวจจับการล้มถูกคอมไพล์ด้วย Microsoft C++")
            print("      จึงต้องใช้ไลบรารีตัวนี้ ตัวติดตั้ง Python ไม่ได้ลงให้")
            print()
            print("      วิธีแก้:")
            print(f"        1. โหลด {VC_URL}")
            print("        2. ดับเบิลคลิกติดตั้ง (จะมีหน้าต่างขออนุญาต ให้กด Yes)")
            print("        3. ดับเบิลคลิก ZENTRA.bat อีกครั้ง")
        else:
            # Say plainly that the usual explanation is ruled out. OpenCV is
            # built against the same runtime; if it imported, the runtime is
            # there, and claiming otherwise sends people to reinstall
            # something they already have.
            print("      ไลบรารี C++ ของระบบ *มีครบแล้ว* จึงไม่ใช่สาเหตุนี้")
            print("      ข้อมูลด้านล่างคือหลักฐานที่เก็บได้จากเครื่องนี้:")
            print()
            for line in _dll_forensics():
                print("      " + line)
            print()
            print("      สิ่งที่ควรลองตามลำดับ:")
            print("        1. ถ้าเห็นว่า CPU ไม่มี AVX2 -> เครื่องนี้ใช้ PyTorch")
            print("           รุ่นมาตรฐานไม่ได้ ต้องเปลี่ยนเครื่อง")
            print("        2. ปิดโปรแกรมป้องกันไวรัสชั่วคราว แล้วลงไลบรารีใหม่:")
            print("             .venv\\Scripts\\python.exe -m pip install "
                  "--force-reinstall --no-cache-dir torch torchvision")
            print("        3. ถ้ายังไม่หาย ลบโฟลเดอร์ .venv ทิ้งแล้วกด "
                  "ZENTRA.bat ใหม่")
            print("        4. ส่งภาพหน้าจอทั้งหมดนี้ให้ผู้ดูแลระบบ")

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
