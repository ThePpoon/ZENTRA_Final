@echo off
rem ===========================================================================
rem  ZENTRA Safety AI - เปิดโปรแกรม
rem
rem  หมายเหตุสำหรับผู้พัฒนา:
rem  - ไฟล์นี้ต้องเป็น CRLF (build_windows_dist.sh แปลงให้)
rem  - ใช้ python.exe ไม่ใช่ pythonw.exe โดยตั้งใจ: app.py เปิดแบบเต็มจอไม่มี
rem    title bar ถ้า WebView2 มีปัญหาจะได้จอดำที่ปิดไม่ได้และไม่มี log เลย
rem    หน้าต่าง console นี้คือช่องทาง debug เดียวบนเครื่องที่เราเข้าไม่ถึง
rem  - env ที่ตั้งตรงนี้ชนะค่าใน backend\.env เสมอ เพราะ config.py เรียก
rem    load_dotenv(override=False)
rem ===========================================================================
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto NOT_INSTALLED

rem ── ภาษาไทย / emoji ใน console (โมดูล AI print ทั้งไทยและ emoji) ─────────
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem ── ค่าประมวลผล ─────────────────────────────────────────────────────────
rem STREAM_FPS = เฟรม/วินาที ที่ส่งขึ้นจอ "ต่อกล้อง 1 ตัว" (ไม่ใช่ค่ารวม)
rem กล้อง 4 ตัว x 30fps = 120 ภาพ/วินาที (~12 MB/s ของ base64 JPEG) ผ่าน
rem WebSocket เส้นเดียวและ WebView หน้าต่างเดียว -> ส่งไม่ทัน คิวบวม
rem และภาพในกริดจะค้างอยู่ที่เฟรมเดิม. 15 พอสำหรับ CCTV; ปรับละเอียดได้ที่
rem data\settings.json -> display (stream_fps / stream_width / stream_height)
rem ซึ่งจะทับค่านี้ตอนเปิดโปรแกรม
set STREAM_FPS=15
set PPE_IMGSZ=640
set INFERENCE_CONFIDENCE=0.30
rem ล็อกไว้ที่ yolo: distribution ไม่ได้ลง mediapipe (ดู requirements-win.in)
rem ถ้ามีใครไปแก้ .env เป็น mediapipe จะ ImportError ทันที บรรทัดนี้กันไว้
set FALL_POSE_BACKEND=yolo

rem ── ไม่มีการ์ดจอ -> ตรึงจำนวน thread ของ OMP/MKL (เร็วขึ้นราว 2.7 เท่า) ──
echo กำลังเริ่มระบบ ...
.venv\Scripts\python.exe -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 goto CPU_MODE
echo โหมด: การ์ดจอ NVIDIA (CUDA)
goto RUN

:CPU_MODE
echo โหมด: CPU
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set KMP_BLOCKTIME=1

:RUN
echo.
echo ------------------------------------------------------------
echo  อย่าปิดหน้าต่างสีดำนี้ระหว่างใช้งาน (เป็นตัวรันโปรแกรม)
echo  ออกจากโปรแกรม: กดปุ่มออกในโปรแกรม (มุมขวาบน) หรือกด F11
echo ------------------------------------------------------------
echo.
.venv\Scripts\python.exe app.py
if errorlevel 1 goto CRASHED
exit /b 0

:CRASHED
echo.
echo ============================================================
echo  [ผิดพลาด] โปรแกรมปิดตัวลงผิดปกติ
echo  กรุณาถ่ายภาพข้อความด้านบนส่งให้ผู้ดูแลระบบ
echo.
echo  ถ้าเปิดแล้วเจอจอดำ/จอว่าง: เครื่องอาจไม่มี WebView2
echo  ติดตั้งจาก https://developer.microsoft.com/microsoft-edge/webview2/
echo  แล้วดูหัวข้อ "ปัญหาที่พบบ่อย" ในคู่มือ MANUAL_TH.html
echo ============================================================
echo.
pause
exit /b 1

:NOT_INSTALLED
echo.
echo [ผิดพลาด] ยังไม่ได้ติดตั้งระบบ
echo กรุณาดับเบิลคลิก INSTALL.bat ก่อน แล้วค่อยเปิด ZENTRA.bat อีกครั้ง
echo.
pause
exit /b 1
