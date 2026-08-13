@echo off
rem ===========================================================================
rem  ZENTRA Safety AI - ตัวติดตั้ง (Windows 11 x64)
rem  รันซ้ำได้เสมอ ปลอดภัย ไม่ทับ backend\.env ที่ผู้ใช้แก้ไว้
rem
rem  หมายเหตุสำหรับผู้พัฒนา:
rem  - ไฟล์นี้ต้องเป็น CRLF ไม่งั้น CMD จะ parse if/goto พังแบบเงียบ ๆ
rem    (build_windows_dist.sh แปลงให้ตอน build)
rem  - ห้ามอ่านตัวแปรที่ set ไว้ใน if(...) block เดียวกัน - ใช้ goto แทน
rem  - Python/WebView2 ติดมากับ ZIP ใน setup\ ไม่ต้องพึ่ง python.org ตอนติดตั้ง
rem  - ติดตั้ง Python แบบ per-user (InstallAllUsers=0) เพื่อไม่ให้เด้ง UAC
rem ===========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo ==========================================
echo    ZENTRA Safety AI - ติดตั้งระบบ
echo ==========================================
echo.
echo ขั้นตอนนี้ใช้อินเทอร์เน็ต และใช้เวลาประมาณ 10-20 นาที
echo (ดาวน์โหลดไลบรารีประมาณ 500MB - 2.5GB ขึ้นกับว่ามีการ์ดจอ NVIDIA หรือไม่)
echo ระหว่างนี้จะเห็นข้อความวิ่งเยอะ ๆ เป็นเรื่องปกติ
echo.

rem ── [1/7] Python 3.11 - ติดตั้งให้เองถ้าไม่มี ────────────────────────────
echo [1/7] ตรวจหา Python 3.11 ...
set "PY311="
py -3.11 -V >nul 2>&1
if not errorlevel 1 goto PY_FOUND_LAUNCHER
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" goto PY_FOUND_LOCAL
goto PY_INSTALL

:PY_FOUND_LAUNCHER
set "PY311=py -3.11"
for /f "tokens=*" %%v in ('py -3.11 -V 2^>^&1') do echo       พบ %%v
goto MAKE_VENV

:PY_FOUND_LOCAL
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
echo       พบ Python 3.11 ติดตั้งไว้แล้ว
goto MAKE_VENV

:PY_INSTALL
echo       ไม่พบ Python 3.11 - กำลังติดตั้งให้อัตโนมัติ
if not exist "setup\python-3.11.9-amd64.exe" goto PY_MISSING_FILE
echo       (ใช้เวลาประมาณ 1-3 นาที ห้ามปิดหน้าต่างนี้)
rem InstallAllUsers=0 = ลงเฉพาะผู้ใช้นี้ ไม่ต้องขอสิทธิ์แอดมิน ไม่เด้ง UAC
rem PrependPath=1 = ใส่ใน PATH ให้ (มีผลกับหน้าต่างที่เปิดใหม่เท่านั้น)
start /wait "" "setup\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_doc=0
rem PATH ของหน้าต่างนี้ถูกอ่านไปตั้งแต่ตอนเปิด จึงยังไม่เห็น Python ที่เพิ่งลง
rem ต้องเรียกจาก path ตรง ๆ ห้ามพึ่ง py/python ในหน้าต่างนี้
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" goto PY_INSTALL_OK
goto PY_INSTALL_FAIL

:PY_INSTALL_OK
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
echo       ติดตั้ง Python 3.11.9 เรียบร้อย
goto MAKE_VENV

:PY_MISSING_FILE
echo.
echo [ผิดพลาด] ไม่พบไฟล์ setup\python-3.11.9-amd64.exe
echo ไฟล์ ZIP แตกไม่ครบ - กรุณาแตกไฟล์ ZIP ใหม่ทั้งหมดอีกครั้ง
echo.
pause
exit /b 1

:PY_INSTALL_FAIL
echo.
echo [ผิดพลาด] ติดตั้ง Python อัตโนมัติไม่สำเร็จ
echo.
echo กรุณาติดตั้งเอง แล้วรัน INSTALL.bat ใหม่:
echo   ดับเบิลคลิก setup\python-3.11.9-amd64.exe
echo   ** ตอนติดตั้งต้องติ๊กช่อง "Add python.exe to PATH" **
echo.
pause
exit /b 1

rem ── [2/7] virtual environment ───────────────────────────────────────────
:MAKE_VENV
echo [2/7] เตรียม virtual environment (.venv) ...
if exist ".venv\Scripts\python.exe" goto VENV_READY
"%PY311%" -m venv .venv
if errorlevel 1 goto VENV_FAIL
echo       สร้าง .venv เรียบร้อย
goto PIP_UPGRADE

:VENV_FAIL
echo.
echo [ผิดพลาด] สร้าง .venv ไม่สำเร็จ
echo อาจเกิดจากพื้นที่ดิสก์ไม่พอ หรือโฟลเดอร์นี้ไม่มีสิทธิ์เขียน
echo ลองย้ายโฟลเดอร์ ZENTRA ไปไว้ที่ C:\ZENTRA แล้วรันใหม่
echo.
pause
exit /b 1

:VENV_READY
echo       ใช้ .venv เดิมที่มีอยู่แล้ว

:PIP_UPGRADE
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip -q

rem ── [3/7] WebView2 - ตัวแสดงผลหน้าต่างโปรแกรม ───────────────────────────
echo [3/7] ตรวจหา WebView2 (ตัวแสดงผลหน้าจอ) ...
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1
if not errorlevel 1 goto WV_OK
reg query "HKCU\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1
if not errorlevel 1 goto WV_OK
echo       ไม่พบ WebView2 - กำลังติดตั้งให้ (ถ้าไม่มีตัวนี้ โปรแกรมจะขึ้นจอดำ)
if not exist "setup\MicrosoftEdgeWebview2Setup.exe" goto WV_SKIP
start /wait "" "setup\MicrosoftEdgeWebview2Setup.exe" /silent /install
if errorlevel 1 goto WV_SKIP
echo       ติดตั้ง WebView2 เรียบร้อย
goto SETUP_ENV

:WV_OK
echo       พบ WebView2 อยู่แล้ว
goto SETUP_ENV

:WV_SKIP
echo       [เตือน] ติดตั้ง WebView2 ไม่สำเร็จ - ติดตั้งต่อไปก่อน
echo       ถ้าเปิดโปรแกรมแล้วเจอจอดำ ให้ดับเบิลคลิก setup\MicrosoftEdgeWebview2Setup.exe เอง

rem ── [4/7] ตั้งค่า .env + ตรวจไฟล์โมเดล ──────────────────────────────────
:SETUP_ENV
echo [4/7] ตั้งค่าไฟล์ config และตรวจไฟล์โมเดล ...
if exist "backend\.env" goto ENV_OK
copy "backend\.env.example" "backend\.env" >nul
echo       สร้าง backend\.env จากค่าเริ่มต้น
goto CHECK_MODELS

:ENV_OK
echo       พบ backend\.env เดิมอยู่แล้ว - ไม่แก้ไข (ค่าที่ตั้งไว้ยังอยู่ครบ)

:CHECK_MODELS
for %%W in (ppe_finetuned.pt yolo11m.pt yolo11n-pose.pt) do (
  if not exist "backend\models\%%W" goto MODEL_MISSING
)
if not exist "backend\assets\models\fall_detection_transformer.tflite" goto MODEL_MISSING
echo       ไฟล์โมเดลครบถ้วน
goto GPU_CHECK

:MODEL_MISSING
echo.
echo [ผิดพลาด] ไฟล์โมเดล AI ไม่ครบ
echo ไฟล์ ZIP อาจแตกไม่สมบูรณ์ กรุณาแตกไฟล์ ZIP ใหม่ทั้งหมดอีกครั้ง
echo (ถ้าไม่มีไฟล์โมเดล โปรแกรมจะเปิดได้แต่จะขึ้นว่า "ระบบ AI ไม่ทำงาน")
echo.
pause
exit /b 1

rem ── [5/7] ตรวจหาการ์ดจอ NVIDIA ──────────────────────────────────────────
:GPU_CHECK
echo [5/7] ตรวจหาการ์ดจอ NVIDIA ...
where nvidia-smi >nul 2>&1
if errorlevel 1 goto CPU_TORCH
nvidia-smi >nul 2>&1
if errorlevel 1 goto CPU_TORCH

echo       พบการ์ดจอ NVIDIA - ติดตั้ง PyTorch แบบ CUDA 12.6
echo       (ไฟล์ใหญ่ประมาณ 2.5GB ใช้เวลานานหน่อย แต่จะเร็วกว่ามากตอนใช้งาน)
rem หมายเหตุ: ใช้ --index-url ไม่ใช่ --extra-index-url เพราะ extra-index จะทำให้
rem pip แอบเลือก wheel แบบ CPU จาก PyPI แทน (กับดักคลาสสิกของ torch)
"%PY%" -m pip install torch==2.12.1+cu126 torchvision==0.27.1+cu126 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto TORCH_FAIL
goto INSTALL_REST

:CPU_TORCH
echo       ไม่พบการ์ดจอ NVIDIA - ติดตั้ง PyTorch แบบ CPU
echo       (ใช้งานได้ปกติ แต่ประมวลผลช้ากว่าเครื่องที่มีการ์ดจอ)
"%PY%" -m pip install torch==2.12.1 torchvision==0.27.1
if errorlevel 1 goto TORCH_FAIL
goto INSTALL_REST

:TORCH_FAIL
echo.
echo [ผิดพลาด] ติดตั้ง PyTorch ไม่สำเร็จ
echo มักเกิดจากอินเทอร์เน็ตหลุดกลางทาง - ลองรัน INSTALL.bat ใหม่อีกครั้ง
echo (ของที่โหลดไปแล้วจะไม่โหลดซ้ำ)
echo.
pause
exit /b 1

rem ── [6/7] ติดตั้งไลบรารีที่เหลือ ────────────────────────────────────────
:INSTALL_REST
echo [6/7] ติดตั้งไลบรารีที่เหลือ ...
rem ห้ามใส่ --upgrade: torch ที่เพิ่งลง (2.12.1+cu126) ตรงกับ pin `==2.12.1`
rem ตาม PEP 440 อยู่แล้ว pip จะข้ามไป แต่ --upgrade จะไปดึงตัว CPU มาทับ
"%PY%" -m pip install -r requirements-win.txt
if errorlevel 1 goto DEPS_FAIL
goto SMOKE

:DEPS_FAIL
echo.
echo [ผิดพลาด] ติดตั้งไลบรารีไม่สำเร็จ
echo ลองรัน INSTALL.bat ใหม่อีกครั้ง (อาจเป็นเพราะอินเทอร์เน็ตหลุด)
echo.
pause
exit /b 1

rem ── [7/7] ทดสอบระบบ ─────────────────────────────────────────────────────
:SMOKE
echo [7/7] ทดสอบระบบ - โหลดโมเดลจริง อาจใช้เวลาสักครู่ ...
"%PY%" packaging_smoke.py
if errorlevel 1 goto SMOKE_FAIL

echo.
echo ==========================================
echo    ติดตั้งสำเร็จ!
echo ==========================================
echo.
echo เปิดโปรแกรมด้วยการดับเบิลคลิก  ZENTRA.bat
echo คู่มือการใช้งาน  MANUAL_TH.html  (ดับเบิลคลิกเปิดในเบราว์เซอร์)
echo.
echo หมายเหตุ: เข้าโปรแกรมครั้งแรกจะยังไม่มีโมเดลใดทำงาน
echo ให้ไปที่หน้า "กล้อง" เพื่อเลือกโมเดลที่ต้องการใช้ก่อน
echo.
pause
exit /b 0

:SMOKE_FAIL
echo.
echo [ผิดพลาด] ทดสอบระบบไม่ผ่าน - ดูข้อความด้านบนประกอบ
echo กรุณาถ่ายภาพหน้าจอนี้ส่งให้ผู้ดูแลระบบ
echo.
pause
exit /b 1
