@echo off
rem ===========================================================================
rem  ZENTRA Safety AI - ตัวติดตั้ง (Windows 10/11 x64)
rem  รันซ้ำได้เสมอ ปลอดภัย ไม่ทับ backend\.env ที่ผู้ใช้แก้ไว้
rem
rem  ปกติผู้ใช้ไม่ต้องเรียกไฟล์นี้เอง - ZENTRA.bat จะเรียกให้อัตโนมัติตอนเปิด
rem  ครั้งแรก ไฟล์นี้เก็บไว้สำหรับ "ซ่อมระบบ" (รันซ้ำเมื่อติดตั้งค้างกลางทาง)
rem
rem  หมายเหตุสำหรับผู้พัฒนา:
rem  - .gitattributes บังคับให้ไฟล์นี้เป็น CRLF เสมอ ไม่ว่าผู้ใช้จะโหลดมาทาง
rem    git clone หรือปุ่ม Download ZIP ของ GitHub
rem  - ห้ามอ่านตัวแปรที่ set ไว้ใน if(...) block เดียวกัน - ใช้ goto แทน
rem  - Python/WebView2: ถ้ามีไฟล์ใน setup\ จะใช้ตัวนั้น (ติดตั้งออฟไลน์ได้)
rem    ถ้าไม่มี (เช่นโหลดมาจาก GitHub ซึ่ง .gitignore ตัด setup\*.exe ออก)
rem    จะดาวน์โหลดจากต้นทางทางการให้เอง ด้วย curl.exe ที่มากับ Windows
rem  - ติดตั้ง Python แบบ per-user (InstallAllUsers=0) เพื่อไม่ให้เด้ง UAC
rem  - เรียกด้วย  INSTALL.bat /auto  เพื่อข้าม pause ตอนจบ (ZENTRA.bat ใช้)
rem ===========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "AUTOMODE="
if /i "%~1"=="/auto" set "AUTOMODE=1"

set "PY_VER=3.11.9"
set "PY_EXE=setup\python-%PY_VER%-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.exe"
set "WV_EXE=setup\MicrosoftEdgeWebview2Setup.exe"
set "WV_URL=https://go.microsoft.com/fwlink/p/?LinkId=2124703"

echo.
echo ==========================================
echo    ZENTRA Safety AI - ติดตั้งระบบ
echo ==========================================
echo.
echo ขั้นตอนนี้ใช้อินเทอร์เน็ต และใช้เวลาประมาณ 10-20 นาที
echo (ดาวน์โหลดไลบรารีประมาณ 500MB - 2.5GB ขึ้นกับว่ามีการ์ดจอ NVIDIA หรือไม่)
echo ระหว่างนี้จะเห็นข้อความวิ่งเยอะ ๆ เป็นเรื่องปกติ
echo.

rem ── [1/8] ตรวจว่าโฟลเดอร์นี้อยู่ลึกเกินไปหรือไม่ ─────────────────────────
rem   ไลบรารี torch มีไฟล์ header ที่ลึกถึง 145 ตัวอักษรนับจากโฟลเดอร์นี้
rem   ส่วน Windows จำกัดความยาวพาธไว้ที่ 260 ถ้าโฟลเดอร์นี้อยู่ลึกเกินไป
rem   pip จะแตกไฟล์ไม่ออกและพังกลางทางแบบงง ๆ
rem   แทนที่จะเดาเอาจากการนับตัวอักษร เราลอง "สร้างไฟล์ที่ลึกเท่าของจริง"
rem   ดูตรง ๆ เลย - ได้คำตอบที่ตรงกับความจริงของเครื่องนั้น ๆ
echo [1/8] ตรวจความยาวของพาธโฟลเดอร์ ...
set "PROBE=.pchk\Lib\site-packages\torch\include\ATen\native\transformers\cuda\mem_eff_attention\iterators"
if exist ".pchk" rmdir /s /q ".pchk"
mkdir "%PROBE%" 2>nul
echo probe> "%PROBE%\predicated_tile_access_iterator_residual_last.h" 2>nul
if not exist "%PROBE%\predicated_tile_access_iterator_residual_last.h" goto PATH_TOO_LONG
rmdir /s /q ".pchk"
echo       ความยาวพาธใช้ได้
goto PY_CHECK

:PATH_TOO_LONG
if exist ".pchk" rmdir /s /q ".pchk"
echo.
echo [ผิดพลาด] โฟลเดอร์นี้อยู่ลึกเกินไปสำหรับ Windows
echo.
echo ตอนนี้อยู่ที่:
echo   %CD%
echo.
echo วิธีแก้: ตัด (Ctrl+X) โฟลเดอร์นี้ทั้งโฟลเดอร์ไปวางไว้ที่ไดรฟ์ C: ตรง ๆ
echo   ให้ได้หน้าตาแบบนี้ ->  C:\ZENTRA
echo แล้วดับเบิลคลิก ZENTRA.bat ใหม่อีกครั้ง
echo.
echo (สาเหตุ: Windows จำกัดความยาวชื่อพาธไว้ที่ 260 ตัวอักษร
echo  ไลบรารี AI มีไฟล์ที่ชื่อยาวมาก จึงต้องเหลือที่ว่างให้พอ)
echo.
pause
exit /b 1

rem ── [2/8] Python 3.11 - ติดตั้งให้เองถ้าไม่มี ────────────────────────────
:PY_CHECK
echo [2/8] ตรวจหา Python 3.11 ...
set "PY311="
py -3.11 -V >nul 2>&1
if not errorlevel 1 goto PY_FOUND_LAUNCHER
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" goto PY_FOUND_LOCAL
goto PY_NEED_INSTALL

:PY_FOUND_LAUNCHER
rem ต้องแปลง "py -3.11" ให้เป็นพาธ python.exe จริงก่อน
rem ห้ามเก็บเป็น "py -3.11" ตรง ๆ เพราะเวลาเรียกต้องครอบ quote กันพาธมีเว้นวรรค
rem แล้ว "py -3.11" ที่อยู่ใน quote จะกลายเป็นชื่อโปรแกรมเดียว -> CMD หาไม่เจอ
rem (บั๊กนี้ทำให้ขั้น [3/8] ล้มบนทุกเครื่องที่มี Python launcher ติดอยู่)
set "PY311="
for /f "delims=" %%p in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "PY311=%%p"
if not defined PY311 goto PY_NEED_INSTALL
if not exist "%PY311%" goto PY_NEED_INSTALL
for /f "tokens=*" %%v in ('py -3.11 -V 2^>^&1') do echo       พบ %%v
goto MAKE_VENV

:PY_FOUND_LOCAL
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
echo       พบ Python 3.11 ติดตั้งไว้แล้ว
goto MAKE_VENV

:PY_NEED_INSTALL
echo       ไม่พบ Python 3.11 - กำลังจัดการให้อัตโนมัติ
if exist "%PY_EXE%" goto PY_RUN_INSTALLER

rem ไม่มีตัวติดตั้งในเครื่อง (ปกติคือโหลดโปรเจกต์มาจาก GitHub ซึ่งไม่แถม
rem ไฟล์ .exe มาด้วยเพราะใหญ่) -> โหลดจาก python.org ให้เลย
echo       ดาวน์โหลด Python %PY_VER% จาก python.org (~25 MB)
if not exist "setup" mkdir "setup"
curl.exe -L --fail --retry 2 --retry-delay 3 --connect-timeout 30 -o "%PY_EXE%" "%PY_URL%"
if errorlevel 1 goto PY_DOWNLOAD_FAIL
if not exist "%PY_EXE%" goto PY_DOWNLOAD_FAIL
rem กันไฟล์ที่โหลดมาไม่ครบ / ได้หน้า error เป็น HTML แทน (ตัวจริง ~25 MB)
for %%A in ("%PY_EXE%") do if %%~zA LSS 20000000 goto PY_DOWNLOAD_BAD
echo       ดาวน์โหลดเรียบร้อย

:PY_RUN_INSTALLER
echo       กำลังติดตั้ง Python (1-3 นาที ห้ามปิดหน้าต่างนี้)
rem InstallAllUsers=0 = ลงเฉพาะผู้ใช้นี้ ไม่ต้องขอสิทธิ์แอดมิน ไม่เด้ง UAC
rem PrependPath=1 = ใส่ใน PATH ให้ (มีผลกับหน้าต่างที่เปิดใหม่เท่านั้น)
start /wait "" "%PY_EXE%" /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_doc=0
rem PATH ของหน้าต่างนี้ถูกอ่านไปตั้งแต่ตอนเปิด จึงยังไม่เห็น Python ที่เพิ่งลง
rem ต้องเรียกจาก path ตรง ๆ ห้ามพึ่ง py/python ในหน้าต่างนี้
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" goto PY_INSTALL_OK
goto PY_INSTALL_FAIL

:PY_INSTALL_OK
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
echo       ติดตั้ง Python %PY_VER% เรียบร้อย
goto MAKE_VENV

:PY_DOWNLOAD_FAIL
echo.
echo [ผิดพลาด] ดาวน์โหลด Python ไม่สำเร็จ
echo.
echo มักเกิดจากอินเทอร์เน็ตหลุด หรือไฟร์วอลล์ขององค์กรบล็อก python.org
echo.
echo ทางแก้ที่ 1: เช็คอินเทอร์เน็ตแล้วดับเบิลคลิก ZENTRA.bat ใหม่
echo ทางแก้ที่ 2: โหลดเองจาก  %PY_URL%
echo              แล้วดับเบิลคลิกติดตั้ง ตอนติดตั้งต้องติ๊กช่อง
echo              "Add python.exe to PATH" จากนั้นเปิด ZENTRA.bat ใหม่
echo.
pause
exit /b 1

:PY_DOWNLOAD_BAD
del /q "%PY_EXE%" 2>nul
echo.
echo [ผิดพลาด] ไฟล์ Python ที่ดาวน์โหลดมาไม่สมบูรณ์ (ขนาดเล็กผิดปกติ)
echo มักเกิดจากเครือข่ายมีหน้า login ดักไว้ (Wi-Fi โรงแรม/สนามบิน/องค์กร)
echo ลองเชื่อมเน็ตเส้นอื่นแล้วดับเบิลคลิก ZENTRA.bat ใหม่
echo.
pause
exit /b 1

:PY_INSTALL_FAIL
echo.
echo [ผิดพลาด] ติดตั้ง Python อัตโนมัติไม่สำเร็จ
echo.
echo กรุณาติดตั้งเอง แล้วเปิด ZENTRA.bat ใหม่:
echo   ดับเบิลคลิก %PY_EXE%
echo   ** ตอนติดตั้งต้องติ๊กช่อง "Add python.exe to PATH" **
echo.
pause
exit /b 1

rem ── [3/8] virtual environment ───────────────────────────────────────────
:MAKE_VENV
echo [3/8] เตรียม virtual environment (.venv) ...
if exist ".venv\Scripts\python.exe" goto VENV_READY
"%PY311%" -m venv .venv
if errorlevel 1 goto VENV_FAIL
if not exist ".venv\Scripts\python.exe" goto VENV_FAIL
echo       สร้าง .venv เรียบร้อย
goto PIP_UPGRADE

:VENV_FAIL
echo.
echo [ผิดพลาด] สร้าง .venv ไม่สำเร็จ
echo อาจเกิดจากพื้นที่ดิสก์ไม่พอ (ต้องว่างอย่างน้อย 8 GB)
echo หรือโฟลเดอร์นี้ไม่มีสิทธิ์เขียน
echo ลองย้ายโฟลเดอร์ ZENTRA ไปไว้ที่ C:\ZENTRA แล้วเปิด ZENTRA.bat ใหม่
echo.
pause
exit /b 1

:VENV_READY
echo       ใช้ .venv เดิมที่มีอยู่แล้ว

:PIP_UPGRADE
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip -q

rem ── [4/8] WebView2 - ตัวแสดงผลหน้าต่างโปรแกรม ───────────────────────────
echo [4/8] ตรวจหา WebView2 (ตัวแสดงผลหน้าจอ) ...
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1
if not errorlevel 1 goto WV_OK
reg query "HKCU\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1
if not errorlevel 1 goto WV_OK
echo       ไม่พบ WebView2 - กำลังติดตั้งให้ (ถ้าไม่มีตัวนี้ โปรแกรมจะขึ้นจอดำ)
if exist "%WV_EXE%" goto WV_RUN
echo       ดาวน์โหลดตัวติดตั้ง WebView2 จาก Microsoft (~2 MB)
if not exist "setup" mkdir "setup"
curl.exe -L --fail --retry 2 --retry-delay 3 --connect-timeout 30 -o "%WV_EXE%" "%WV_URL%"
if errorlevel 1 goto WV_SKIP
if not exist "%WV_EXE%" goto WV_SKIP

:WV_RUN
start /wait "" "%WV_EXE%" /silent /install
if errorlevel 1 goto WV_SKIP
echo       ติดตั้ง WebView2 เรียบร้อย
goto SETUP_ENV

:WV_OK
echo       พบ WebView2 อยู่แล้ว
goto SETUP_ENV

:WV_SKIP
echo       [เตือน] ติดตั้ง WebView2 ไม่สำเร็จ - ติดตั้งต่อไปก่อน
echo       Windows 11 ส่วนใหญ่มีตัวนี้ติดมาแล้ว ถ้าเปิดโปรแกรมแล้วเจอจอดำ
echo       ให้โหลดเองจาก https://developer.microsoft.com/microsoft-edge/webview2/

rem ── [5/8] ตั้งค่า .env + ตรวจไฟล์โมเดล ──────────────────────────────────
:SETUP_ENV
echo [5/8] ตั้งค่าไฟล์ config และตรวจไฟล์โมเดล ...
if exist "backend\.env" goto ENV_OK
if not exist "backend\.env.example" goto ENV_NO_TEMPLATE
copy "backend\.env.example" "backend\.env" >nul
echo       สร้าง backend\.env จากค่าเริ่มต้น
goto CHECK_MODELS

:ENV_NO_TEMPLATE
rem ไม่ใช่เรื่องคอขาดบาดตาย: config.py มีค่าเริ่มต้นครบอยู่แล้วในตัว
echo       [เตือน] ไม่พบ backend\.env.example - ใช้ค่าเริ่มต้นในโปรแกรมแทน
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
echo.
echo ถ้าโหลดมาจาก GitHub ด้วยปุ่ม "Download ZIP":
echo   ให้แตกไฟล์ ZIP ใหม่ทั้งหมดอีกครั้ง (คลิกขวา -^> Extract All)
echo   ห้ามเปิดดูในหน้าต่าง ZIP แล้วลากไฟล์ออกมาทีละอัน - จะได้ไม่ครบ
echo.
echo ไฟล์ที่ต้องมี:
echo   backend\models\ppe_finetuned.pt      (39 MB)
echo   backend\models\yolo11m.pt            (39 MB)
echo   backend\models\yolo11n-pose.pt       ( 6 MB)
echo   backend\assets\models\fall_detection_transformer.tflite
echo.
pause
exit /b 1

rem ── [6/8] ตรวจหาการ์ดจอ NVIDIA ──────────────────────────────────────────
:GPU_CHECK
echo [6/8] ตรวจหาการ์ดจอ NVIDIA ...
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
echo มักเกิดจากอินเทอร์เน็ตหลุดกลางทาง - ลองเปิด ZENTRA.bat ใหม่อีกครั้ง
echo (ของที่โหลดไปแล้วจะไม่โหลดซ้ำ)
echo.
pause
exit /b 1

rem ── [7/8] ติดตั้งไลบรารีที่เหลือ ────────────────────────────────────────
:INSTALL_REST
echo [7/8] ติดตั้งไลบรารีที่เหลือ ...
rem ห้ามใส่ --upgrade: torch ที่เพิ่งลง (2.12.1+cu126) ตรงกับ pin `==2.12.1`
rem ตาม PEP 440 อยู่แล้ว pip จะข้ามไป แต่ --upgrade จะไปดึงตัว CPU มาทับ
"%PY%" -m pip install -r requirements-win.txt
if errorlevel 1 goto DEPS_FAIL
goto SMOKE

:DEPS_FAIL
echo.
echo [ผิดพลาด] ติดตั้งไลบรารีไม่สำเร็จ
echo ลองเปิด ZENTRA.bat ใหม่อีกครั้ง (อาจเป็นเพราะอินเทอร์เน็ตหลุด)
echo.
pause
exit /b 1

rem ── [8/8] ทดสอบระบบ ─────────────────────────────────────────────────────
:SMOKE
echo [8/8] ทดสอบระบบ - โหลดโมเดลจริง อาจใช้เวลาสักครู่ ...
"%PY%" packaging_smoke.py
if errorlevel 1 goto SMOKE_FAIL

echo.
echo ==========================================
echo    ติดตั้งสำเร็จ!
echo ==========================================
echo.
echo หมายเหตุ: เข้าโปรแกรมครั้งแรกจะยังไม่มีโมเดลใดทำงาน
echo ให้ไปที่หน้า "กล้อง" เพื่อเลือกโมเดลที่ต้องการใช้ก่อน
echo คู่มือการใช้งาน  MANUAL_TH.html  (ดับเบิลคลิกเปิดในเบราว์เซอร์)
echo.
if defined AUTOMODE exit /b 0
echo เปิดโปรแกรมด้วยการดับเบิลคลิก  ZENTRA.bat
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
