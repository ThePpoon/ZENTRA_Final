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
rem  - Python/WebView2/VC++: ถ้ามีไฟล์ใน setup\ จะใช้ตัวนั้น (ออฟไลน์ได้)
rem    ถ้าไม่มี (เช่นโหลดมาจาก GitHub ซึ่ง .gitignore ตัด setup\*.exe ออก)
rem    จะดาวน์โหลดจากต้นทางทางการให้เอง ด้วย curl.exe ที่มากับ Windows
rem    แล้วตรวจ "ลายเซ็นดิจิทัล" ทุกไฟล์ก่อนรัน - จับไฟล์ที่โหลดมาไม่ครบได้
rem    แน่นอนกว่าการดูขนาดไฟล์
rem  - ติดตั้ง Python แบบ per-user (InstallAllUsers=0) เพื่อไม่ให้เด้ง UAC
rem    ข้อยกเว้นเดียวคือ VC++ Redistributable ซึ่งเป็นของทั้งเครื่อง จึงต้อง UAC
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
set "VC_EXE=setup\vc_redist.x64.exe"
set "VC_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
set "CURL=curl.exe -L --fail --retry 2 --retry-delay 3 --connect-timeout 30"

echo.
echo ==========================================
echo    ZENTRA Safety AI - ติดตั้งระบบ
echo ==========================================
echo.
echo ขั้นตอนนี้ใช้อินเทอร์เน็ต และใช้เวลาประมาณ 10-20 นาที
echo (ดาวน์โหลดไลบรารีประมาณ 500MB - 2.5GB ขึ้นกับว่ามีการ์ดจอ NVIDIA หรือไม่)
echo ระหว่างนี้จะเห็นข้อความวิ่งเยอะ ๆ เป็นเรื่องปกติ
echo.

rem ── [1/10] ตรวจว่าโฟลเดอร์นี้อยู่ลึกเกินไปหรือไม่ ────────────────────────
rem   ไลบรารี torch มีไฟล์ header ที่ลึกถึง 145 ตัวอักษรนับจากโฟลเดอร์นี้
rem   ส่วน Windows จำกัดความยาวพาธไว้ที่ 260 ถ้าโฟลเดอร์นี้อยู่ลึกเกินไป
rem   pip จะแตกไฟล์ไม่ออกและพังกลางทางแบบงง ๆ
rem   แทนที่จะเดาเอาจากการนับตัวอักษร เราลอง "สร้างไฟล์ที่ลึกเท่าของจริง"
rem   ดูตรง ๆ เลย - ได้คำตอบที่ตรงกับความจริงของเครื่องนั้น ๆ
echo [1/10] ตรวจความยาวของพาธโฟลเดอร์ ...
set "PROBE=.pchk\Lib\site-packages\torch\include\ATen\native\transformers\cuda\mem_eff_attention\iterators"
if exist ".pchk" rmdir /s /q ".pchk"
mkdir "%PROBE%" 2>nul
echo probe> "%PROBE%\predicated_tile_access_iterator_residual_last.h" 2>nul
if not exist "%PROBE%\predicated_tile_access_iterator_residual_last.h" goto PATH_TOO_LONG
rmdir /s /q ".pchk"
echo        ความยาวพาธใช้ได้
goto DISK_CHECK

:PATH_TOO_LONG
if exist ".pchk" rmdir /s /q ".pchk"
echo.
echo [ผิดพลาด] โฟลเดอร์นี้อยู่ลึกเกินไปสำหรับ Windows
echo.
echo ตอนนี้อยู่ที่:
echo   %CD%
echo.
echo วิธีแก้: ตัด (Ctrl+X) โฟลเดอร์นี้ทั้งโฟลเดอร์ไปวางไว้ที่ไดรฟ์ C: ตรง ๆ
echo   ให้ได้หน้าตาแบบนี้ -^>  C:\ZENTRA
echo แล้วดับเบิลคลิก ZENTRA.bat ใหม่อีกครั้ง
echo.
echo (สาเหตุ: Windows จำกัดความยาวชื่อพาธไว้ที่ 260 ตัวอักษร
echo  ไลบรารี AI มีไฟล์ที่ชื่อยาวมาก จึงต้องเหลือที่ว่างให้พอ)
echo.
pause
exit /b 1

rem ── [2/10] พื้นที่ดิสก์ ──────────────────────────────────────────────────
rem   ดิสก์เต็มระหว่างติดตั้งคือหายนะเงียบ: ตัวติดตั้ง Python เขียนไฟล์ไม่ครบ
rem   แล้วไปพังตอน import ทีหลังเป็น SyntaxError ในไฟล์ของ Python เอง
rem   ซึ่งไม่มีทางเดาสาเหตุถูกเลย เช็คก่อนดีกว่า
:DISK_CHECK
echo [2/10] ตรวจพื้นที่ว่างในดิสก์ ...
set "FREEGB="
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "[int]([IO.DriveInfo]::new('%~d0\').AvailableFreeSpace/1GB)" 2^>nul`) do set "FREEGB=%%a"
if not defined FREEGB goto DISK_UNKNOWN
if %FREEGB% LSS 8 goto DISK_LOW
echo        เหลือ %FREEGB% GB - เพียงพอ
goto PY_CHECK

:DISK_UNKNOWN
echo        [เตือน] อ่านพื้นที่ว่างไม่ได้ - ข้ามการตรวจ
goto PY_CHECK

:DISK_LOW
echo.
echo [ผิดพลาด] พื้นที่ว่างในไดรฟ์ %~d0 เหลือแค่ %FREEGB% GB
echo.
echo ต้องการอย่างน้อย 8 GB (ไลบรารี AI อย่างเดียวก็ 2.5 GB แล้ว)
echo.
echo ถ้าฝืนติดตั้งทั้งที่ดิสก์ใกล้เต็ม ไฟล์จะถูกเขียนไม่ครบ
echo แล้วโปรแกรมจะพังแบบหาสาเหตุยากมาก จึงหยุดไว้ก่อนตรงนี้
echo.
echo วิธีแก้: ลบไฟล์ที่ไม่ใช้ออก หรือย้ายโฟลเดอร์ ZENTRA ไปไดรฟ์อื่นที่ว่างพอ
echo แล้วดับเบิลคลิก ZENTRA.bat ใหม่อีกครั้ง
echo.
pause
exit /b 1

rem ── [3/10] Python 3.11 - ติดตั้งให้เองถ้าไม่มี ───────────────────────────
:PY_CHECK
echo [3/10] ตรวจหา Python 3.11 ...
set "PY311="
py -3.11 -V >nul 2>&1
if not errorlevel 1 goto PY_FOUND_LAUNCHER
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" goto PY_FOUND_LOCAL
goto PY_NEED_INSTALL

:PY_FOUND_LAUNCHER
rem ต้องแปลง "py -3.11" ให้เป็นพาธ python.exe จริงก่อน
rem ห้ามเก็บเป็น "py -3.11" ตรง ๆ เพราะเวลาเรียกต้องครอบ quote กันพาธมีเว้นวรรค
rem แล้ว "py -3.11" ที่อยู่ใน quote จะกลายเป็นชื่อโปรแกรมเดียว -^> CMD หาไม่เจอ
rem (บั๊กนี้ทำให้ขั้น [4/10] ล้มบนทุกเครื่องที่มี Python launcher ติดอยู่)
set "PY311="
for /f "delims=" %%p in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "PY311=%%p"
if not defined PY311 goto PY_NEED_INSTALL
if not exist "%PY311%" goto PY_NEED_INSTALL
for /f "tokens=*" %%v in ('py -3.11 -V 2^>^&1') do echo        พบ %%v
goto PY_HEALTH

:PY_FOUND_LOCAL
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
echo        พบ Python 3.11 ติดตั้งไว้แล้ว
goto PY_HEALTH

rem   "python -V" เดินได้ ไม่ได้แปลว่า Python ใช้งานได้จริง
rem   ถ้าตอนติดตั้งเขียนไฟล์ไม่ครบ ไฟล์ในไลบรารีมาตรฐานจะขาดกลางคัน
rem   แล้วไปโผล่เป็น SyntaxError ตอนท้ายสุด ซึ่งไกลจากต้นเหตุมาก
rem   ลอง import ของจริงตรงนี้เลย จะได้เจอตั้งแต่ตอนที่ยังบอกวิธีแก้ได้
:PY_HEALTH
"%PY311%" -c "import wsgiref.handlers,ssl,sqlite3,ctypes,zlib,lzma,venv,encodings.idna" >nul 2>&1
if errorlevel 1 goto PY_BROKEN
echo        ตรวจไฟล์ภายใน Python - ครบถ้วน
goto MAKE_VENV

:PY_BROKEN
echo.
echo [ผิดพลาด] Python 3.11 ในเครื่องนี้ติดตั้งไม่สมบูรณ์
echo.
echo   %PY311%
echo.
echo เปิดโปรแกรมได้ แต่ไฟล์ข้างในขาดหายไปบางส่วน
echo (มักเกิดจากดิสก์เต็มตอนติดตั้ง หรือโปรแกรมป้องกันไวรัสขัดจังหวะ)
echo.
echo วิธีแก้:
echo   1. เปิด Settings -^> Apps -^> Installed apps
echo   2. หา "Python 3.11.9" -^> กดปุ่ม ... -^> Modify -^> Repair
echo      (ถ้าไม่มีตัวเลือก Repair ให้ Uninstall แล้วปล่อยให้ ZENTRA ลงใหม่)
echo   3. ลบโฟลเดอร์ .venv ในโฟลเดอร์นี้ทิ้ง ถ้ามี
echo   4. ดับเบิลคลิก ZENTRA.bat ใหม่อีกครั้ง
echo.
echo รายละเอียดข้อผิดพลาด:
"%PY311%" -c "import wsgiref.handlers,ssl,sqlite3,ctypes,zlib,lzma,venv,encodings.idna"
echo.
pause
exit /b 1

:PY_NEED_INSTALL
echo        ไม่พบ Python 3.11 - กำลังจัดการให้อัตโนมัติ
if exist "%PY_EXE%" goto PY_VERIFY

rem ไม่มีตัวติดตั้งในเครื่อง (ปกติคือโหลดโปรเจกต์มาจาก GitHub ซึ่งไม่แถม
rem ไฟล์ .exe มาด้วยเพราะใหญ่) -> โหลดจาก python.org ให้เลย
echo        ดาวน์โหลด Python %PY_VER% จาก python.org (~25 MB)
if not exist "setup" mkdir "setup"
%CURL% -o "%PY_EXE%" "%PY_URL%"
if errorlevel 1 goto PY_DOWNLOAD_FAIL
if not exist "%PY_EXE%" goto PY_DOWNLOAD_FAIL

rem ตรวจลายเซ็นดิจิทัล: จับได้ทั้งไฟล์โหลดไม่ครบ ไฟล์เสีย และไฟล์ปลอม
rem (เช็คขนาดไฟล์อย่างเดียวไม่พอ - หน้า login ของ Wi-Fi องค์กรก็ใหญ่ได้)
:PY_VERIFY
echo        ตรวจลายเซ็นดิจิทัลของไฟล์ติดตั้ง ...
powershell -NoProfile -Command "if ((Get-AuthenticodeSignature -LiteralPath '%PY_EXE%').Status -ne 'Valid') { exit 1 }" >nul 2>&1
if errorlevel 1 goto PY_DOWNLOAD_BAD
echo        ลายเซ็นถูกต้อง (Python Software Foundation)

echo        กำลังติดตั้ง Python (1-3 นาที ห้ามปิดหน้าต่างนี้)
rem InstallAllUsers=0 = ลงเฉพาะผู้ใช้นี้ ไม่ต้องขอสิทธิ์แอดมิน ไม่เด้ง UAC
rem PrependPath=1 = ใส่ใน PATH ให้ (มีผลกับหน้าต่างที่เปิดใหม่เท่านั้น)
start /wait "" "%PY_EXE%" /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_doc=0
rem PATH ของหน้าต่างนี้ถูกอ่านไปตั้งแต่ตอนเปิด จึงยังไม่เห็น Python ที่เพิ่งลง
rem ต้องเรียกจาก path ตรง ๆ ห้ามพึ่ง py/python ในหน้าต่างนี้
if not exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" goto PY_INSTALL_FAIL
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
echo        ติดตั้ง Python %PY_VER% เรียบร้อย
goto PY_HEALTH

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
echo [ผิดพลาด] ไฟล์ติดตั้ง Python ที่ได้มาไม่ถูกต้อง (ลายเซ็นไม่ผ่าน)
echo.
echo มักเกิดจากเครือข่ายมีหน้า login ดักไว้ (Wi-Fi โรงแรม/สนามบิน/องค์กร)
echo ทำให้สิ่งที่โหลดมาไม่ใช่ตัวติดตั้งจริง ผมลบไฟล์นั้นทิ้งให้แล้ว
echo.
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

rem ── [4/10] virtual environment ──────────────────────────────────────────
:MAKE_VENV
echo [4/10] เตรียม virtual environment (.venv) ...
if exist ".venv\Scripts\python.exe" goto VENV_READY
"%PY311%" -m venv .venv
if errorlevel 1 goto VENV_FAIL
if not exist ".venv\Scripts\python.exe" goto VENV_FAIL
echo        สร้าง .venv เรียบร้อย
goto PIP_UPGRADE

:VENV_FAIL
echo.
echo [ผิดพลาด] สร้าง .venv ไม่สำเร็จ
echo อาจเกิดจากโฟลเดอร์นี้ไม่มีสิทธิ์เขียน หรือถูกโปรแกรมป้องกันไวรัสกัน
echo ลองย้ายโฟลเดอร์ ZENTRA ไปไว้ที่ C:\ZENTRA แล้วเปิด ZENTRA.bat ใหม่
echo.
pause
exit /b 1

:VENV_READY
echo        ใช้ .venv เดิมที่มีอยู่แล้ว

:PIP_UPGRADE
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip -q

rem ── [5/10] Visual C++ Redistributable ───────────────────────────────────
rem   torch (c10.dll) และตัวตรวจจับการล้ม (LiteRT) ถูกคอมไพล์ด้วย MSVC จึงต้องใช้
rem   msvcp140.dll และ vcruntime140_1.dll ซึ่ง "ตัวติดตั้ง Python ไม่ได้ลงให้"
rem   (Python แถมมาแค่ vcruntime140.dll วางไว้ข้าง python.exe เท่านั้น)
rem   เครื่องที่ไม่เคยลงโปรแกรม C++ มาก่อนจะขาดสองตัวนี้ แล้วพังเป็น
rem   WinError 1114 ตอน import torch - ซึ่งอ่านแล้วเดาสาเหตุไม่ออกเลย
echo [5/10] ตรวจ Visual C++ Redistributable (ไลบรารีที่ AI ต้องใช้) ...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if not errorlevel 1 goto VC_OK
if not exist "%SystemRoot%\System32\msvcp140.dll" goto VC_NEED
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" goto VC_NEED
goto VC_OK

:VC_OK
echo        มีอยู่แล้ว
goto SETUP_WV

:VC_NEED
echo        ไม่พบ - ต้องติดตั้งก่อน ไม่งั้น AI จะทำงานไม่ได้
if exist "%VC_EXE%" goto VC_VERIFY
echo        ดาวน์โหลดจาก Microsoft (~25 MB)
if not exist "setup" mkdir "setup"
%CURL% -o "%VC_EXE%" "%VC_URL%"
if errorlevel 1 goto VC_FAIL
if not exist "%VC_EXE%" goto VC_FAIL

:VC_VERIFY
powershell -NoProfile -Command "if ((Get-AuthenticodeSignature -LiteralPath '%VC_EXE%').Status -ne 'Valid') { exit 1 }" >nul 2>&1
if errorlevel 1 goto VC_BADFILE
echo.
echo        ** ตัวนี้เป็นของทั้งเครื่อง Windows จะขึ้นหน้าต่างขออนุญาต **
echo        ** กรุณากด "Yes" เมื่อหน้าต่างเด้งขึ้นมา                  **
echo.
start /wait "" "%VC_EXE%" /install /quiet /norestart
set "VCRC=%ERRORLEVEL%"
if "%VCRC%"=="0"    goto VC_DONE
if "%VCRC%"=="1638" goto VC_DONE
if "%VCRC%"=="3010" goto VC_DONE
goto VC_FAIL

:VC_DONE
echo        ติดตั้ง Visual C++ Redistributable เรียบร้อย
goto SETUP_WV

:VC_BADFILE
del /q "%VC_EXE%" 2>nul
echo        [เตือน] ไฟล์ที่โหลดมาลายเซ็นไม่ผ่าน - ลบทิ้งแล้ว
goto VC_FAIL

:VC_FAIL
echo.
echo ============================================================
echo  [เตือนสำคัญ] ติดตั้ง Visual C++ Redistributable ไม่สำเร็จ
echo.
echo  ถ้าไม่มีตัวนี้ ระบบ AI จะทำงานไม่ได้เลย (ขึ้น error DLL)
echo  แต่จะติดตั้งส่วนที่เหลือต่อไปก่อน
echo.
echo  กรุณาติดตั้งเองหลังจากนี้:
echo    1. โหลดจาก %VC_URL%
echo    2. ดับเบิลคลิกติดตั้ง กด "Yes" ตอนขออนุญาต
echo    3. ดับเบิลคลิก ZENTRA.bat ใหม่อีกครั้ง
echo ============================================================
echo.
pause

rem ── [6/10] WebView2 - ตัวแสดงผลหน้าต่างโปรแกรม ──────────────────────────
:SETUP_WV
echo [6/10] ตรวจหา WebView2 (ตัวแสดงผลหน้าจอ) ...
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1
if not errorlevel 1 goto WV_OK
reg query "HKCU\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1
if not errorlevel 1 goto WV_OK
echo        ไม่พบ WebView2 - กำลังติดตั้งให้ (ถ้าไม่มีตัวนี้ โปรแกรมจะขึ้นจอดำ)
if exist "%WV_EXE%" goto WV_RUN
echo        ดาวน์โหลดตัวติดตั้ง WebView2 จาก Microsoft (~2 MB)
if not exist "setup" mkdir "setup"
%CURL% -o "%WV_EXE%" "%WV_URL%"
if errorlevel 1 goto WV_SKIP
if not exist "%WV_EXE%" goto WV_SKIP

:WV_RUN
start /wait "" "%WV_EXE%" /silent /install
if errorlevel 1 goto WV_SKIP
echo        ติดตั้ง WebView2 เรียบร้อย
goto SETUP_ENV

:WV_OK
echo        พบ WebView2 อยู่แล้ว
goto SETUP_ENV

:WV_SKIP
echo        [เตือน] ติดตั้ง WebView2 ไม่สำเร็จ - ติดตั้งต่อไปก่อน
echo        Windows 11 ส่วนใหญ่มีตัวนี้ติดมาแล้ว ถ้าเปิดโปรแกรมแล้วเจอจอดำ
echo        ให้โหลดเองจาก https://developer.microsoft.com/microsoft-edge/webview2/

rem ── [7/10] ตั้งค่า .env + ตรวจไฟล์โมเดล ─────────────────────────────────
:SETUP_ENV
echo [7/10] ตั้งค่าไฟล์ config และตรวจไฟล์โมเดล ...
if exist "backend\.env" goto ENV_OK
if not exist "backend\.env.example" goto ENV_NO_TEMPLATE
copy "backend\.env.example" "backend\.env" >nul
echo        สร้าง backend\.env จากค่าเริ่มต้น
goto CHECK_MODELS

:ENV_NO_TEMPLATE
rem ไม่ใช่เรื่องคอขาดบาดตาย: config.py มีค่าเริ่มต้นครบอยู่แล้วในตัว
echo        [เตือน] ไม่พบ backend\.env.example - ใช้ค่าเริ่มต้นในโปรแกรมแทน
goto CHECK_MODELS

:ENV_OK
echo        พบ backend\.env เดิมอยู่แล้ว - ไม่แก้ไข (ค่าที่ตั้งไว้ยังอยู่ครบ)

:CHECK_MODELS
for %%W in (ppe_finetuned.pt yolo11m.pt yolo11n-pose.pt) do (
  if not exist "backend\models\%%W" goto MODEL_MISSING
)
if not exist "backend\assets\models\fall_detection_transformer.tflite" goto MODEL_MISSING
echo        ไฟล์โมเดลครบถ้วน
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

rem ── [8/10] ตรวจหาการ์ดจอ NVIDIA + ติดตั้ง PyTorch ───────────────────────
:GPU_CHECK
echo [8/10] ตรวจหาการ์ดจอ NVIDIA ...
where nvidia-smi >nul 2>&1
if errorlevel 1 goto CPU_TORCH
nvidia-smi >nul 2>&1
if errorlevel 1 goto CPU_TORCH

echo        พบการ์ดจอ NVIDIA - ติดตั้ง PyTorch แบบ CUDA 12.6
echo        (ไฟล์ใหญ่ประมาณ 2.5GB ใช้เวลานานหน่อย แต่จะเร็วกว่ามากตอนใช้งาน)
rem หมายเหตุ: ใช้ --index-url ไม่ใช่ --extra-index-url เพราะ extra-index จะทำให้
rem pip แอบเลือก wheel แบบ CPU จาก PyPI แทน (กับดักคลาสสิกของ torch)
"%PY%" -m pip install torch==2.12.1+cu126 torchvision==0.27.1+cu126 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto TORCH_FAIL
goto INSTALL_REST

:CPU_TORCH
echo        ไม่พบการ์ดจอ NVIDIA - ติดตั้ง PyTorch แบบ CPU
echo        (ใช้งานได้ปกติ แต่ประมวลผลช้ากว่าเครื่องที่มีการ์ดจอ)
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

rem ── [9/10] ติดตั้งไลบรารีที่เหลือ ───────────────────────────────────────
:INSTALL_REST
echo [9/10] ติดตั้งไลบรารีที่เหลือ ...
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

rem ── [10/10] ทดสอบระบบ ───────────────────────────────────────────────────
:SMOKE
echo [10/10] ทดสอบระบบ - โหลดโมเดลจริง อาจใช้เวลาสักครู่ ...
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
echo ============================================================
echo  [ผิดพลาด] ทดสอบระบบไม่ผ่าน
echo.
echo  อ่าน "สาเหตุที่แท้จริง" ด้านบน - มีบอกวิธีแก้ไว้ทีละขั้นแล้ว
echo  แก้เสร็จแล้วดับเบิลคลิก ZENTRA.bat ใหม่อีกครั้ง
echo ============================================================
echo.
pause
exit /b 1
