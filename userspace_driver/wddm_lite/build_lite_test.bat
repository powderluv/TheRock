@echo off
REM build_lite_test.bat - MSVC build for the ROCr lite:: direct-queue NOP-fence
REM harness over wddm_lite (lite_direct_queue_test.exe).
REM
REM Builds:
REM   lite_direct_queue_test.cpp   (the harness + WddmLiteDirectPlatform)
REM   amd_lite_direct_queue.cpp    (SHARED ROCr lite:: queue logic, from the
REM                                 winscaffold rocr-runtime tree)
REM   wddm_lite.cpp + gpu_init.cpp  (the proven wddm_lite driver)
REM
REM Adjust ROCR to the rocr-runtime hsa-runtime path as seen by the guest.
setlocal
set VSTOOLS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if exist %VSTOOLS% ( call %VSTOOLS% amd64 ) else ( echo ERROR: VS Build Tools not found & exit /b 1 )
cd /d "%~dp0"
set WDK_INC=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set WDK_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0

REM Path to the winscaffold rocr-runtime hsa-runtime root (contains inc\hsa.h and
REM core\inc\amd_lite_direct_queue.h + core\driver\lite\amd_lite_direct_queue.cpp).
REM Override by setting ROCR before invoking this script.
if "%ROCR%"=="" set ROCR=Z:\rocr-runtime\runtime\hsa-runtime

echo Building lite_direct_queue_test.exe...
echo   ROCR=%ROCR%
cl.exe /nologo /EHsc /W3 /O2 /std:c++17 ^
  /DWIN32_LEAN_AND_MEAN /DNOMINMAX ^
  /I"%WDK_INC%\shared" /I"%WDK_INC%\um" ^
  /I. /I"%ROCR%" /I"%ROCR%\inc" ^
  lite_direct_queue_test.cpp ^
  "%ROCR%\core\driver\lite\amd_lite_direct_queue.cpp" ^
  wddm_lite.cpp gpu_init.cpp ^
  /Fe:lite_direct_queue_test.exe ^
  /link /LIBPATH:"%WDK_LIB%\um\x64" gdi32.lib user32.lib
if %ERRORLEVEL% NEQ 0 ( echo BUILD FAILED & exit /b 1 )
echo BUILD SUCCEEDED
