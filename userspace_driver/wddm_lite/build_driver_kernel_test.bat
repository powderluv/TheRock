@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
if "%ROCR%"=="" set ROCR=Z:\rocm-systems-macos-os-darwin\projects\rocr-runtime\runtime\hsa-runtime
set HSAKMTINC=Z:\rocr-build-canon\install\include
cd /d "%~dp0"
cl.exe /nologo /EHsc /W3 /O2 /std:c++20 /DWIN32 /D_WINDOWS /DNOMINMAX ^
  /I. /I"%ROCR%" /I"%ROCR%\inc" /I"%HSAKMTINC%" ^
  windows_lite_driver_kernel_test.cpp ^
  "%ROCR%\core\driver\lite\windows\amd_windows_lite_driver.cpp" ^
  "%ROCR%\core\driver\lite\amd_lite_direct_queue.cpp" ^
  "%ROCR%\core\driver\driver.cpp" ^
  wddm_lite.cpp gpu_init.cpp ^
  windows_lite_test_stubs.cpp ^
  /Fe:windows_lite_driver_kernel_test.exe ^
  /link gdi32.lib user32.lib
echo DRVKERN_BUILD_RC=%errorlevel%
