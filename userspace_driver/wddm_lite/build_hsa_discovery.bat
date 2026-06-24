@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
set ROCR=Z:\rocm-systems-macos-os-darwin\projects\rocr-runtime\runtime\hsa-runtime
set DLL=Z:\rocr-build-canon\hsa-runtime
cd /d "%~dp0"
cl.exe /nologo /EHsc /W3 /O2 /std:c++20 /DWIN32 /D_WINDOWS /DNOMINMAX /I"%ROCR%\inc" ^
  hsa_discovery_test.cpp /Fe:%DLL%\hsa_discovery_test.exe ^
  /link /LIBPATH:"%DLL%" hsa-runtime64.lib
echo DISC_BUILD_RC=%errorlevel%
