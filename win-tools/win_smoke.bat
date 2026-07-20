@echo off
if exist B:\logs\win_smoke.done del /q B:\logs\win_smoke.done
call B:\tvenv\Scripts\activate.bat
python -m pip install -q pytest expecttest 1>B:\logs\win_smoke_pip.log 2>&1
if not exist C:\smoke mkdir C:\smoke
copy /Y Z:\win-tools\tri_os_smoke.py C:\smoke\ >/dev/null
copy /Y Z:\external-builds\pytorch\smoke-tests\pytorch_smoke_test.py C:\smoke\ >/dev/null
for /f "delims=" %%i in ('python -m rocm_sdk path --root') do set "RROOT=%%i"
set "PATH=%RROOT%\bin;%PATH%"
set ROCR_WINDOWS_FORCE_DIRECT_COMPUTE=1
set ROCR_AMDGPU_LITE_HOST_BLIT_ONLY=1
set ROCR_LITE_DEVICE_ONLY_SKIP_MEMSET=1
set ROCR_WINDOWS_USE_MES_QUEUE=1
set ROCR_WINDOWS_MES_MMIO_WPTR=1
set ROCR_WINDOWS_MES_WPTR_POLL=1
set ROCR_WINDOWS_MES_ACTIVATE_SCHED_HQD=1
set ROCR_WINDOWS_SHMEM_APERTURE=1
set ROCR_MACOS_AQL_ENABLE_SCRATCH=1
set ROCR_WINDOWS_TRACE_DIRECT_QUEUE=1
set ROCR_MACOS_TRACE_DMA_COPY=1
rem SMOKE_ISOLATE now OPT-IN (single-process is the true torch mode; isolate=1 fails after subprocess 1 because Windows MES bring-up is one-shot per power cycle). Set SMOKE_ISOLATE=1 externally to force per-test isolation.
set SMOKE_TEST_TIMEOUT=150
set SMOKE_FILE=C:\smoke\pytorch_smoke_test.py
set SMOKE_JUNIT=B:\logs\smoke_junit.xml
cd /d C:\smoke
python -u C:\smoke\tri_os_smoke.py 1>B:\logs\win_smoke.log 2>&1
echo WIN_SMOKE_RC=%errorlevel% > B:\logs\win_smoke.done
