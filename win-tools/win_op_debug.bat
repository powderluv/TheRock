@echo off
call B:\tvenv\Scripts\activate.bat
if not exist C:\smoke mkdir C:\smoke
copy /Y Z:\win-tools\win_op_debug.py C:\smoke\ >nul
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
set ROCR_LITE_SKIP_POST_BRINGUP_MEMSET=1
cd /d C:\smoke
python -u C:\smoke\win_op_debug.py 1>C:\smoke\win_op_debug.log 2>&1
echo WIN_OP_RC=%errorlevel%>>C:\smoke\win_op_debug.log
