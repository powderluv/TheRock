@echo off
chcp 65001 >nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul
set "PATH=B:\tools\git\cmd;B:\tools\git\usr\bin;Z:\win-tools\cmake\cmake-3.30.5-windows-x86_64\bin;Z:\win-tools\ninja;C:\Program Files\Python312;C:\Program Files\Python312\Scripts;%PATH%;B:\tools\strawberry\perl\bin;B:\tools\strawberry\c\bin"
set "GIT_CONFIG_COUNT=2"
set "GIT_CONFIG_KEY_0=safe.directory"
set "GIT_CONFIG_VALUE_0=*"
set "GIT_CONFIG_KEY_1=core.autocrlf"
set "GIT_CONFIG_VALUE_1=false"
