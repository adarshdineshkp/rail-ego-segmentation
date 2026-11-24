@echo off
REM Pure model inference time on the GPU, for the trained checkpoint.
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)
set CKPT=runs\rs19ego_pidnet_s_b4\best.pt
if not "%~1"=="" set CKPT=%~1
if not exist logs mkdir logs
python tools\benchmark.py --ckpt "%CKPT%" > logs\benchmark.log 2>&1
type logs\benchmark.log
pause
