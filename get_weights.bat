@echo off
REM Download Cityscapes-pretrained PIDNet-S (MMSegmentation mirror) and convert it for this repo.
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)
if not exist logs mkdir logs
python tools\convert_mmseg_pidnet.py --model s > logs\get_weights.log 2>&1
type logs\get_weights.log
echo.
echo (This output is also saved in logs\get_weights.log)
pause
