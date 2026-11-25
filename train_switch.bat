@echo off
REM Train the switch-state classifier (left / right) on RailSem19 switch crops.
REM   train_switch.bat                 left / right
REM   train_switch.bat unknown         also learn "unknown" (state not visible)
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)

python -c "import torchvision" 2>nul
if errorlevel 1 (
  echo Installing torchvision ...
  for /f "delims=" %%V in ('python -c "import torch;print(torch.__version__)"') do set TORCHV=%%V
  python -m pip install torchvision "torch==!TORCHV!" --index-url https://download.pytorch.org/whl/cu126 || goto :fail
)

set EXTRA=
if /i "%~1"=="unknown" set EXTRA=--with-unknown
if exist data\switch_crops rmdir /s /q data\switch_crops
echo Cutting switch crops from RailSem19 ...
python tools\prepare_switch_crops.py --root data/railsem19 --splits data/splits %EXTRA% || goto :fail

if not exist logs mkdir logs
echo Training ...
python train_switch.py --workers 4
if errorlevel 1 goto :fail
echo.
echo Done. Model: runs\switch_cls\best.pt   metrics: runs\switch_cls\metrics.json
pause
exit /b 0
:fail
echo FAILED - send the messages above to Claude.
pause
exit /b 1
