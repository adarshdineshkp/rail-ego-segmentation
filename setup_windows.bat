@echo off
REM One-time setup on Windows: virtual env, PyTorch (CUDA 12.4), requirements, smoke test, weights.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run this file again.
  pause
  exit /b 1
)
nvidia-smi >nul 2>nul
if errorlevel 1 echo WARNING: nvidia-smi not found - update your NVIDIA driver if the GPU is not detected.

if not exist .venv (
  echo Creating virtual environment .venv ...
  python -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail

python -m pip install --upgrade pip
echo Installing PyTorch with CUDA 12.4 (about 2.5 GB download) ...
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || goto :fail
python -m pip install -r requirements.txt || goto :fail

python -c "import torch; ok=torch.cuda.is_available(); print('PyTorch', torch.__version__, '| CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU FOUND')"

echo.
echo Running smoke test ...
python tools\smoke_test.py || goto :fail

echo.
echo Downloading pretrained PIDNet-S weights ...
python tools\download_weights.py --model s

echo.
echo ================================================================
echo  Setup finished.
echo  Next: download rs19_val.zip from https://www.wilddash.cc/railsem19
echo  and extract it into  %~dp0data\railsem19
echo  then double-click  train_rtx3050.bat
echo ================================================================
pause
exit /b 0

:fail
echo.
echo SETUP FAILED - copy the messages above and send them to Claude.
pause
exit /b 1
