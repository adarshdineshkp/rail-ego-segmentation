@echo off
REM Run the trained model on everything in test_footage\ (videos and/or images), with a live
REM preview window: q or Esc stops (a partial video is still saved); for images, any key = next.
REM Results go to preds\ : <name>_seg.mp4 for videos, <name>_overlay.jpg + <name>_label.png for images.
REM Optional first argument: another checkpoint, e.g.  test_footage.bat runs\other\best.pt
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)

set CKPT=runs\rs19ego_pidnet_s_b4\best.pt
if not "%~1"=="" set CKPT=%~1
if not exist "%CKPT%" (
  echo Checkpoint %CKPT% not found.
  pause
  exit /b 1
)
if not exist test_footage (
  mkdir test_footage
  echo Put your videos or images into %~dp0test_footage and run this again.
  pause
  exit /b 1
)

REM The preview window needs OpenCV with display support (setup installed the headless build)
python -c "import cv2,re,sys; sys.exit(0 if re.search(r'(Win32 UI|GTK\S*|Cocoa|QT)\s*:\s*YES', cv2.getBuildInformation()) else 1)"
if errorlevel 1 (
  echo Switching OpenCV to the build with display support ...
  python -m pip uninstall -y opencv-python-headless
  python -m pip install opencv-python
)

echo Model: %CKPT%
for %%F in (test_footage\*.mp4 test_footage\*.avi test_footage\*.mov test_footage\*.mkv) do (
  echo === %%F
  python infer.py --ckpt "%CKPT%" --input "%%F" --out preds --show
)
dir /b test_footage\*.jpg test_footage\*.jpeg test_footage\*.png test_footage\*.bmp >nul 2>nul && (
  echo === images in test_footage  ^(any key = next image, q = stop^)
  python infer.py --ckpt "%CKPT%" --input test_footage --out preds --show
)
echo.
echo Results are in %~dp0preds
pause
