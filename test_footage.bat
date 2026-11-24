@echo off
REM Run the trained model on everything in test_footage\ (videos and/or images).
REM Results go to preds\ : <name>_seg.mp4 for videos, <name>_overlay.jpg + <name>_label.png for images.
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)
if not exist logs mkdir logs

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

echo Model: %CKPT%
(
  for %%F in (test_footage\*.mp4 test_footage\*.avi test_footage\*.mov test_footage\*.mkv) do (
    echo === %%F
    python infer.py --ckpt "%CKPT%" --input "%%F" --out preds
  )
  dir /b test_footage\*.jpg test_footage\*.jpeg test_footage\*.png test_footage\*.bmp >nul 2>nul && (
    echo === images in test_footage
    python infer.py --ckpt "%CKPT%" --input test_footage --out preds
  )
) > logs\test_footage.log 2>&1
type logs\test_footage.log
echo.
echo Results are in %~dp0preds  (log: logs\test_footage.log)
pause
