@echo off
REM Simulated drives: Blender renders a railway scene from a cab camera, and the collision
REM warning runs live on each frame as it is rendered. Ends with a scoreboard.
REM   run_sim.bat                         all four scenarios at 36 km/h
REM   run_sim.bat switch_clear            one scenario: vehicle, person, switch_blocked, switch_clear
REM   run_sim.bat all 15                  all scenarios at 15 m/s (54 km/h)
REM Your own YOLO models: put the .pt files in a "yolo" folder next to this script
REM (and pip install ultralytics); they are then used alongside the segmentation.
REM TEP-Net comparison: copy the "weights" folder from egopath.zip to a "tep_weights" folder
REM next to this script; each rendered drive is then also evaluated with TEP-Net's ego path.
REM Results: sim_out\<scenario>_<speed>\collision.mp4, collision_log.csv, summary.json
REM          (and collision_tep.mp4, summary_tep.json for the TEP-Net pass)
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)

set SCENARIOS=%~1
if "%SCENARIOS%"=="" set SCENARIOS=all
if /i "%SCENARIOS%"=="all" set SCENARIOS=vehicle person switch_blocked switch_clear
set SPEED=%~2
if "%SPEED%"=="" set SPEED=10
set CKPT=runs\rs19ego_pidnet_s_b4\best.pt
if not exist "%CKPT%" (echo Checkpoint %CKPT% not found. & pause & exit /b 1)

REM --- find Blender: BLENDER variable, remembered path, or a search (PATH, .blend file
REM     association, installed programs, Store execution alias, common folders, all drives)
if not exist logs mkdir logs
if not defined BLENDER if exist logs\blender_path.txt set /p BLENDER=<logs\blender_path.txt
if defined BLENDER if not exist "!BLENDER!" set BLENDER=
if defined BLENDER echo "%BLENDER%" | findstr /i /l /c:"BlenderFoundation.Blender_" >nul && set BLENDER=
if not defined BLENDER (
  echo Looking for Blender ...
  for /f "usebackq delims=" %%B in (`powershell -NoProfile -ExecutionPolicy Bypass -File sim\find_blender.ps1`) do set "BLENDER=%%B"
)
if not defined BLENDER (
  echo No Blender that can be started from a script was found.
  echo The Microsoft Store version cannot be started by scripts. Either:
  echo   - Settings ^> Apps ^> Advanced app settings ^> App execution aliases: turn on Blender, or
  echo   - download the Windows "Portable (.zip)" from https://www.blender.org/download/ and extract it
  echo Or set the path yourself:  set BLENDER=C:\path\to\blender.exe  then run this again.
  pause
  exit /b 1
)
> logs\blender_path.txt echo !BLENDER!
echo Blender: !BLENDER!

REM --- optional YOLO models
set YOLO_ARGS=
if exist yolo\*.pt (
  python -c "import ultralytics" 2>nul
  if errorlevel 1 (
    echo Found yolo\*.pt but the ultralytics package is missing - running without YOLO.
    echo To use them:  .venv\Scripts\pip install ultralytics
  ) else (
    set YOLO_ARGS=--yolo
    for %%W in (yolo\*.pt) do set YOLO_ARGS=!YOLO_ARGS! "%%W"
    echo YOLO models: !YOLO_ARGS!
  )
)

REM --- optional TEP-Net (needs torchvision, matched to the installed torch)
set TEP=0
if exist tep_weights (
  python -c "import torchvision" 2>nul
  if errorlevel 1 (
    echo Installing torchvision for TEP-Net ...
    for /f "delims=" %%V in ('python -c "import torch;print(torch.__version__)"') do set TORCHV=%%V
    python -m pip install torchvision "torch==!TORCHV!" --index-url https://download.pytorch.org/whl/cu126
  )
  python -c "import torchvision" 2>nul
  if errorlevel 1 (
    echo torchvision could not be installed - skipping the TEP-Net comparison.
  ) else (
    set TEP=1
    echo TEP-Net comparison: on ^(tep_weights^)
  )
)

for %%S in (%SCENARIOS%) do (
  set RUN=sim_out\%%S_%SPEED%
  echo.
  echo ==================== %%S at %SPEED% m/s ====================
  if exist "!RUN!" rmdir /s /q "!RUN!"
  mkdir "!RUN!"
  start "Blender render - %%S" /min cmd /c ""!BLENDER!" -b --factory-startup --python sim\build_scene.py -- --out "!RUN!" --scenario %%S --speed %SPEED% > logs\blender_%%S_%SPEED%.log 2>&1"
  python sim\run_collision.py --run "!RUN!" --ckpt "%CKPT%" --watch --show !YOLO_ARGS!
  if "!TEP!"=="1" (
    echo --- same drive with TEP-Net's ego path ---
    python sim\run_collision.py --run "!RUN!" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --tag tep !YOLO_ARGS!
    echo --- TEP-Net + path lock ---
    python sim\run_collision.py --run "!RUN!" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --tag teplock !YOLO_ARGS!
    if exist runs\switch_cls\best.pt python sim\has_switch.py "!RUN!" && (
      echo --- TEP-Net + path lock + switch state from the camera ---
      python sim\run_collision.py --run "!RUN!" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --switch-cls runs\switch_cls\best.pt --tag tepswitch !YOLO_ARGS!
    )
  )
)

python sim\summarize.py sim_out
echo.
echo Videos and logs are in %~dp0sim_out
pause
