@echo off
REM Switch scenarios only: re-score every rendered switch drive (no Blender needed) with
REM   teplock        TEP-Net + path lock
REM   tepswitch      + switch state read by the camera (runs\switch_cls\best.pt)
REM   tepplan        + planned route from the map
REM   tepplanswitch  + planned route + switch state (catches a switch set against the route)
REM   rescore_switch.bat         runs at 10 m/s (sim_out\*_10)
REM   rescore_switch.bat 15      runs at 15 m/s
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)
set SPEED=%~1
if "%SPEED%"=="" set SPEED=10
set CKPT=runs\rs19ego_pidnet_s_b4\best.pt
if not exist tep_weights (echo tep_weights folder not found. & pause & exit /b 1)
set SW=
if exist runs\switch_cls\best.pt set SW=1
for /d %%D in (sim_out\*_%SPEED%) do (
  python sim\has_switch.py "%%D"
  if not errorlevel 1 if exist "%%D\done.txt" (
    echo.
    echo ==================== %%~nxD : path lock ====================
    python sim\run_collision.py --run "%%D" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --tag teplock
    echo ==================== %%~nxD : + route plan ====================
    python sim\run_collision.py --run "%%D" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --route-plan --tag tepplan
    if defined SW (
      echo ==================== %%~nxD : + switch state from the camera ====================
      python sim\run_collision.py --run "%%D" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --switch-cls runs\switch_cls\best.pt --tag tepswitch
      echo ==================== %%~nxD : + route plan + switch state ====================
      python sim\run_collision.py --run "%%D" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --route-plan --switch-cls runs\switch_cls\best.pt --tag tepplanswitch
    )
  )
)
python sim\summarize.py sim_out
pause
