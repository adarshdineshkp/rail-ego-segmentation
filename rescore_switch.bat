@echo off
REM Switch scenarios only: re-score the rendered switch drives with the path lock, without and
REM with the switch state read from the camera (runs\switch_cls\best.pt). No Blender needed.
REM   rescore_switch.bat         speed folder _10 (36 km/h)
REM   rescore_switch.bat 15      sim_out\switch_*_15
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)
set SPEED=%~1
if "%SPEED%"=="" set SPEED=10
set CKPT=runs\rs19ego_pidnet_s_b4\best.pt
if not exist runs\switch_cls\best.pt (echo runs\switch_cls\best.pt not found - run train_switch.bat first. & pause & exit /b 1)
if not exist tep_weights (echo tep_weights folder not found. & pause & exit /b 1)
for %%R in (switch_blocked_%SPEED% switch_clear_%SPEED%) do (
  if exist "sim_out\%%R\done.txt" (
    echo.
    echo ==================== %%R : TEP-Net + path lock ====================
    python sim\run_collision.py --run "sim_out\%%R" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --tag teplock
    echo ==================== %%R : + switch state from the camera ====================
    python sim\run_collision.py --run "sim_out\%%R" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --lock --switch-cls runs\switch_cls\best.pt --tag tepswitch
  ) else (
    echo sim_out\%%R not rendered yet - run run_switch.bat first.
  )
)
python sim\summarize.py sim_out
pause
