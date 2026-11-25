@echo off
REM Re-run the collision warning on drives that were already rendered (no Blender needed).
REM Useful after changing the logic, or if a pass was stopped early.
REM   rescore_sim.bat                       every run in sim_out
REM   rescore_sim.bat switch_clear_10       one run
REM Press q / Esc only if you want to cut a pass short (its verdict then covers fewer frames).
setlocal EnableDelayedExpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)
set CKPT=runs\rs19ego_pidnet_s_b4\best.pt
set RUNS=%~1
if "%RUNS%"=="" (
  set RUNS=
  for /d %%D in (sim_out\*) do set RUNS=!RUNS! %%~nxD
)
set YOLO_ARGS=
if exist yolo\*.pt (
  python -c "import ultralytics" 2>nul
  if not errorlevel 1 (
    set YOLO_ARGS=--yolo
    for %%W in (yolo\*.pt) do set YOLO_ARGS=!YOLO_ARGS! "%%W"
  )
)
for %%R in (%RUNS%) do (
  if exist "sim_out\%%R\done.txt" (
    echo.
    echo ==================== %%R ====================
    python sim\run_collision.py --run "sim_out\%%R" --ckpt "%CKPT%" --show !YOLO_ARGS!
    if exist tep_weights (
      echo --- same drive with TEP-Net's ego path ---
      python sim\run_collision.py --run "sim_out\%%R" --ckpt "%CKPT%" --show --ego tep --tep-weights tep_weights --tag tep !YOLO_ARGS!
    )
  ) else (
    echo %%R: rendering not finished ^(no done.txt^), skipped
  )
)
python sim\summarize.py sim_out
pause
