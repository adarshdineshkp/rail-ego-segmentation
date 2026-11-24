@echo off
REM Train PIDNet-S (rail classes) on a 6 GB RTX 3050. Run again to resume after a stop.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

if not exist data\railsem19 (
  echo RailSem19 not found. Extract rs19_val.zip into %~dp0data\railsem19 first.
  pause
  exit /b 1
)
if not exist data\splits\train.txt (
  python tools\prepare_railsem19.py --root data/railsem19 --out data/splits
  if errorlevel 1 goto :fail
)

python train.py --config configs/railsem19_pidnet_s_rails.yaml --resume auto ^
  train.batch_size=2 train.lr=0.0025 "data.train.crop_size=[640,1024]" ^
  train.epochs=40 data.num_workers=4
if errorlevel 1 goto :fail
echo Training finished. Checkpoints: runs\rs19_pidnet_s_rails\best.pt
pause
exit /b 0

:fail
echo TRAINING STOPPED WITH AN ERROR - send the messages above to Claude.
pause
exit /b 1
