@echo off
REM Train PIDNet-S on a 6 GB RTX 3050. Run again to resume after a stop.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
if not exist logs mkdir logs

if not exist data\railsem19 (
  echo RailSem19 not found. Extract it into %~dp0data\railsem19 first.
  pause
  exit /b 1
)

set CONFIG=configs/railsem19_pidnet_s_rails.yaml
set RASTER=0
if not exist data\railsem19\uint8 set RASTER=1
if exist data\railsem19\json-classes.json set RASTER=1

if "%RASTER%"=="1" (
  REM Copy without dense label maps: draw masks from the JSON annotations
  REM ^(skips masks that are already up to date^)
  python tools\rasterize_rs19_json.py --root data/railsem19
  if errorlevel 1 goto :fail
  set CONFIG=configs/railsem19_json_pidnet_s.yaml
  if exist data\railsem19\rs19_egopath.json set CONFIG=configs/railsem19_json_ego_pidnet_s.yaml
)

python tools\prepare_railsem19.py --root data/railsem19 --out data/splits --stats 0
if errorlevel 1 goto :fail

echo Config: %CONFIG%
REM batch 4 uses ~2 GB of the 6 GB; 6 workers keep the GPU fed (each worker uses one CPU core)
python train.py --config %CONFIG% --resume auto ^
  train.batch_size=4 train.lr=0.005 "data.train.crop_size=[640,1024]" ^
  train.epochs=40 data.num_workers=6 train.log_every=100
if errorlevel 1 goto :fail
echo Training finished. Checkpoints are in the runs folder.
pause
exit /b 0

:fail
echo TRAINING STOPPED WITH AN ERROR - send the messages above to Claude.
pause
exit /b 1
