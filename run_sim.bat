@echo off
REM Simulated drive: Blender renders a railway scene from a cab camera, and the collision
REM warning runs live on each frame as it is rendered.
REM   run_sim.bat                    vehicle on the track in a curve, 36 km/h
REM   run_sim.bat person             person standing on the track
REM   run_sim.bat vehicle 15         same, at 15 m/s (54 km/h)
REM Results: sim_out\<scenario>\collision.mp4, collision_log.csv, summary.json
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & pause & exit /b 1)

set SCEN=%~1
if "%SCEN%"=="" set SCEN=vehicle
set SPEED=%~2
if "%SPEED%"=="" set SPEED=10
set RUN=sim_out\%SCEN%_%SPEED%
set CKPT=runs\rs19ego_pidnet_s_b4\best.pt
if not exist "%CKPT%" (echo Checkpoint %CKPT% not found. & pause & exit /b 1)

REM --- find Blender: BLENDER variable, PATH, or the default install folders
if not defined BLENDER (
  for /f "delims=" %%B in ('where blender 2^>nul') do if not defined BLENDER set "BLENDER=%%B"
)
if not defined BLENDER (
  for /d %%D in ("%ProgramFiles%\Blender Foundation\Blender*") do if exist "%%D\blender.exe" set "BLENDER=%%D\blender.exe"
)
if not defined BLENDER (
  if exist "%ProgramFiles(x86)%\Steam\steamapps\common\Blender\blender.exe" set "BLENDER=%ProgramFiles(x86)%\Steam\steamapps\common\Blender\blender.exe"
)
if not defined BLENDER (
  echo Blender was not found. Install it from https://www.blender.org/download/
  echo or run:  winget install BlenderFoundation.Blender
  echo If it is installed somewhere else:  set BLENDER=C:\path\to\blender.exe  then run this again.
  pause
  exit /b 1
)
echo Blender: %BLENDER%

if exist "%RUN%" rmdir /s /q "%RUN%"
mkdir "%RUN%"
if not exist logs mkdir logs

echo Starting the render in a second window ...
start "Blender render - %SCEN%" cmd /c ""%BLENDER%" -b --factory-startup --python sim\build_scene.py -- --out "%RUN%" --scenario %SCEN% --speed %SPEED% > logs\blender_%SCEN%_%SPEED%.log 2>&1"

echo Collision warning, live (q or Esc in the window to stop) ...
python sim\run_collision.py --run "%RUN%" --ckpt "%CKPT%" --watch --show
echo.
echo Video and log: %~dp0%RUN%   (Blender log: logs\blender_%SCEN%_%SPEED%.log)
pause
