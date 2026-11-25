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

REM --- find Blender: BLENDER variable, remembered path, or a search (PATH, .blend file
REM     association, installed programs, Microsoft Store, Steam, common folders, all drives)
if not exist logs mkdir logs
if not defined BLENDER if exist logs\blender_path.txt set /p BLENDER=<logs\blender_path.txt
if defined BLENDER if not exist "%BLENDER%" set BLENDER=
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
  echo Find blender.exe ^(right-click the Blender shortcut, Open file location^), then run:
  echo     set BLENDER=C:\path\to\blender.exe
  echo     run_sim.bat
  echo or paste that path into logs\blender_path.txt
  pause
  exit /b 1
)
> logs\blender_path.txt echo %BLENDER%
echo Blender: %BLENDER%

if exist "%RUN%" rmdir /s /q "%RUN%"
mkdir "%RUN%"

echo Starting the render in a second window ...
start "Blender render - %SCEN%" cmd /c ""%BLENDER%" -b --factory-startup --python sim\build_scene.py -- --out "%RUN%" --scenario %SCEN% --speed %SPEED% > logs\blender_%SCEN%_%SPEED%.log 2>&1"

echo Collision warning, live (q or Esc in the window to stop) ...
python sim\run_collision.py --run "%RUN%" --ckpt "%CKPT%" --watch --show
echo.
echo Video and log: %~dp0%RUN%   (Blender log: logs\blender_%SCEN%_%SPEED%.log)
pause
