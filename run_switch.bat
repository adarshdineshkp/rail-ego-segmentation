@echo off
REM Only the two switch scenarios (switch_blocked, switch_clear). Optional speed in m/s:
REM   run_switch.bat        36 km/h
REM   run_switch.bat 15     54 km/h
call "%~dp0run_sim.bat" "switch_blocked switch_clear" %1
