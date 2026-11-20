@echo off
REM Run the software-only test suite (no cameras / GPU / PLC required).
setlocal
cd /d "%~dp0.."
REM Workstations with ROS 2 installed ship pytest plugins that fail to import.
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest
pause
