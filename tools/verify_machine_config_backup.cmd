@echo off
REM Re-verify the newest machine configuration backup.
setlocal
cd /d "%~dp0.."
python tools\machine_config_backup.py verify
if errorlevel 1 (
  echo.
  echo *** VERIFICATION FAILED ***
  pause
  exit /b 1
)
echo.
echo All backups verified.
pause
