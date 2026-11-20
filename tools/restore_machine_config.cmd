@echo off
REM Restore machine-specific configuration from the newest VERIFIED backup.
setlocal
cd /d "%~dp0.."
echo Close AIKensaMaster before restoring.
pause
python tools\machine_config_backup.py restore --dry-run
echo.
echo The lines above list what WOULD be restored.
set /p confirm=Restore now? (Y/N):
if /i not "%confirm%"=="Y" (
  echo Cancelled.
  pause
  exit /b 0
)
python tools\machine_config_backup.py restore
pause
