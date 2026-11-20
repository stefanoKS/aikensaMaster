@echo off
REM ===================================================================
REM AIKensaMaster - machine configuration backup (RUN BEFORE git pull)
REM ===================================================================
REM 1. Close AIKensaMaster.
REM 2. Double-click this file (or run it from cmd.exe).
REM 3. Confirm it prints "BACKUP VERIFIED".
REM 4. Only then run: git pull
REM
REM The backup is written OUTSIDE the repository work tree
REM (default: ..\aikensa_backups) so a pull cannot delete it.
setlocal
cd /d "%~dp0.."
if not defined AIKENSA_BACKUP_DIR (
  set "AIKENSA_BACKUP_DIR=%~dp0..\..\aikensa_backups"
)
echo Backup destination: %AIKENSA_BACKUP_DIR%
python tools\machine_config_backup.py backup --reason prepull
if errorlevel 1 (
  echo.
  echo *** BACKUP FAILED OR NOT VERIFIED - DO NOT RUN git pull ***
  pause
  exit /b 1
)
echo.
echo Backup verified. It is now safe to run: git pull
pause
