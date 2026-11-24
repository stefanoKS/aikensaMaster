@echo off
setlocal
cd /d "%~dp0.."
python tools\checkgates.py %*
exit /b %ERRORLEVEL%
