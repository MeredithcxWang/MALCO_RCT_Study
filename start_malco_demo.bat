@echo off
setlocal
cd /d "%~dp0"
set NEXT_TELEMETRY_DISABLED=1
cd malco_framework
call npm run dev
exit /b %errorlevel%
