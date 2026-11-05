@echo off
setlocal
cd /d "%~dp0"
if not defined MALCO_MODE set MALCO_MODE=demo
set NEXT_TELEMETRY_DISABLED=1
cd malco_framework
call npm run dev
exit /b %errorlevel%
