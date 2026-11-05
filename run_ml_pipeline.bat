@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe scripts\run_demo.py
) else (
  python scripts\run_demo.py
)
exit /b %errorlevel%
