@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
if errorlevel 1 exit /b 1
pushd malco_framework
call npm ci --no-audit --no-fund
if errorlevel 1 (popd & exit /b 1)
popd
echo Setup complete. R analyses are optional: run Rscript statistical_analysis/setup.R
