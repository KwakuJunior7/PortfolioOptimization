@echo off
REM One-time setup: creates .venv and installs everything in requirements.txt
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 -m venv .venv) else (python -m venv .venv)
if not exist .venv\Scripts\python.exe (
  echo Could not create the virtual environment. Install Python 3.10+ from python.org and tick "Add to PATH".
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (echo. & echo pip install failed - see the error above. & exit /b 1)
echo.
echo Setup complete. In VS Code: Ctrl+Shift+P, "Python: Select Interpreter", pick .venv
echo Then press F5 and choose "1. Run full pipeline (live data)".
