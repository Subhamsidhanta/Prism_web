@echo off
REM Quick start script for Prism Backend
echo 🔮 Starting Prism Backend...

REM Navigate to backend directory
cd /d "%~dp0backend"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found. Using system Python.
)

REM Run the server
echo 🚀 Starting server...
python run_server.py

pause