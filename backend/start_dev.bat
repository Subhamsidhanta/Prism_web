@echo off
REM Activate virtual environment and run the backend server
echo 🔮 Starting Prism Backend Development Environment...

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/update dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if model exists
if not exist "models\llm\mistral-7b-instruct-v0.2.Q4_K_M.gguf" (
    echo ⚠️  LLM model not found!
    echo Please download mistral-7b-instruct-v0.2.Q4_K_M.gguf to models\llm\
    echo Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
    pause
)

REM Run the server
echo 🚀 Starting development server...
python run_server.py

pause