#!/bin/bash
# Activate virtual environment and run the backend server for Unix systems

echo "🔮 Starting Prism Backend Development Environment..."

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if model exists
if [ ! -f "models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf" ]; then
    echo "⚠️  LLM model not found!"
    echo "Please download mistral-7b-instruct-v0.2.Q4_K_M.gguf to models/llm/"
    echo "Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    read -p "Press Enter to continue..."
fi

# Run the server
echo "🚀 Starting development server..."
python run_server.py