#!/usr/bin/env python3
"""
Development setup script for Prism Backend
This script helps set up the development environment and install dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_venv():
    """Check if we're running in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def setup_venv():
    """Set up virtual environment if it doesn't exist"""
    venv_path = Path(".venv")
    
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment"):
            return False
        print("✅ Virtual environment created")
    
    # Get the correct python executable for the venv
    if platform.system() == "Windows":
        venv_python = venv_path / "Scripts" / "python.exe"
        venv_pip = venv_path / "Scripts" / "pip.exe"
    else:
        venv_python = venv_path / "bin" / "python"
        venv_pip = venv_path / "bin" / "pip"
    
    return str(venv_python), str(venv_pip)

def main():
    """Main setup function"""
    print("🔧 Setting up Prism Backend Development Environment")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Set up virtual environment
    if not check_venv():
        print("\\n🔄 Setting up virtual environment...")
        venv_result = setup_venv()
        if not venv_result:
            print("❌ Failed to set up virtual environment")
            return False
        venv_python, venv_pip = venv_result
        print(f"✅ Virtual environment ready at: {venv_python}")
        
        # Install dependencies in venv
        print("\\n📦 Installing Python dependencies in virtual environment...")
        if not run_command(f"{venv_pip} install -r requirements.txt", "Installing from requirements.txt"):
            print("❌ Failed to install dependencies from requirements.txt")
            return False
    else:
        print("\\n📦 Installing Python dependencies...")
        if not run_command("pip install -r requirements.txt", "Installing from requirements.txt"):
            print("❌ Failed to install dependencies from requirements.txt")
            return False
    
    # Create necessary directories
    print("\\n📁 Creating necessary directories...")
    directories = [
        "data/uploads",
        "data/processed", 
        "data/indices",
        "models/llm"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    # Check for LLM model
    model_path = Path("models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    if not model_path.exists():
        print("\\n⚠️  LLM Model not found!")
        print(f"Please download mistral-7b-instruct-v0.2.Q4_K_M.gguf to: {model_path}")
        print("Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    else:
        print(f"✅ LLM model found: {model_path}")
    
    # Test imports
    print("\\n🧪 Testing imports...")
    test_imports = [
        "fastapi",
        "uvicorn", 
        "PyPDF2",
        "docx",
        "tiktoken",
        "pydantic"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - install failed")
    
    # Test llama-cpp-python separately (it's more complex)
    try:
        import llama_cpp
        print("✅ llama_cpp")
    except ImportError:
        print("❌ llama_cpp - may need system dependencies or different build")
        print("   Try: pip install llama-cpp-python --force-reinstall --no-cache-dir")
    
    print("\\n🎉 Setup complete!")
    print("\\nTo run the development server:")
    print("  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)