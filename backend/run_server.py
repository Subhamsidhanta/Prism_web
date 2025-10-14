#!/usr/bin/env python3
"""
Startup script for Prism backend server
Automatically uses virtual environment if available
"""
import os
import sys
import subprocess
from pathlib import Path

def check_venv():
    """Check if we're running in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def get_venv_python():
    """Get the path to virtual environment Python"""
    current_dir = Path(__file__).parent
    venv_python = current_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return None

def main():
    current_dir = Path(__file__).parent
    
    # If not in venv and venv exists, restart with venv python
    if not check_venv():
        venv_python = get_venv_python()
        if venv_python:
            print(f"🔄 Switching to virtual environment: {venv_python}")
            subprocess.run([venv_python, __file__] + sys.argv[1:])
            return
        else:
            print("⚠️  Virtual environment not found. Run: python -m venv .venv")
            print("   Then activate it and install dependencies: pip install -r requirements.txt")
    
    # Add the current directory to Python path
    sys.path.insert(0, str(current_dir))
    
    # Import uvicorn after path setup
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn not found. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    print("🚀 Starting Prism Backend Server...")
    print(f"📁 Working directory: {current_dir}")
    print(f"🐍 Python: {sys.executable}")
    print(f"🌐 Server will be available at: http://localhost:8000")
    print(f"📖 API docs at: http://localhost:8000/docs")
    
    # Use import string format for proper reload functionality
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=[str(current_dir)],
        log_level="info"
    )

if __name__ == "__main__":
    main()