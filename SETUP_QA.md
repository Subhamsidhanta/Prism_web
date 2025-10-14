# Document Q&A Setup Instructions

This guide will help you set up the Document Q&A feature using Mistral 7B with llama.cpp.

## Prerequisites

1. **Python 3.8+** with pip
2. **Node.js 16+** with npm
3. **At least 8GB RAM** for running Mistral 7B
4. **4GB disk space** for the model file

## Step 1: Download the Mistral 7B Model

Download the quantized GGUF model file:

```bash
# Create models directory
mkdir -p models/llm

# Download Mistral 7B Instruct v0.2 Q4_K_M (recommended for balance of speed/quality)
# Option 1: Using wget
wget -O models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Option 2: Using curl
curl -L -o models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Option 3: Manual download
# Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
# Download and place in models/llm/ directory
```

**Alternative Model Sizes:**
- `Q2_K.gguf` - Smallest, fastest, lower quality (~2.3GB)
- `Q4_K_M.gguf` - **Recommended** - Good balance (~4.1GB)
- `Q5_K_M.gguf` - Higher quality, slower (~4.8GB)
- `Q8_0.gguf` - Highest quality, slowest (~7.2GB)

## Step 2: Set Up Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with specific optimizations (optional)
# For CPU only:
pip install llama-cpp-python --force-reinstall --no-cache-dir

# For GPU acceleration (if you have CUDA):
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# For Metal acceleration (macOS with Apple Silicon):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Step 3: Set Up Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Install additional dependency for API calls if not present
npm install axios
```

## Step 4: Start the Application

### Start Backend (Terminal 1):
```bash
cd backend
# Activate virtual environment if not already active
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Start FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend (Terminal 2):
```bash
cd frontend
npm run dev
```

## Step 5: Using the Q&A Feature

1. **Open your browser** and go to `http://localhost:3000`
2. **Click on "Document Q&A"** in the sidebar
3. **Upload a PDF or DOCX file** using the upload button
4. **Wait for processing** - the document will be chunked and indexed
5. **Ask questions** about the document content in the chat interface

### Example Questions:
- "What is the main topic of this document?"
- "Summarize the key points from page 3"
- "What are the conclusions mentioned in the document?"
- "Find any information about budget or costs"

## Troubleshooting

### Model Not Loading
- **Check file path**: Ensure the model file is in `models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- **Check file size**: The Q4_K_M model should be ~4.1GB
- **Memory issues**: Try a smaller model like Q2_K if you have limited RAM

### Backend Errors
```bash
# Check if all dependencies are installed
pip list | grep -E "(fastapi|llama-cpp-python|PyPDF2|python-docx)"

# Reinstall llama-cpp-python if needed
pip uninstall llama-cpp-python
pip install llama-cpp-python --no-cache-dir
```

### API Connection Issues
- Ensure backend is running on `http://localhost:8000`
- Check CORS settings in `backend/app/main.py`
- Verify firewall settings

### Document Processing Issues
- **PDF files**: Ensure they contain readable text (not just images)
- **DOCX files**: Make sure they're valid Word documents
- **File size**: Maximum 100MB per file

## Performance Tips

1. **Hardware Requirements**:
   - **RAM**: 8GB minimum, 16GB recommended
   - **CPU**: Modern multi-core processor
   - **Storage**: SSD recommended for faster model loading

2. **Model Selection**:
   - Use Q4_K_M for best balance
   - Use Q2_K for faster responses on limited hardware
   - Use Q8_0 for highest quality if you have powerful hardware

3. **Optimization**:
   - Close other applications to free up RAM
   - Use GPU acceleration if available
   - Process smaller documents for faster responses

## Configuration

### Customize Model Parameters
Edit `backend/app/services/llm_service.py`:

```python
# Adjust context window size (default 4096)
self.n_ctx = 4096  # Reduce for less RAM usage

# Adjust thread count (default 4)
self.n_threads = 8  # Increase for faster processing

# Adjust temperature (default 0.7)
temperature = 0.3  # Lower for more focused answers
```

### Customize Chunking
Edit `backend/ingestion/chunker.py`:

```python
# Adjust chunk size (default 1000 tokens)
chunk_size = 800  # Smaller for more precise answers

# Adjust overlap (default 200 tokens)
chunk_overlap = 100  # Less overlap for speed
```

## API Endpoints

The backend provides these endpoints:

- `GET /` - Health check and model status
- `POST /api/upload` - Upload and process documents
- `POST /api/question` - Ask questions about documents
- `GET /api/documents` - List processed documents
- `GET /api/documents/{file_id}` - Get document information
- `DELETE /api/documents/{file_id}` - Delete processed document
- `GET /api/model/status` - Check model loading status

## Security Notes

- All processing happens locally - no data sent to external services
- Documents are stored in `data/uploads/` and `data/processed/`
- Consider encrypting sensitive documents at rest
- The API runs on localhost by default for security

## Getting Help

If you encounter issues:

1. Check the browser console for frontend errors
2. Check the terminal running the backend for Python errors
3. Verify all dependencies are installed correctly
4. Ensure the model file is downloaded and in the correct location
5. Check available system memory and disk space