# 🔮 Prism - Multimodal RAG System

<div align="center">

![Prism Logo](https://img.shields.io/badge/Prism-Multimodal%20RAG-blueviolet?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMjIgMjBIMloiIGZpbGw9InVybCgjZ3JhZGllbnQpIi8+CjxkZWZzPgo8bGluZWFyR3JhZGllbnQgaWQ9ImdyYWRpZW50IiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+CjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiM2MzY2ZjEiLz4KPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjOWMzNGY0Ii8+CjwvbGluZWFyR3JhZGllbnQ+CjwvZGVmcz4KPHN2Zz4K)

**Your intelligent multimodal search companion**

[![React](https://img.shields.io/badge/React-18.2.0-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🌟 Overview

Prism is a **multimodal RAG (Retrieval-Augmented Generation) system** that enables intelligent search and question-answering across documents (PDF, DOCX), images, and audio files. Built with a **local-first approach**, all processing happens on-device with no cloud dependencies, ensuring complete privacy and security.

### ✨ Key Features

- 🔍 **Multimodal Search** - Search across documents, images, and audio with natural language
- 🧠 **Document Q&A** - Ask questions about uploaded documents using Mistral 7B LLM
- 🏠 **Local-First** - All processing happens on your device, no cloud dependencies
- 📄 **Document Processing** - Support for PDF and DOCX files with intelligent chunking
- 🎨 **Modern UI** - Beautiful React interface with Tailwind CSS and Framer Motion
- 🚀 **Fast API** - High-performance FastAPI backend with async processing
- 🔒 **Privacy-Focused** - Your data never leaves your device

### 🎯 Current Status

**Implemented Features:**
- ✅ Document upload and processing (PDF/DOCX)
- ✅ Intelligent text chunking with token awareness
- ✅ Document Q&A using Mistral 7B via llama.cpp
- ✅ Modern React frontend with multiple interfaces
- ✅ FastAPI backend with comprehensive endpoints
- ✅ Local LLM integration for offline operation

**In Development:**
- 🔨 Image OCR and processing
- 🔨 Audio transcription
- 🔨 Vector embeddings for semantic search
- 🔨 Full multimodal search interface

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **8GB+ RAM** (for running Mistral 7B)
- **4GB+ disk space** (for model storage)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd prism
```

### 2. Set Up the Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the LLM Model

```bash
# Create models directory
mkdir -p models/llm

# Download Mistral 7B (4.1GB)
curl -L -o models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 4. Set Up the Frontend

```bash
cd frontend
npm install
```

### 5. Start the Application

**Terminal 1 (Backend):**
```bash
cd backend
.venv\Scripts\activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

Open your browser and go to `http://localhost:3000` 🎉

> 📋 For detailed setup instructions, see [SETUP_QA.md](SETUP_QA.md)

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │ Mistral 7B LLM  │
│   (Port 3000)   │◄──►│   (Port 8000)   │◄──►│  (llama.cpp)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  File Upload    │    │ Document        │    │ Question        │
│  Interface      │    │ Processing      │    │ Answering       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🔧 Tech Stack

**Frontend:**
- React 18 with Vite
- Tailwind CSS for styling
- Framer Motion for animations
- Lucide React for icons
- Axios for API calls

**Backend:**
- FastAPI for high-performance APIs
- llama.cpp for LLM inference
- PyPDF2 & python-docx for document parsing
- tiktoken for intelligent text chunking

**AI/ML:**
- Mistral 7B Instruct v0.2 (GGUF format)
- Token-aware document chunking
- Local inference with no cloud dependencies

---

## 📁 Project Structure

```
prism/
├── 🎨 frontend/                 # React application
│   ├── src/
│   │   ├── components/          # UI components
│   │   │   ├── DocumentQA.jsx   # Q&A interface
│   │   │   ├── FileUpload.jsx   # File upload UI
│   │   │   ├── SearchInterface.jsx
│   │   │   └── ...
│   │   ├── App.jsx              # Main app component
│   │   └── index.css            # Tailwind styles
│   └── package.json
│
├── 🔧 backend/                  # FastAPI application
│   ├── app/
│   │   ├── main.py              # API endpoints
│   │   └── services/
│   │       ├── llm_service.py   # Mistral LLM integration
│   │       └── qa_service.py    # Document Q&A logic
│   ├── ingestion/
│   │   ├── parse_pdf.py         # PDF/DOCX processing
│   │   └── chunker.py           # Intelligent text chunking
│   └── requirements.txt
│
├── 🧠 models/
│   └── llm/                     # LLM model storage
│       └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
│
├── 📊 data/                     # Application data
│   ├── uploads/                 # Uploaded files
│   ├── processed/               # Processed documents
│   └── indices/                 # Search indices
│
└── 📚 docs/                     # Documentation
    ├── architecture.md
    └── runbook.md
```

---

## 🎯 Usage

### Document Q&A

1. **Upload Documents** - Upload PDF or DOCX files via the Q&A interface
2. **Processing** - Documents are automatically chunked and indexed
3. **Ask Questions** - Type natural language questions about your documents
4. **Get Answers** - Receive contextual answers with source citations

**Example Questions:**
- "What are the main conclusions in this report?"
- "Summarize the budget section"
- "What recommendations are mentioned?"
- "Find information about project timelines"

### Search Interface

- **Text Search** - Search across all processed documents
- **Multimodal Support** - Future support for image and audio search
- **Smart Results** - Relevance scoring and source citations

---

## 🔧 Configuration

### Model Settings

Edit `backend/app/services/llm_service.py`:

```python
# Context window size (adjust for memory usage)
self.n_ctx = 4096

# Thread count (adjust for CPU cores)
self.n_threads = 4

# Temperature (0.0 = focused, 1.0 = creative)
temperature = 0.7
```

### Chunking Settings

Edit `backend/ingestion/chunker.py`:

```python
# Chunk size in tokens
chunk_size = 1000

# Overlap between chunks
chunk_overlap = 200
```

### Switching to API LLM Mode

By default Prism runs a local GGUF model. You can switch to using a hosted API model (e.g. OpenAI) without changing any code by setting environment variables.

Set these before starting the backend:

```powershell
$env:LLM_MODE = "api"            # Use remote API instead of local model
$env:LLM_PROVIDER = "openai"     # Currently supported provider
$env:LLM_API_KEY = "sk-your-key" # Your API key
$env:LLM_MODEL_NAME = "gpt-4o-mini" # Remote model name
# Optional override if needed:
$env:LLM_API_BASE = "https://api.openai.com/v1"
```

Then start the backend normally:

```powershell
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If you want to revert to the local model just remove or unset `LLM_MODE` (or set it to `local`).

Local mode variables (optional overrides):

```powershell
$env:LLM_MODE = "local"
$env:LLM_MODEL_PATH = "C:\path\to\model.gguf"  # Overrides auto-discovery
```

The backend auto-selects implementation based on `LLM_MODE` and exposes the same endpoints. The health endpoint `/` will show whether the model is loaded (`llm_ready`).

#### Using Google Gemini

Install dependencies (already in `requirements.txt` after update):

```powershell
pip install -r requirements.txt
```

Set environment variables:

```powershell
$env:LLM_MODE = "api"
$env:LLM_PROVIDER = "google"
$env:LLM_API_KEY = "your-gemini-key"        # DO NOT commit this
$env:LLM_MODEL_NAME = "gemini-2.5-flash"     # Or gemini-1.5-pro / gemini-2.0-flash-exp
```

Start backend and test `/api/chat`.

Security tip: If an API key is accidentally exposed publicly, rotate it immediately in the provider console and update your environment variable.

### Environment File (.env)

You can store your configuration in a `.env` file (never commit real secrets):

1. Copy `backend/.env.example` to `backend/.env`.
2. Fill in values (replace `YOUR_API_KEY_HERE`).
3. The backend automatically loads `.env` via `python-dotenv`.

PowerShell example (without .env):
```powershell
$env:LLM_MODE = "api"
$env:LLM_PROVIDER = "google"
$env:LLM_API_KEY = "your-key"
```

Git best practice:
- Add `.env` to your `.gitignore`.
- Rotate any key that was ever committed or shared.
- Use separate keys for development and production.

Key Rotation Steps (Gemini/OpenAI):
1. Log in to provider console.
2. Create a new key.
3. Update local `.env` or environment variables.
4. Delete the old key in the console.
5. Redeploy/restart backend.

### Supported / Planned Providers

- openai (implemented)
- anthropic (planned)
- groq (planned)
- ollama (planned local API)

Provider additions only require a new class implementing the `BaseLLM` interface in `backend/app/services/llm_service.py`.

---

## 🛠️ Development

### Frontend Development

```bash
cd frontend
npm run dev     # Start development server
npm run build   # Build for production
npm run lint    # Run ESLint
```

### Backend Development

```bash
cd backend
.venv\Scripts\activate
python -m uvicorn app.main:app --reload  # Start with auto-reload
```

### Testing

```bash
# Run backend tests
cd backend
python -m pytest tests/

# Run frontend tests
cd frontend
npm test
```

---

## 📊 API Documentation

The FastAPI backend provides comprehensive API documentation:

- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and model status |
| `POST` | `/api/upload` | Upload and process documents |
| `POST` | `/api/question` | Ask questions about documents |
| `GET` | `/api/documents` | List processed documents |
| `DELETE` | `/api/documents/{file_id}` | Delete documents |
| `GET` | `/api/model/status` | Check LLM model status |

---

## 🔒 Privacy & Security

- **Local Processing** - All data processing happens on your device
- **No Cloud Dependencies** - Documents never leave your machine
- **Offline Operation** - Works completely offline after setup
- **Data Control** - You own and control all your data

---

## 🤝 Contributing

We welcome contributions! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Documentation**: Check [SETUP_QA.md](SETUP_QA.md) for detailed setup
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join discussions for questions and ideas

---

<div align="center">

**Built with ❤️ for privacy-focused AI**

[⭐ Star this repo](../../stargazers) • [🐛 Report Bug](../../issues) • [💡 Request Feature](../../issues)

</div>