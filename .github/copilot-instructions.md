# Copilot Instructions for Prism

## Project Overview
Prism is a **multimodal RAG (Retrieval-Augmented Generation) system** that enables intelligent search across documents (PDF, DOCX), images, and audio files. The architecture follows a **local-first approach** - all processing happens on-device with no cloud dependencies.

**NEW**: Document Q&A feature using **Mistral 7B** with llama.cpp for answering questions about uploaded documents.

## Architecture & Service Boundaries

### Core Components
- **Frontend**: React 18 + Vite at `frontend/src/` - UI for search, upload, and results display
- **Backend**: FastAPI at `backend/app/main.py` - API with Q&A endpoints for document processing
- **LLM Integration**: Mistral 7B via llama.cpp at `backend/app/services/llm_service.py`
- **Q&A Service**: Document processing and question answering at `backend/app/services/qa_service.py`
- **Processing Pipeline**: Modular ingestion at `backend/ingestion/` - document parsing, OCR, transcription
- **Document Chunking**: Smart text splitting at `backend/ingestion/chunker.py` with token-aware chunking
- **Embeddings**: Text/image vectorization at `backend/embeddings/` - currently placeholder
- **Workers**: Background processing at `workers/` - async file processing
- **Data Storage**: Organized at `data/` with `uploads/`, `processed/`, `indices/` directories

### Key Data Flow
1. **Upload**: Files → `frontend/FileUpload.jsx` → `data/uploads/`
2. **Processing**: `workers/ingest_worker.py` → `backend/ingestion/` modules → `data/processed/`
3. **Q&A Processing**: `DocumentQA.jsx` → API → `qa_service.py` → `llm_service.py` (Mistral 7B)
4. **Document Chunking**: `parse_pdf.py` → `chunker.py` → stored chunks with metadata
5. **Search**: `frontend/SearchInterface.jsx` → embeddings → results via `ResultsDisplay.jsx`

## Development Workflows

### Frontend Development
```bash
cd frontend
npm run dev    # Development server on localhost:3000
npm run build  # Production build
npm run lint   # ESLint check
```

### Backend Development
```bash
# Virtual environment in .venv/
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
cd backend
# Install dependencies including llama-cpp-python, fastapi, tiktoken
pip install -r requirements.txt
# Start FastAPI server with Q&A endpoints
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### LLM Setup
- **Model**: Download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` to `models/llm/`
- **Service**: `llm_service.py` handles Mistral 7B via llama.cpp
- **Integration**: `qa_service.py` combines document processing with LLM responses
- **Setup Guide**: See `SETUP_QA.md` for detailed installation instructions

### File Processing Pipeline
- **Documents**: Use `backend/ingestion/parse_pdf.py` - supports both PDF (PyPDF2) and DOCX (python-docx)
- **Return format**: `[{"file_id": int, "page": int, "text": str}]` for chunks
- **Chunking**: `chunker.py` splits documents into token-aware chunks with overlap
- **Q&A Flow**: Upload → Process → Chunk → Store → Query → LLM Response
- **Images**: Process via `backend/ingestion/ocr_image.py` (placeholder)
- **Audio**: Handle through `backend/ingestion/transcribe_audio.py` (placeholder)

## Project-Specific Patterns

### Frontend Conventions
- **Components**: Use functional components with hooks, located in `src/components/`
- **Styling**: Tailwind classes with custom utilities in `src/index.css`:
  - `.search-container` - white rounded cards with shadow
  - `.btn-primary` - gradient primary buttons with hover effects
  - `.gradient-text` - primary-to-purple text gradient
  - `.card-hover` - standard hover animation (lift + shadow)
- **Animations**: Framer Motion for all transitions - use `motion.div` with `initial/animate` props
- **State Management**: Local useState - no global state library
- **Icons**: Lucide React exclusively (`Search`, `Upload`, `FileText`, `Image`, `Mic`)
- **Notifications**: React Hot Toast for user feedback

### Styling System
```css
/* Custom Tailwind utilities defined in src/index.css */
@layer components {
  .search-container { /* Main card container */ }
  .btn-primary { /* Primary action buttons */ }
  .gradient-text { /* Hero text styling */ }
}
```

### Backend Patterns
- **File Processing**: All ingestion functions take `(path, file_id=1)` parameters
- **Error Handling**: Use descriptive `ValueError` for unsupported formats
- **Return Format**: Consistent chunk structure with `file_id`, `page`/`timestamp`, `text` fields
- **LLM Integration**: `mistral_llm.answer_question(context, question)` for Q&A
- **API Responses**: FastAPI with Pydantic models for type safety
- **Dependencies**: PyPDF2, python-docx, FastAPI, llama-cpp-python, tiktoken

### Directory Conventions
- **Empty Folders**: Many folders (`backend/api/`, `models/embed/`, etc.) are placeholder structure
- **Separation**: Clear frontend/backend/data/docs division
- **Processing**: Use `workers/` for background tasks, `backend/ingestion/` for core processing
- **Storage**: `data/uploads/` → `data/processed/` → `data/indices/` workflow

## Integration Points
- **Frontend ↔ Backend**: API endpoints implemented:
  - `POST /api/upload` - file upload and processing
  - `POST /api/question` - Q&A queries with context
  - `GET /api/documents` - list processed documents
  - `GET /api/documents/{file_id}` - document metadata
  - `DELETE /api/documents/{file_id}` - remove documents
  - `GET /api/model/status` - LLM model status
- **File Types**: Support matrix in `frontend/FileUpload.jsx` - PDF/DOCX, images (JPG/PNG/GIF), audio (MP3/WAV/M4A)
- **Q&A Interface**: `DocumentQA.jsx` - chat interface with document context and source citations
- **Processing**: Real-time via FastAPI - document chunking and LLM inference

## Development Notes
- **Local Development**: Frontend on :3000, backend setup minimal
- **File Uploads**: Max 100MB per file, drag-and-drop interface with progress simulation
- **Search Interface**: Voice search placeholder, suggested queries, recent history
- **Privacy**: Emphasize local processing in UI copy - "never leave your device"
- **Responsive**: Mobile-first with Tailwind breakpoints (md:, lg: prefixes)

## Debugging & Tools
- **Debug Tools**: `tools/debug_viewer.py` and `tools/export_citation.py` (placeholders)
- **Build Tools**: Vite for frontend, standard Python for backend
- **Testing**: Placeholder structure at `tests/unit/` and `tests/integration/`


