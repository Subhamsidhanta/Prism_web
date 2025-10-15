"""
Document Q&A Service
Combines document processing with LLM for answering questions
"""
import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

# Add the backend directory to the Python path for ingestion modules
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from ingestion.parse_pdf import parse_document
    from ingestion.chunker import document_chunker
except ImportError as e:
    logger.error(f"Import error for ingestion modules: {e}")
    raise ImportError(f"Failed to import ingestion modules. Please ensure dependencies are installed: {e}")

from .llm_service import mistral_llm
from .progress_service import progress_service

class DocumentQAService:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Document Q&A service with performance optimizations

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # In-memory document store with caching
        self.document_chunks: Dict[str, List[Dict]] = {}
        self.document_metadata: Dict[str, Dict] = {}
        self._chunks_cache: Dict[str, List[Dict]] = {}

        # Load existing processed documents on startup
        self._load_existing_documents()

    def process_document_with_progress(self, file_path: str, file_id: str = None, progress_file_id: str = None) -> Dict:
        """
        Process a document and store its chunks with progress tracking

        Args:
            file_path: Path to the document file
            file_id: Unique identifier for the document
            progress_file_id: Unique identifier for progress tracking

        Returns:
            Processing result with metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_id = file_id or file_path.stem

            logger.info(f"Processing document: {file_path}")

            # Step 1: Parse document into pages (with progress tracking)
            if progress_file_id:
                progress_service.update_progress(progress_file_id, 1, "Starting document parsing...")

            pages = parse_document(str(file_path), file_id=1, progress_file_id=progress_file_id)

            # Step 2: Chunk the document (with progress tracking)
            if progress_file_id:
                progress_service.update_progress(progress_file_id, 2, "Starting text chunking...")

            chunks = document_chunker.chunk_document_pages(pages, progress_file_id=progress_file_id)

            # Step 3: Store chunks and metadata
            if progress_file_id:
                progress_service.update_progress(progress_file_id, 3, "Storing processed data...")

            self.document_chunks[file_id] = chunks
            self.document_metadata[file_id] = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "num_pages": len(pages),
                "num_chunks": len(chunks),
                "total_tokens": sum(chunk.get("token_count", 0) for chunk in chunks)
            }

            # Save processed data
            self._save_processed_document(file_id, chunks, self.document_metadata[file_id])

            if progress_file_id:
                progress_service.update_progress(progress_file_id, 3, "Processing completed", 100)

            logger.info(f"Document processed successfully: {len(chunks)} chunks created")

            return {
                "success": True,
                "file_id": file_id,
                "metadata": self.document_metadata[file_id]
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            if progress_file_id:
                progress_service.set_error(progress_file_id, str(e))
            return {
                "success": False,
                "error": str(e)
            }

    def answer_question(self, question: str, file_id: str = None, max_chunks: int = 3) -> Dict:
        """
        Answer a question based on processed documents with speed optimizations

        Args:
            question: User question
            file_id: Specific document to search (None for all documents)
            max_chunks: Maximum number of chunks to include in context (reduced for speed)

        Returns:
            Answer with sources and metadata
        """
        try:
            if not mistral_llm.is_ready():
                mode = os.getenv("LLM_MODE", "local")
                provider = os.getenv("LLM_PROVIDER", "gemini")
                # Provide clearer guidance depending on mode
                if mode == "api":
                    hint = (
                        "API LLM not ready. Ensure LLM_API_KEY is set and provider/model names are valid. "
                        "Example (PowerShell): $env:LLM_MODE='api'; $env:LLM_PROVIDER='gemini'; $env:LLM_API_KEY='YOUR_KEY'; "
                        "$env:LLM_MODEL_NAME='gemini-2.5-flash' then restart server."
                    )
                else:
                    hint = (
                        "Local LLM not ready. Either place a .gguf model under models/llm/ or switch to API mode by setting "
                        "LLM_MODE=api and providing API credentials."
                    )
                return {"success": False, "error": f"LLM not ready (mode={mode}, provider={provider}). {hint}"}

            # Get relevant chunks with caching
            cache_key = f"{question[:50]}_{file_id}_{max_chunks}"
            if cache_key in self._chunks_cache:
                relevant_chunks = self._chunks_cache[cache_key]
            else:
                relevant_chunks = self._find_relevant_chunks(question, file_id, max_chunks)
                self._chunks_cache[cache_key] = relevant_chunks

            if not relevant_chunks:
                return {
                    "success": False,
                    "error": "No relevant documents found. Please upload and process documents first."
                }

            # Build optimized context for speed and completeness balance
            context = self._build_context(relevant_chunks, max_length=1000)

            # Generate answer with optimized settings
            answer = mistral_llm.answer_question(context, question)

            # Extract sources
            sources = self._extract_sources(relevant_chunks)

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "chunks_used": len(relevant_chunks),
                "question": question
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _find_relevant_chunks(self, question: str, file_id: str = None, max_chunks: int = 5) -> List[Dict]:
        """
        Find relevant chunks for the question
        For now, uses simple keyword matching. Can be enhanced with embeddings later.
        """
        all_chunks = []

        # Get chunks from specified document or all documents
        if file_id and file_id in self.document_chunks:
            all_chunks = self.document_chunks[file_id]
        else:
            for chunks in self.document_chunks.values():
                all_chunks.extend(chunks)

        if not all_chunks:
            return []

        # Simple keyword-based relevance scoring
        question_keywords = set(question.lower().split())

        scored_chunks = []
        for chunk in all_chunks:
            chunk_text = chunk["text"].lower()
            # Count keyword matches
            matches = sum(1 for keyword in question_keywords if keyword in chunk_text)
            # Add partial matches (substring matching)
            partial_matches = sum(1 for keyword in question_keywords 
                                if any(keyword in word for word in chunk_text.split()))

            total_score = matches * 2 + partial_matches  # Exact matches worth more

            chunk_with_score = chunk.copy()
            chunk_with_score["relevance_score"] = total_score
            scored_chunks.append(chunk_with_score)

        # Sort by relevance and return top chunks
        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

        # If no good matches, return the first few chunks anyway
        if not any(chunk["relevance_score"] > 0 for chunk in scored_chunks[:max_chunks]):
            logger.info("No keyword matches found, returning first chunks as fallback")
            return all_chunks[:max_chunks]

        return scored_chunks[:max_chunks]

    def _build_context(self, chunks: List[Dict], max_length: int = 1000) -> str:
        """Build context string from relevant chunks optimized for speed and completeness"""
        context_parts = []
        current_length = 0

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk["text"]
            page_info = f"Page {chunk.get('page', 'Unknown')}" if 'page' in chunk else ""

            # Optimize chunk size for speed while ensuring key information is preserved
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."

            chunk_entry = f"[Chunk {i}] {page_info}\n{chunk_text}\n"

            # Check if adding this chunk would exceed max_length
            if current_length + len(chunk_entry) > max_length:
                break

            context_parts.append(chunk_entry)
            current_length += len(chunk_entry)

        return "\n".join(context_parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks"""
        sources = []
        for chunk in chunks:
            file_id = chunk.get("file_id", "unknown")
            if file_id in self.document_metadata:
                metadata = self.document_metadata[file_id]
                sources.append({
                    "file_name": metadata["file_name"],
                    "page": chunk.get("page", 0),
                    "chunk_id": chunk.get("chunk_id", 0)
                })
        return sources

    def _save_processed_document(self, file_id: str, chunks: List[Dict], metadata: Dict):
        """Save processed document data"""
        try:
            output_file = self.processed_dir / f"{file_id}.json"
            data = {
                "metadata": metadata,
                "chunks": chunks
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving processed document: {e}")

    def _load_existing_documents(self):
        """Load all existing processed documents into memory"""
        try:
            for json_file in self.processed_dir.glob("*.json"):
                file_id = json_file.stem
                if self.load_processed_document(file_id):
                    logger.info(f"Loaded existing document: {file_id}")
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")

    def load_processed_document(self, file_id: str) -> bool:
        """Load previously processed document"""
        try:
            input_file = self.processed_dir / f"{file_id}.json"
            if not input_file.exists():
                return False

            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.document_chunks[file_id] = data["chunks"]
            self.document_metadata[file_id] = data["metadata"]

            logger.info(f"Loaded processed document: {file_id}")
            return True

        except Exception as e:
            logger.error(f"Error loading processed document: {e}")
            return False

    def list_documents(self) -> List[Dict]:
        """List all processed documents"""
        return [
            {
                "file_id": file_id,
                **metadata
            }
            for file_id, metadata in self.document_metadata.items()
        ]

    def get_document_info(self, file_id: str) -> Optional[Dict]:
        """Get information about a specific document"""
        if file_id in self.document_metadata:
            return {
                "file_id": file_id,
                **self.document_metadata[file_id],
                "chunks_count": len(self.document_chunks.get(file_id, []))
            }
        return None

# Global service instance
qa_service = DocumentQAService()