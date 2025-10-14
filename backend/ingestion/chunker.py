"""
Text chunking utilities for document processing
"""
import re
from typing import List, Dict
import tiktoken
import sys
from pathlib import Path

# Add the backend directory to the Python path for progress service
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from app.services.progress_service import progress_service
except ImportError:
    # Fallback if progress service is not available
    progress_service = None

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize document chunker
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: Model name for tokenizer (fallback to approximate counting)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback to approximate word counting if tiktoken fails
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 0.75 words
            return int(len(text.split()) * 1.33)
    
    def chunk_text(self, text: str, file_id: int = 1) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            file_id: File identifier
            
        Returns:
            List of chunk dictionaries with file_id, chunk_id, text, and token_count
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    "file_id": file_id,
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "token_count": current_tokens
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_id += 1
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                "file_id": file_id,
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "token_count": current_tokens
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.,!?\-\(\)\[\]"\':]', ' ', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting on common punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """Get overlap text from end of chunk"""
        if not self.tokenizer:
            # Approximate word-based overlap
            words = text.split()
            overlap_words = int(max_tokens * 0.75)  # Approximate tokens to words
            return " ".join(words[-overlap_words:]) if len(words) > overlap_words else text
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        overlap_tokens = tokens[-max_tokens:]
        return self.tokenizer.decode(overlap_tokens)
    
    def chunk_document_pages(self, pages: List[Dict], progress_file_id: str = None) -> List[Dict]:
        """
        Chunk a document that's already split into pages
        
        Args:
            pages: List of page dictionaries from parse_pdf.py
            progress_file_id: File ID for progress tracking
            
        Returns:
            List of chunk dictionaries
        """
        if progress_service and progress_file_id:
            progress_service.update_progress(progress_file_id, 2, "Starting text chunking...")
        
        all_chunks = []
        total_pages = len(pages)
        
        for page_idx, page in enumerate(pages):
            if progress_service and progress_file_id:
                page_progress = (page_idx / total_pages) * 90  # 90% of step 2
                progress_service.update_progress(
                    progress_file_id, 
                    2, 
                    f"Chunking page {page_idx + 1} of {total_pages}", 
                    page_progress
                )
            
            page_chunks = self.chunk_text(page["text"], page["file_id"])
            # Add page information to chunks
            for chunk in page_chunks:
                chunk["page"] = page["page"]
                chunk["source_page"] = page["page"]
            all_chunks.extend(page_chunks)
        
        if progress_service and progress_file_id:
            progress_service.update_progress(
                progress_file_id, 
                2, 
                f"Chunking completed - {len(all_chunks)} chunks created", 
                100
            )
        
        return all_chunks

# Global chunker instance
document_chunker = DocumentChunker() 
