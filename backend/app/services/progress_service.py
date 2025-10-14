"""
Progress tracking service for document processing
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass
import threading

@dataclass
class ProcessingProgress:
    file_id: str
    status: str  # 'starting', 'parsing', 'chunking', 'completed', 'error'
    progress: float  # 0-100
    current_step: str
    total_steps: int
    current_step_number: int
    start_time: float
    estimated_remaining: Optional[float] = None
    error_message: Optional[str] = None

class ProgressService:
    """Service to track document processing progress"""
    
    def __init__(self):
        self._progress_store: Dict[str, ProcessingProgress] = {}
        self._lock = threading.Lock()
    
    def start_processing(self, file_id: str, total_steps: int = 4) -> None:
        """Start tracking progress for a file"""
        with self._lock:
            self._progress_store[file_id] = ProcessingProgress(
                file_id=file_id,
                status='starting',
                progress=0.0,
                current_step='Initializing processing...',
                total_steps=total_steps,
                current_step_number=0,
                start_time=time.time()
            )
    
    def update_progress(self, file_id: str, step_number: int, step_name: str, 
                       progress_within_step: float = 0.0) -> None:
        """Update progress for a file"""
        with self._lock:
            if file_id not in self._progress_store:
                return
            
            progress_item = self._progress_store[file_id]
            progress_item.current_step_number = step_number
            progress_item.current_step = step_name
            
            # Calculate overall progress
            base_progress = (step_number / progress_item.total_steps) * 100
            step_progress = (progress_within_step / progress_item.total_steps) * 100
            progress_item.progress = min(100.0, base_progress + step_progress)
            
            # Update status based on step
            if step_number == 1:
                progress_item.status = 'parsing'
            elif step_number == 2:
                progress_item.status = 'chunking'
            elif step_number == 3:
                progress_item.status = 'storing'
            elif step_number >= progress_item.total_steps:
                progress_item.status = 'completed'
                progress_item.progress = 100.0
            
            # Estimate remaining time
            elapsed = time.time() - progress_item.start_time
            if progress_item.progress > 0:
                total_estimated = elapsed * (100 / progress_item.progress)
                progress_item.estimated_remaining = max(0, total_estimated - elapsed)
    
    def set_error(self, file_id: str, error_message: str) -> None:
        """Set error status for a file"""
        with self._lock:
            if file_id in self._progress_store:
                progress_item = self._progress_store[file_id]
                progress_item.status = 'error'
                progress_item.error_message = error_message
    
    def complete_processing(self, file_id: str) -> None:
        """Mark processing as completed"""
        with self._lock:
            if file_id in self._progress_store:
                progress_item = self._progress_store[file_id]
                progress_item.status = 'completed'
                progress_item.progress = 100.0
                progress_item.current_step = 'Processing completed'
                progress_item.current_step_number = progress_item.total_steps
    
    def get_progress(self, file_id: str) -> Optional[ProcessingProgress]:
        """Get current progress for a file"""
        with self._lock:
            return self._progress_store.get(file_id)
    
    def cleanup_progress(self, file_id: str, delay: float = 300) -> None:
        """Clean up progress data after delay (5 minutes default)"""
        def cleanup():
            time.sleep(delay)
            with self._lock:
                self._progress_store.pop(file_id, None)
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

# Global progress service instance
progress_service = ProgressService()