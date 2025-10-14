"""Backfill existing processed documents into the vector database.
Usage:
  python backfill_vector_db.py
Ensure environment variables for Qdrant are set and dependencies installed.
"""
import os
import json
from pathlib import Path
from app.services.vector_service import vector_service

# Load environment variables
VECTOR_DB = os.getenv("VECTOR_DB", "qdrant")
QDRANT_EMBEDDED = os.getenv("QDRANT_EMBEDDED", "1")
QDRANT_STORAGE = os.getenv("QDRANT_STORAGE", "./data/qdrant_storage")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "prism_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PROCESSED_DIR = Path("data/processed")

def iter_processed():
    for f in PROCESSED_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            yield f.stem, data.get("chunks", [])
        except Exception as e:
            print(f"Failed to read {f}: {e}")

def main():
    if not vector_service.is_ready():
        print("Vector service not ready. Set VECTOR_DB=qdrant and install dependencies.")
        return
    count_files = 0
    count_chunks = 0
    for file_id, chunks in iter_processed():
        if not chunks:
            continue
        vector_service.upsert_chunks(file_id, chunks)
        count_files += 1
        count_chunks += len(chunks)
    print(f"Backfill complete: {count_files} files, {count_chunks} chunks.")

if __name__ == "__main__":
    main()