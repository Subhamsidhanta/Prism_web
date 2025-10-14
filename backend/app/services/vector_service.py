"""Vector database service abstraction for Prism.
Supports Qdrant for storing document chunk embeddings and performing semantic search.
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except ImportError:
    QdrantClient = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class VectorService:
    def __init__(self):
        self.enabled = os.getenv("VECTOR_DB", "none").lower() == "qdrant"
        self.collection_name = os.getenv("QDRANT_COLLECTION", "prism_chunks")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.embed_model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        # Embedded (in-process) Qdrant mode. Set QDRANT_EMBEDDED=1 to use local storage instead of a running service.
        self.embedded = os.getenv("QDRANT_EMBEDDED", "0").lower() in ("1", "true", "yes")
        self.storage_path = os.getenv("QDRANT_STORAGE", "./data/qdrant_storage")
        # Use Any to avoid type errors when packages not yet installed locally
        self.client: Optional[Any] = None
        self.embed_model: Optional[Any] = None
        self.vector_size: Optional[int] = None
        if self.enabled:
            self._init()

    def _init(self):
        if QdrantClient is None or SentenceTransformer is None:
            logger.error("Vector dependencies missing (qdrant-client or sentence-transformers).")
            self.enabled = False
            return
        try:
            if self.embedded:
                os.makedirs(self.storage_path, exist_ok=True)
                # Use local file-based storage (single-process). For ephemeral, path=":memory:" could be used.
                self.client = QdrantClient(path=self.storage_path)
                logger.info(f"Qdrant embedded mode enabled at {self.storage_path}")
            else:
                if self.qdrant_api_key:
                    self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
                else:
                    # For internal docker network use host name 'qdrant'
                    self.client = QdrantClient(url=self.qdrant_url or "http://localhost:6333")
            self.embed_model = SentenceTransformer(self.embed_model_name)
            # Infer embedding dimension
            test_vec = self.embed_model.encode(["test"])[0]
            self.vector_size = len(test_vec)
            self._ensure_collection()
            logger.info(f"VectorService initialized: collection={self.collection_name} dim={self.vector_size}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorService: {e}")
            self.enabled = False

    def _ensure_collection(self):
        assert self.client and self.vector_size
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def is_ready(self) -> bool:
        return self.enabled and self.client is not None and self.embed_model is not None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.is_ready():
            raise RuntimeError("VectorService not ready")
        return self.embed_model.encode(texts).tolist()  # type: ignore

    def upsert_chunks(self, file_id: str, chunks: List[Dict]):
        if not self.is_ready():
            logger.warning("VectorService disabled; skipping upsert.")
            return
        vectors = self.embed_texts([c["text"] for c in chunks])
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=f"{file_id}-{chunk.get('chunk_id', idx)}",
                vector=vector,
                payload={
                    "file_id": file_id,
                    "chunk_id": chunk.get("chunk_id", idx),
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                }
            ))
        self.client.upsert(collection_name=self.collection_name, points=points)  # type: ignore
        logger.info(f"Upserted {len(points)} chunks into vector DB for file {file_id}")

    def query(self, question: str, top_k: int = 5, file_id: Optional[str] = None) -> List[Dict]:
        if not self.is_ready():
            logger.warning("VectorService disabled; returning empty query result.")
            return []
        q_vec = self.embed_texts([question])[0]
        # Filter by file_id if provided
        query_filter = None
        if file_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))])
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            limit=top_k,
            query_filter=query_filter
        )  # type: ignore
        out = []
        for r in results:
            payload = r.payload
            out.append({
                "file_id": payload.get("file_id"),
                "chunk_id": payload.get("chunk_id"),
                "page": payload.get("page"),
                "text": payload.get("text"),
                "score": r.score,
            })
        return out


# Global instance
vector_service = VectorService()
