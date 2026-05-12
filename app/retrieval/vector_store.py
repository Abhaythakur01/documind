"""Qdrant vector store wrapper.

Operates in two modes, auto-detected:
- Server mode: if QDRANT_URL is set and reachable (Docker container running).
- Local mode: disk-based Qdrant under data/qdrant/ — zero-config, no Docker.

Local mode is the default for portfolio dev. Switching to server mode is a
one-line .env change + `docker-compose up qdrant`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.ingest.types import Chunk

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client. Try server URL first, fall back to local disk."""
    url = settings.qdrant_url
    if url and url.startswith(("http://", "https://")):
        try:
            client = QdrantClient(url=url, timeout=2, check_compatibility=False)
            client.get_collections()  # ping
            logger.info("Connected to Qdrant server at %s", url)
            return client
        except Exception as exc:
            logger.warning(
                "Qdrant server at %s unreachable (%s) — falling back to local mode",
                url,
                exc,
            )

    local_path = settings.data_dir / "qdrant"
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info("Using local Qdrant at %s", local_path)
    return QdrantClient(path=str(local_path))


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id (Qdrant requires UUID or int IDs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    def ensure_collection(self) -> None:
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, cosine)",
                self.collection,
                self.dim,
            )

    def recreate_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        logger.info("Recreated Qdrant collection '%s'", self.collection)

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            PointStruct(
                id=chunk_id_to_uuid(c.chunk_id),
                vector=v,
                payload=c.to_dict(),
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 20,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        filt = None
        if doc_id:
            filt = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        # query_points() replaced the deprecated search() in qdrant-client 1.12+
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=filt,
            with_payload=True,
        )
        return [{**p.payload, "score": float(p.score)} for p in response.points]
