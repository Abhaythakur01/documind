"""Local sentence-transformers embeddings (Groq doesn't host embeddings).

We use BAAI/bge-large-en-v1.5 — strong on MTEB, 1024-dim, runs comfortably on
CPU. Embeddings are L2-normalized so cosine similarity reduces to dot product.

Note: bge-v1.5 has asymmetric query/passage encoding baked in, but the README
recommends a query-side instruction prefix for retrieval. We apply it via
`encode_query`. Passages are encoded with plain `encode`.
"""

from __future__ import annotations

import logging
from functools import cached_property

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Thin sentence-transformers wrapper with lazy model loading."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model

    @cached_property
    def model(self) -> SentenceTransformer:
        logger.info("Loading embedding model: %s", self.model_name)
        return SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        # sentence-transformers 3.x renamed this method; support both.
        getter = getattr(
            self.model, "get_embedding_dimension", None
        ) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def encode(self, texts: list[str], *, batch_size: int = 16) -> list[list[float]]:
        """Encode passages (no instruction prefix)."""
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a query (with bge instruction prefix)."""
        text = f"{_BGE_QUERY_INSTRUCTION}{query}"
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()
