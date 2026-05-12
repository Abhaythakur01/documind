"""Hybrid retriever: fuses dense and lexical rankings via Reciprocal Rank Fusion.

Why RRF and not weighted sums of raw scores?
- Dense and BM25 scores live on incompatible scales (cosine in [0,1] vs BM25
  unbounded). Normalizing them to add is fragile.
- RRF only looks at *ranks*, so the math is robust regardless of score
  distribution. Empirically it beats hand-tuned linear fusion on most
  benchmarks (Cormack et al. 2009).

Constant k=60 follows the original paper. Tuning it has marginal effect; what
matters is that it's high enough to dampen the contribution of unmatched docs.
"""

from __future__ import annotations

import logging
from typing import Any

from app.retrieval.bm25_store import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = 60,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists into one using RRF."""
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for ranking in ranked_lists:
        for rank, item in enumerate(ranking):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            items[cid] = item

    ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:top_k]
    return [{**items[c], "rrf_score": scores[c]} for c in ranked]


class HybridRetriever:
    """Dense (Qdrant) + Lexical (BM25), fused with RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 20,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        qv = self.embedder.encode_query(query)
        dense = self.vector_store.search(qv, top_k=top_k, doc_id=doc_id)
        lex = self.bm25.search(query, top_k=top_k)
        fused = reciprocal_rank_fusion([dense, lex], top_k=top_k)
        logger.debug(
            "Hybrid retrieve: dense=%d, bm25=%d, fused=%d",
            len(dense),
            len(lex),
            len(fused),
        )
        return fused
