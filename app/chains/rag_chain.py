"""The DocuMind RAG chain: retrieve → rerank → synthesize.

Exposes:
- DocuMindChain.answer(query) → {answer, citations}
- DocuMindChain.stream(query)  → iterator of structured events
- build_default_chain() → preconfigured for the demo doc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from app.chains.synthesizer import stream_synthesize, synthesize
from app.config import settings
from app.llm.groq_client import Role
from app.retrieval.bm25_store import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import VectorStore, get_qdrant_client


@dataclass
class ChainConfig:
    retrieve_k: int = field(default_factory=lambda: settings.top_k_retrieve)
    rerank_k: int = field(default_factory=lambda: settings.top_k_rerank)
    judge_role: Role = Role.FAST
    synth_role: Role = Role.SMART


class DocuMindChain:
    def __init__(
        self,
        retriever: HybridRetriever,
        config: ChainConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or ChainConfig()

    def answer(
        self, query: str, *, doc_id: str | None = None
    ) -> dict[str, Any]:
        candidates = self.retriever.retrieve(
            query, top_k=self.config.retrieve_k, doc_id=doc_id
        )
        reranked = rerank(
            query,
            candidates,
            top_k=self.config.rerank_k,
            model_role=self.config.judge_role,
        )
        result = synthesize(
            query, reranked, model_role=self.config.synth_role
        )
        return {
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "candidates": candidates,
        }

    def stream(
        self, query: str, *, doc_id: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Stream structured events: retrieve → rerank → tokens → done.

        Event shape: {"event": str, "data": dict}.
        Consumers can render the answer progressively and reveal citations at
        end-of-stream.
        """
        candidates = self.retriever.retrieve(
            query, top_k=self.config.retrieve_k, doc_id=doc_id
        )
        yield {"event": "retrieve", "data": {"n_candidates": len(candidates)}}

        reranked = rerank(
            query,
            candidates,
            top_k=self.config.rerank_k,
            model_role=self.config.judge_role,
        )
        yield {
            "event": "rerank",
            "data": {
                "citations": [
                    {
                        "n": i + 1,
                        "chunk_id": c["chunk_id"],
                        "page": c["page"],
                        "chunk_type": c["chunk_type"],
                        "rerank_score": c.get("rerank_score", -1),
                    }
                    for i, c in enumerate(reranked)
                ]
            },
        }

        full_answer_parts: list[str] = []
        for token in stream_synthesize(
            query, reranked, model_role=self.config.synth_role
        ):
            full_answer_parts.append(token)
            yield {"event": "token", "data": {"text": token}}

        yield {
            "event": "done",
            "data": {
                "answer": "".join(full_answer_parts),
                "citations": reranked,
            },
        }


def build_default_chain() -> DocuMindChain:
    """Wire up the retriever + chain using the most recently built BM25 index."""
    embedder = Embedder()
    vs = VectorStore(
        get_qdrant_client(),
        settings.qdrant_collection,
        dim=embedder.dim,
    )

    bm25_dir = settings.cache_dir / "bm25"
    pkl_files = sorted(
        bm25_dir.glob("*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not pkl_files:
        raise FileNotFoundError(
            f"No BM25 index found in {bm25_dir}. Run `python -m app.retrieval.build_index ...` first."
        )
    bm25 = BM25Index.load(pkl_files[0])

    retriever = HybridRetriever(vs, bm25, embedder)
    return DocuMindChain(retriever)
