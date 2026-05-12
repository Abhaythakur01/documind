"""End-to-end retrieval probe: hybrid retrieve + LLM rerank.

Usage:
    python -m app.retrieval.smoke_retrieve "what BLEU score did the Transformer achieve on EN-DE?"
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import settings
from app.llm.groq_client import Role
from app.retrieval.bm25_store import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import VectorStore, get_qdrant_client


def _trim(s: str, n: int = 110) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--retrieve-k", type=int, default=settings.top_k_retrieve)
    parser.add_argument("--rerank-k", type=int, default=settings.top_k_rerank)
    parser.add_argument(
        "--smart",
        action="store_true",
        help="Use the 70B model for reranking (slower, higher quality).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the judge's raw reasoning output.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    embedder = Embedder()
    vs = VectorStore(get_qdrant_client(), settings.qdrant_collection, dim=embedder.dim)

    pkl_files = sorted(
        (settings.cache_dir / "bm25").glob("*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not pkl_files:
        print("No BM25 index found. Run build_index first.")
        return 1
    bm25 = BM25Index.load(pkl_files[0])
    retriever = HybridRetriever(vs, bm25, embedder)

    candidates = retriever.retrieve(args.query, top_k=args.retrieve_k)

    bar = "─" * 76
    print(f"\nQUERY: {args.query!r}")
    print(f"\n{bar}\n  HYBRID  (Dense + BM25, RRF-fused, top {args.retrieve_k})\n{bar}")
    for i, h in enumerate(candidates, 1):
        print(
            f"  {i:>2}. [{h['chunk_type']:>6} p{h['page']:>2}] "
            f"rrf={h['rrf_score']:.4f}  {_trim(h['content'])}"
        )

    role = Role.SMART if args.smart else Role.FAST
    final = rerank(args.query, candidates, top_k=args.rerank_k, model_role=role)
    judge_name = "llama-3.3-70b" if args.smart else "llama-3.1-8b"
    print(f"\n{bar}\n  RERANKED  (Groq {judge_name} judge, top {args.rerank_k})\n{bar}")
    for i, h in enumerate(final, 1):
        score = h["rerank_score"]
        score_str = f"{score:>2}/10" if score >= 0 else " — "
        print(
            f"  {i:>2}. [{h['chunk_type']:>6} p{h['page']:>2}] "
            f"score={score_str}  {_trim(h['content'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
