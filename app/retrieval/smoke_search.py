"""Quick check that vector + BM25 indexes return sensible hits.

Usage:
    python -m app.retrieval.smoke_search "what is multi-head attention?"
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import settings
from app.retrieval.bm25_store import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore, get_qdrant_client

logging.basicConfig(level=logging.WARNING)


def _trim(s: str, n: int = 110) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    embedder = Embedder()
    qv = embedder.encode_query(args.query)
    store = VectorStore(get_qdrant_client(), settings.qdrant_collection, dim=embedder.dim)
    dense_hits = store.search(qv, top_k=args.top_k)

    bm25_dir = settings.cache_dir / "bm25"
    pkl_files = sorted(
        bm25_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not pkl_files:
        print("No BM25 index found. Run build_index first.")
        return 1
    bm25 = BM25Index.load(pkl_files[0])
    bm25_hits = bm25.search(args.query, top_k=args.top_k)

    bar = "─" * 76
    print(f"\nQUERY: {args.query!r}")
    print(f"\n{bar}\n  DENSE  (bge-large-en-v1.5 + Qdrant)\n{bar}")
    for i, h in enumerate(dense_hits, 1):
        print(
            f"  {i}. [{h['chunk_type']:>6} p{h['page']:>2}] "
            f"score={h['score']:.3f}  {_trim(h['content'])}"
        )

    print(f"\n{bar}\n  LEXICAL  (BM25)\n{bar}")
    for i, h in enumerate(bm25_hits, 1):
        print(
            f"  {i}. [{h['chunk_type']:>6} p{h['page']:>2}] "
            f"score={h['score']:.3f}  {_trim(h['content'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
