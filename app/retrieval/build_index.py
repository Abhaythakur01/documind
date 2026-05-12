"""CLI: build vector + BM25 indexes from saved chunks JSON.

Usage:
    python -m app.retrieval.build_index data/cache/chunks/attention-XXXX.json
    python -m app.retrieval.build_index data/cache/chunks/attention-XXXX.json --reset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import settings
from app.ingest.types import Chunk
from app.retrieval.bm25_store import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore, get_qdrant_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build vector + BM25 indexes from chunks JSON."
    )
    parser.add_argument("chunks_path", type=Path)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe and recreate the Qdrant collection before indexing.",
    )
    args = parser.parse_args(argv)

    if not args.chunks_path.exists():
        logger.error("Chunks file not found: %s", args.chunks_path)
        return 1

    with args.chunks_path.open(encoding="utf-8") as f:
        data = json.load(f)

    chunks = [Chunk.from_dict(c) for c in data["chunks"]]
    doc_id = data["doc_id"]
    logger.info("Loaded %d chunks for doc_id=%s", len(chunks), doc_id)

    embedder = Embedder()
    texts = [c.content for c in chunks]
    logger.info(
        "Embedding %d chunks with %s (dim=%d)",
        len(texts),
        embedder.model_name,
        embedder.dim,
    )
    vectors = embedder.encode(texts, batch_size=16)

    client = get_qdrant_client()
    store = VectorStore(client, settings.qdrant_collection, dim=embedder.dim)
    if args.reset:
        store.recreate_collection()
    else:
        store.ensure_collection()
    store.upsert_chunks(chunks, vectors)
    logger.info(
        "Upserted %d vectors to Qdrant collection '%s'",
        len(vectors),
        settings.qdrant_collection,
    )

    bm25_path = settings.cache_dir / "bm25" / f"{doc_id}.pkl"
    BM25Index(chunks).save(bm25_path)
    logger.info("Saved BM25 index to %s", bm25_path)

    bar = "─" * 60
    print(f"\n{bar}")
    print(f" ✓ Indexed doc_id={doc_id}")
    print(bar)
    print(f"  vectors:    {len(vectors)} in Qdrant collection '{settings.qdrant_collection}'")
    print(f"  bm25:       {bm25_path}")
    print(f"  embed dim:  {embedder.dim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
