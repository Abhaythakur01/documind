"""Auto-bootstrap: build the indexes the chain needs if they don't already exist.

Locally you run `python -m app.ingest.run` and `python -m app.retrieval.build_index`
once. On a fresh deploy (Hugging Face Space, fresh clone, etc.) those artifacts
aren't there yet — so this module rebuilds them at Streamlit startup, idempotently.

The first launch takes ~45s on a 2-vCPU box; subsequent launches see the cached
indexes and return immediately.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import streamlit as st

from app.config import settings

DEMO_PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"
DEMO_PDF_NAME = "attention.pdf"
from app.ingest.chunker import chunk_document
from app.ingest.pdf_parser import parse_pdf
from app.ingest.types import Chunk  # noqa: F401  (re-exported for clarity)
from app.retrieval.bm25_store import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore, get_qdrant_client

logger = logging.getLogger(__name__)


def _has_indexes() -> bool:
    chunks_dir = settings.cache_dir / "chunks"
    bm25_dir = settings.cache_dir / "bm25"
    return any(chunks_dir.glob("*.json")) and any(bm25_dir.glob("*.pkl"))


def _find_pdf() -> Path | None:
    pdfs = sorted(settings.pdf_dir.glob("*.pdf"))
    if pdfs:
        return pdfs[0]
    return _download_demo_pdf()


def _download_demo_pdf() -> Path | None:
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    target = settings.pdf_dir / DEMO_PDF_NAME
    try:
        with st.spinner(f"Fetching demo paper from arxiv ({DEMO_PDF_NAME}, ~2 MB)…"):
            urllib.request.urlretrieve(DEMO_PDF_URL, target)
        logger.info("Downloaded demo PDF to %s", target)
        return target
    except Exception as exc:
        logger.exception("Failed to download demo PDF")
        st.error(f"Could not fetch demo PDF from {DEMO_PDF_URL}: {exc}")
        return None


@st.cache_resource(show_spinner="Bootstrapping indexes (one-time, ~45s)…")
def ensure_indexes() -> None:
    """Build chunks + vector index + BM25 index from the first PDF if missing."""
    if _has_indexes():
        logger.info("Indexes already present; skipping bootstrap.")
        return

    pdf = _find_pdf()
    if pdf is None:
        st.error(
            "No PDF found in `data/pdfs/`. Add one to the repo (and redeploy) "
            "or run `python -m app.ingest.run data/pdfs/your.pdf` locally."
        )
        st.stop()

    logger.info("Bootstrap: parsing %s", pdf.name)
    parsed = parse_pdf(pdf)
    chunks = chunk_document(parsed)

    chunks_dir = settings.cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / f"{parsed.doc_id}.json").write_text(
        json.dumps(
            {
                "doc_id": parsed.doc_id,
                "source_path": str(pdf),
                "num_pages": parsed.num_pages,
                "chunks": [c.to_dict() for c in chunks],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    embedder = Embedder()
    vectors = embedder.encode([c.content for c in chunks], batch_size=16)
    store = VectorStore(get_qdrant_client(), settings.qdrant_collection, dim=embedder.dim)
    store.recreate_collection()
    store.upsert_chunks(chunks, vectors)

    bm25_dir = settings.cache_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    BM25Index(chunks).save(bm25_dir / f"{parsed.doc_id}.pkl")

    logger.info(
        "Bootstrap complete: %d pages, %d chunks (text=%d, tables=%d, figures=%d)",
        parsed.num_pages,
        len(chunks),
        sum(1 for c in chunks if c.chunk_type == "text"),
        sum(1 for c in chunks if c.chunk_type == "table"),
        sum(1 for c in chunks if c.chunk_type == "figure"),
    )
