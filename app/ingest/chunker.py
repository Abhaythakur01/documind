"""Chunk parsed text into retrievable units with overlap.

Text is chunked per-page so every chunk has a definite page number.
Tables and figures are already self-contained chunks; they pass through.
"""

from __future__ import annotations

import hashlib
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest.types import Chunk, ParsedDocument

logger = logging.getLogger(__name__)


def _text_chunk_id(doc_id: str, page: int, idx: int, text: str) -> str:
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return f"{doc_id}-txt-p{page}-{idx}-{h}"


def chunk_document(
    parsed: ParsedDocument,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Return text chunks + extracted tables + extracted figures."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )

    text_chunks: list[Chunk] = []
    for page_idx, page_text in enumerate(parsed.page_texts):
        page_num = page_idx + 1
        if not page_text.strip():
            continue
        for i, piece in enumerate(splitter.split_text(page_text)):
            piece = piece.strip()
            if len(piece) < 50:  # discard tiny fragments
                continue
            text_chunks.append(
                Chunk(
                    chunk_id=_text_chunk_id(parsed.doc_id, page_num, i, piece),
                    doc_id=parsed.doc_id,
                    page=page_num,
                    chunk_type="text",
                    content=piece,
                )
            )

    all_chunks = text_chunks + parsed.tables + parsed.figures
    logger.info(
        "Chunked doc_id=%s: %d text + %d tables + %d figures = %d total",
        parsed.doc_id,
        len(text_chunks),
        len(parsed.tables),
        len(parsed.figures),
        len(all_chunks),
    )
    return all_chunks
