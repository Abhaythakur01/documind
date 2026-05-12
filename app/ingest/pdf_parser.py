"""Parse PDFs into text, tables, and figures with page + bbox metadata.

Strategy:
- Text + figures: PyMuPDF (fitz) — fast, no external dependencies.
- Tables: pdfplumber — best in class for structured table extraction.

Why both? PyMuPDF's table support is decent but pdfplumber consistently wins
on real-world technical PDFs. Splitting the work plays to each library's
strengths and keeps the table parser swappable.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from app.ingest.types import Chunk, ParsedDocument

logger = logging.getLogger(__name__)

_CAPTION_RE = re.compile(
    r"(Figure|Table|Fig\.?)\s*\d+[\.:]?\s*([^\n]{0,300})", re.IGNORECASE
)


def _doc_id_from_path(path: Path) -> str:
    """Stable doc_id derived from absolute path + size."""
    stat = path.stat()
    h = hashlib.sha256(f"{path.resolve()}::{stat.st_size}".encode()).hexdigest()[:12]
    return f"{path.stem}-{h}"


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """Convert a list-of-rows table into pipe-delimited markdown."""
    if not rows:
        return ""

    cleaned = [[(c or "").strip().replace("\n", " ") for c in row] for row in rows]
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    header, body = cleaned[0], cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _find_caption(text_above: str, text_below: str) -> str | None:
    """Look for 'Figure N: ...' or 'Table N: ...' near a figure."""
    for blob in (text_below, text_above):
        m = _CAPTION_RE.search(blob)
        if m:
            return m.group(0).strip()
    return None


def _extract_figures(doc: "fitz.Document", doc_id: str) -> list[Chunk]:
    figures: list[Chunk] = []
    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_rect = page.rect
        margin = 80

        for img_info in page.get_image_info(xrefs=True):
            bbox = img_info.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox

            # Skip tiny decorations / logos
            if (x1 - x0) < 50 or (y1 - y0) < 50:
                continue

            rect_above = fitz.Rect(0, max(0, y0 - margin), page_rect.width, y0)
            rect_below = fitz.Rect(
                0, y1, page_rect.width, min(page_rect.height, y1 + margin)
            )
            text_above = page.get_textbox(rect_above) or ""
            text_below = page.get_textbox(rect_below) or ""
            caption = _find_caption(text_above, text_below) or "Figure (no caption found)"

            figures.append(
                Chunk(
                    chunk_id=f"{doc_id}-fig-p{page_num}-{len(figures)}",
                    doc_id=doc_id,
                    page=page_num,
                    chunk_type="figure",
                    content=caption,
                    bbox=(x0, y0, x1, y1),
                    metadata={"xref": img_info.get("xref")},
                )
            )
    return figures


def _extract_tables(path: Path, doc_id: str) -> list[Chunk]:
    tables: list[Chunk] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_idx, plumber_page in enumerate(pdf.pages):
            page_num = page_idx + 1
            for t_idx, tbl in enumerate(plumber_page.find_tables() or []):
                rows = tbl.extract()
                if not rows or len(rows) < 2:
                    continue
                md = _table_to_markdown(rows)
                if not md.strip():
                    continue
                tables.append(
                    Chunk(
                        chunk_id=f"{doc_id}-tbl-p{page_num}-{t_idx}",
                        doc_id=doc_id,
                        page=page_num,
                        chunk_type="table",
                        content=md,
                        bbox=tuple(tbl.bbox) if tbl.bbox else None,
                        metadata={
                            "row_count": len(rows),
                            "col_count": max(len(r) for r in rows),
                        },
                    )
                )
    return tables


def parse_pdf(path: str | Path) -> ParsedDocument:
    """Extract raw text, tables, and figures from a PDF."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    doc_id = _doc_id_from_path(path)
    logger.info("Parsing %s (doc_id=%s)", path.name, doc_id)

    with fitz.open(str(path)) as doc:
        num_pages = doc.page_count
        page_texts = [doc[i].get_text("text") for i in range(num_pages)]
        figures = _extract_figures(doc, doc_id)

    tables = _extract_tables(path, doc_id)

    logger.info(
        "Parsed %s: %d pages, %d tables, %d figures",
        path.name,
        num_pages,
        len(tables),
        len(figures),
    )

    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(path),
        num_pages=num_pages,
        page_texts=page_texts,
        tables=tables,
        figures=figures,
    )
