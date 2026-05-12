"""CLI: parse a PDF and save its chunks to disk.

Usage:
    python -m app.ingest.run data/pdfs/attention.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import settings
from app.ingest.chunker import chunk_document
from app.ingest.pdf_parser import parse_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a PDF into DocuMind.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.cache_dir / "chunks",
        help="Output directory for chunk JSON.",
    )
    args = parser.parse_args(argv)

    if not args.pdf_path.exists():
        logger.error("PDF not found: %s", args.pdf_path)
        return 1

    parsed = parse_pdf(args.pdf_path)
    chunks = chunk_document(parsed)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{parsed.doc_id}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "doc_id": parsed.doc_id,
                "source_path": parsed.source_path,
                "num_pages": parsed.num_pages,
                "chunks": [c.to_dict() for c in chunks],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    n_text = sum(1 for c in chunks if c.chunk_type == "text")
    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    n_figure = sum(1 for c in chunks if c.chunk_type == "figure")

    bar = "─" * 60
    print(f"\n{bar}")
    print(f" ✓ Ingested {args.pdf_path.name}")
    print(bar)
    print(f"  doc_id:        {parsed.doc_id}")
    print(f"  pages:         {parsed.num_pages}")
    print(f"  text chunks:   {n_text}")
    print(f"  tables:        {n_table}")
    print(f"  figures:       {n_figure}")
    print(f"  total chunks:  {len(chunks)}")
    print(f"  saved to:      {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
