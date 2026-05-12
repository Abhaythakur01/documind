"""Shared data structures for the ingest pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChunkType = Literal["text", "table", "figure"]


@dataclass
class Chunk:
    """A single retrievable unit extracted from a document."""

    chunk_id: str
    doc_id: str
    page: int  # 1-indexed
    chunk_type: ChunkType
    content: str  # raw text, markdown table, or figure caption
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page": self.page,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "bbox": list(self.bbox) if self.bbox else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        bbox = tuple(data["bbox"]) if data.get("bbox") else None
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            page=data["page"],
            chunk_type=data["chunk_type"],
            content=data["content"],
            bbox=bbox,  # type: ignore[arg-type]
            metadata=data.get("metadata", {}),
        )


@dataclass
class ParsedDocument:
    """A document after raw extraction, before chunking."""

    doc_id: str
    source_path: str
    num_pages: int
    page_texts: list[str]  # raw text per page (page n = page_texts[n-1])
    tables: list[Chunk]
    figures: list[Chunk]
