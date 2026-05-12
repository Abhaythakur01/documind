"""Pydantic request/response schemas for the DocuMind HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    doc_id: str | None = Field(
        default=None,
        description="Restrict retrieval to this document. Omit to search all indexed docs.",
    )


class CitationOut(BaseModel):
    n: int  # citation number as it appears inline ([1], [2], ...)
    chunk_id: str
    doc_id: str
    page: int
    chunk_type: str
    content: str
    bbox: list[float] | None = None
    rerank_score: int = -1


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationOut]
