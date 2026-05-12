"""Lightweight BM25 lexical index over chunk content.

Why hybrid retrieval? Dense embeddings miss rare terms (model names, equation
labels, table cell values). BM25 picks them up cheaply. Fuse the two
ranked lists and you get both semantic and lexical recall.

Tokenization is intentionally naive (regex word-split, lowercased). For a
production system we'd swap in spaCy or a proper analyzer; for portfolio
purposes this is faster and good enough.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.ingest.types import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized = [_tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def search(self, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            {**self.chunks[i].to_dict(), "score": float(scores[i])}
            for i in ranked_idx
            if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "tokenized": self.tokenized,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.chunks = [Chunk.from_dict(d) for d in data["chunks"]]
        instance.tokenized = data["tokenized"]
        instance.bm25 = BM25Okapi(instance.tokenized) if instance.tokenized else None
        return instance
