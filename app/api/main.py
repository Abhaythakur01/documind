"""DocuMind FastAPI server with SSE streaming.

Endpoints:
- GET  /health         — liveness probe
- POST /query          — synchronous: returns full answer + citations
- POST /query/stream   — Server-Sent Events: streams retrieve → rerank → tokens → done
- GET  /docs           — Swagger UI (auto)

Design call: we load the chain ONCE at startup via FastAPI's lifespan context.
The embedding model + BM25 index + Qdrant client are heavy; loading per-request
would dominate latency.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import AnswerResponse, CitationOut, QueryRequest
from app.chains.rag_chain import DocuMindChain, build_default_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("documind.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading DocuMind chain (embedder + Qdrant + BM25)...")
    app.state.chain = build_default_chain()
    logger.info("Chain ready. API is up.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="DocuMind API",
    description="Multi-modal document intelligence over PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _citation_to_out(idx: int, c: dict) -> CitationOut:
    return CitationOut(
        n=idx + 1,
        chunk_id=c["chunk_id"],
        doc_id=c["doc_id"],
        page=c["page"],
        chunk_type=c["chunk_type"],
        content=c["content"],
        bbox=c.get("bbox"),
        rerank_score=int(c.get("rerank_score", -1)),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/query", response_model=AnswerResponse)
def query(req: QueryRequest) -> AnswerResponse:
    chain: DocuMindChain = app.state.chain
    result = chain.answer(req.query, doc_id=req.doc_id)
    return AnswerResponse(
        query=result["query"],
        answer=result["answer"],
        citations=[_citation_to_out(i, c) for i, c in enumerate(result["citations"])],
    )


def _serialize_event(event: dict) -> dict:
    """Convert a chain.stream() event into JSON-serializable SSE form."""
    data = event["data"]
    kind = event["event"]

    if kind == "done":
        data = {
            "answer": data["answer"],
            "citations": [
                _citation_to_out(i, c).model_dump()
                for i, c in enumerate(data["citations"])
            ],
        }
    elif kind == "rerank":
        # 'citations' here is a lighter preview list; serialize as-is
        data = {"citations": data["citations"]}

    return {"event": kind, "data": json.dumps(data)}


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    chain: DocuMindChain = app.state.chain

    def event_generator():
        for event in chain.stream(req.query, doc_id=req.doc_id):
            yield _serialize_event(event)

    return EventSourceResponse(event_generator())
