---
title: DocuMind
emoji: 📄
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: "1.38.0"
app_file: ui/main.py
pinned: false
license: mit
short_description: "Multi-modal RAG over PDFs: hybrid retrieval, cited answers"
---

# DocuMind — Multi-Modal Document Intelligence

> Most RAG bots fail on tables and charts. This one doesn't.

A production-grade retrieval-augmented generation system that ingests PDFs with rich layout — text, tables, and figures — and answers questions with verifiable citations down to the page and bounding box.

## Why this exists

Standard RAG pipelines flatten PDFs into plain text and lose 30–50% of the information density. Scientific papers, technical reports, and financial filings encode their most important content in tables and figures. DocuMind treats each modality as a first-class retrieval target.

## Highlights

- **Multi-modal ingest** — tables extracted as structured HTML, figures captioned by a vision model
- **Hybrid retrieval** — dense embeddings (BAAI/bge-large) + BM25 lexical + table-aware indexing, fused and reranked by an LLM-as-judge
- **Cited answers** — every claim links to a page + bounding box, click-through to highlight in the PDF viewer
- **Production-grade** — FastAPI with SSE streaming, Dockerized, Qdrant for vectors, Ragas for offline eval, LangSmith for online tracing
- **Free-tier-friendly** — runs entirely on Groq's free API by routing 70B for synthesis and 8B for sub-roles, with disk-cached repeats

## Architecture

_Diagram added in the polish phase._

## Stack

| Layer | Tool |
|---|---|
| LLM serving | Groq (llama-3.3-70b + llama-3.1-8b) |
| Embeddings | sentence-transformers (BAAI/bge-large-en-v1.5, local) |
| Vector store | Qdrant |
| Lexical | rank-bm25 |
| Document parse | unstructured.io |
| Orchestration | LangChain + LangGraph |
| API | FastAPI + SSE |
| UI | Streamlit with PDF viewer |
| Eval | Ragas |
| Observability | LangSmith |
| Cache | diskcache |
| Container | Docker + docker-compose |

## Eval results

_Populated after the eval phase._

| Metric | Score |
|---|---|
| Faithfulness | — |
| Answer Relevance | — |
| Context Precision | — |
| Context Recall | — |

## Quickstart

```bash
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)

# 3. Start services
docker-compose up -d              # spins up Qdrant

# 4. Ingest a PDF
python -m app.ingest.run data/pdfs/attention.pdf

# 5. Run the app
uvicorn app.api.main:app --reload  # backend, port 8000
streamlit run ui/main.py           # frontend, port 8501
```

## Project structure

```
langchain_project_01/
├── app/
│   ├── api/           FastAPI routes (streaming SSE)
│   ├── ingest/        PDF → chunks/tables/figures pipeline
│   ├── retrieval/     hybrid retriever, reranker
│   ├── chains/        LangChain/LangGraph composition
│   ├── llm/           Groq client with retry + rate limiting
│   └── eval/          Ragas eval harness
├── ui/                Streamlit UI with PDF viewer
├── data/
│   ├── pdfs/          source documents
│   └── qdrant/        local vector store
├── eval/
│   ├── test_set.jsonl ~30 hand-labeled Q&A pairs
│   └── reports/       eval run outputs
├── docker-compose.yml
└── Dockerfile
```

## Engineering notes

_Expanded after the build is complete._

- Free Groq tier rate-limit handling (token-bucket + tenacity)
- Why hybrid retrieval beats dense-only on technical PDFs
- The reranker's measured effect on top-k quality
- Eval methodology and ablations

## License

MIT
