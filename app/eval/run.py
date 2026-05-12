"""End-to-end DocuMind evaluation.

Loads eval/test_set.jsonl, runs the chain on each question, then computes:
- **Page recall@K**: fraction of expected pages found in retrieved citations
- **Refusal accuracy**: did the system refuse correctly on unanswerable queries
- **Ragas** (faithfulness, answer relevancy, context precision/recall) via Groq judge

Outputs:
- eval/reports/eval_<timestamp>.json  (full data, machine-readable)
- eval/reports/eval_<timestamp>.md    (summary table for the README)

Usage:
    python -m app.eval.run                    # full 25-question run (slow on free tier)
    python -m app.eval.run --limit 5          # quick 5-question check
    python -m app.eval.run --no-ragas         # skip Ragas LLM-judge (much faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from app.chains.rag_chain import build_default_chain
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval")


def load_test_set(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_chain_predictions(chain, questions: list[dict]) -> list[dict]:
    predictions: list[dict] = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        result = chain.answer(q["question"])
        latency = time.perf_counter() - t0
        predictions.append(
            {
                **q,
                "answer": result["answer"],
                "contexts": [c["content"] for c in result["citations"]],
                "retrieved_pages": [c["page"] for c in result["citations"]],
                "citations": result["citations"],
                "latency_s": round(latency, 2),
            }
        )
        logger.info(
            "[%02d/%02d] %s (%s) latency=%.2fs",
            i, len(questions), q["id"], q["category"], latency,
        )
    return predictions


def compute_retrieval_metrics(predictions: list[dict]) -> dict:
    page_hits, page_total = 0, 0
    refusal_correct, refusal_total = 0, 0
    for p in predictions:
        if p["category"] == "refusal":
            refusal_total += 1
            if "do not contain enough information" in p["answer"].lower():
                refusal_correct += 1
        else:
            expected = set(p.get("expected_pages") or [])
            if not expected:
                continue
            retrieved = set(p["retrieved_pages"])
            page_hits += len(expected & retrieved)
            page_total += len(expected)
    return {
        "page_recall": (page_hits / page_total) if page_total else None,
        "refusal_accuracy": (refusal_correct / refusal_total) if refusal_total else None,
        "page_hits": page_hits,
        "page_total": page_total,
        "refusal_correct": refusal_correct,
        "refusal_total": refusal_total,
    }


class _BgeLCAdapter:
    """Expose our local Embedder under the LangChain Embeddings interface so
    Ragas can use it without reloading the model."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode_query(text)


def run_ragas(predictions: list[dict]) -> dict:
    """Score predictions with Ragas, using our Groq LLM + local bge embeddings.

    Refusal-category items are skipped — Ragas's faithfulness/relevancy metrics
    are undefined when there's no factual claim to grade.
    """
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    from app.retrieval.embeddings import Embedder

    eval_items = [p for p in predictions if p["category"] != "refusal"]
    if not eval_items:
        return {"averages": {}, "per_question": [], "n": 0}

    ds = Dataset.from_list(
        [
            {
                "question": p["question"],
                "answer": p["answer"],
                "contexts": p["contexts"] if p["contexts"] else [""],
                "ground_truth": p["ground_truth"],
            }
            for p in eval_items
        ]
    )

    llm = ChatGroq(
        model=settings.groq_model_smart,
        api_key=settings.groq_api_key,
        temperature=0.0,
    )
    embeddings = _BgeLCAdapter(Embedder())

    logger.info("Running Ragas on %d non-refusal items (this is the slow part)...", len(eval_items))
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(max_workers=2, timeout=180),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages: dict[str, float | None] = {}
    for m in metric_names:
        if m in df.columns:
            col = df[m].dropna()
            averages[m] = float(col.mean()) if len(col) else None
        else:
            averages[m] = None

    return {
        "averages": averages,
        "per_question": json.loads(df.to_json(orient="records")),
        "n": len(eval_items),
    }


def render_markdown_report(predictions, retrieval, ragas) -> str:
    L: list[str] = []
    L.append("# DocuMind — Evaluation Report\n")
    L.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    L.append("## Summary\n")
    L.append(f"- Test set: **{len(predictions)} questions**")
    cats: dict[str, int] = {}
    for p in predictions:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    L.append("- Distribution:")
    for cat, n in sorted(cats.items()):
        L.append(f"  - `{cat}`: {n}")
    avg_lat = sum(p["latency_s"] for p in predictions) / len(predictions)
    L.append(f"- Average end-to-end latency: **{avg_lat:.2f}s**\n")

    L.append("## Retrieval & Refusal (custom metrics)\n")
    L.append("| Metric | Score | Detail |")
    L.append("|---|---|---|")
    if retrieval["page_recall"] is not None:
        L.append(
            f"| Page Recall@top-K | **{retrieval['page_recall']:.1%}** | "
            f"{retrieval['page_hits']}/{retrieval['page_total']} expected pages retrieved |"
        )
    if retrieval["refusal_accuracy"] is not None:
        L.append(
            f"| Refusal Accuracy | **{retrieval['refusal_accuracy']:.1%}** | "
            f"{retrieval['refusal_correct']}/{retrieval['refusal_total']} unanswerable queries refused |"
        )
    L.append("")

    if ragas and ragas.get("averages"):
        L.append(f"## Ragas — LLM-judged ({ragas['n']} non-refusal items)\n")
        L.append("| Metric | Score |")
        L.append("|---|---|")
        for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            v = ragas["averages"].get(k)
            label = k.replace("_", " ").title()
            L.append(f"| {label} | {('**' + format(v, '.3f') + '**') if v is not None else '—'} |")
        L.append("")

    L.append("## Per-Question Results\n")
    L.append("| ID | Category | Question | Latency | Pages Hit |")
    L.append("|---|---|---|---|---|")
    for p in predictions:
        qs = p["question"]
        if len(qs) > 70:
            qs = qs[:67] + "…"
        expected = set(p.get("expected_pages") or [])
        retrieved = set(p["retrieved_pages"])
        if expected:
            pages_hit = f"{len(expected & retrieved)}/{len(expected)}"
        else:
            pages_hit = "—"
        L.append(f"| {p['id']} | {p['category']} | {qs} | {p['latency_s']:.1f}s | {pages_hit} |")

    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=Path, default=Path("eval/test_set.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-ragas", action="store_true")
    args = parser.parse_args(argv)

    if not args.test_set.exists():
        logger.error("Test set not found: %s", args.test_set)
        return 1

    questions = load_test_set(args.test_set)
    if args.limit:
        questions = questions[: args.limit]
    logger.info("Loaded %d questions from %s", len(questions), args.test_set)

    chain = build_default_chain()
    predictions = run_chain_predictions(chain, questions)

    retrieval = compute_retrieval_metrics(predictions)
    logger.info("Retrieval metrics: %s", retrieval)

    ragas = None
    if not args.no_ragas:
        try:
            ragas = run_ragas(predictions)
            logger.info("Ragas averages: %s", ragas.get("averages"))
        except Exception as exc:
            logger.error("Ragas eval failed (continuing with retrieval-only): %s", exc)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("eval/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"eval_{ts}.json"
    md_path = out_dir / f"eval_{ts}.md"

    # Trim non-serializable fields from predictions before writing
    clean_preds = []
    for p in predictions:
        cits = [
            {k: v for k, v in c.items() if k not in {"rrf_score"}}
            for c in p["citations"]
        ]
        clean_preds.append({**p, "citations": cits})

    json_path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "n_questions": len(predictions),
                "retrieval": retrieval,
                "ragas": ragas,
                "predictions": clean_preds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown_report(predictions, retrieval, ragas), encoding="utf-8")

    bar = "─" * 60
    print(f"\n{bar}\n ✓ Eval complete\n{bar}")
    print(f"  JSON   : {json_path}")
    print(f"  Report : {md_path}\n")
    if retrieval["page_recall"] is not None:
        print(f"  Page Recall@K    : {retrieval['page_recall']:.1%}")
    if retrieval["refusal_accuracy"] is not None:
        print(f"  Refusal Accuracy : {retrieval['refusal_accuracy']:.1%}")
    if ragas and ragas.get("averages"):
        print("\n  Ragas averages:")
        for k, v in ragas["averages"].items():
            print(f"    {k:>22}: {v:.3f}" if v is not None else f"    {k:>22}: —")
    return 0


if __name__ == "__main__":
    sys.exit(main())
