"""End-to-end RAG smoke test: retrieve → rerank → synthesize with citations.

Usage:
    python -m app.chains.smoke_answer "what BLEU score did the Transformer achieve on EN-DE?"
    python -m app.chains.smoke_answer "what is multi-head attention?" --stream
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.chains.rag_chain import build_default_chain


def _trim(s: str, n: int = 100) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer token-by-token (like the future FastAPI endpoint).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show retrieval + rerank diagnostics.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    chain = build_default_chain()
    bar = "─" * 76

    if args.stream:
        print(f"\nQUERY: {args.query!r}\n")
        print(f"{bar}\n  ANSWER  (streaming)\n{bar}\n")
        citations: list[dict] = []
        for event in chain.stream(args.query):
            kind = event["event"]
            data = event["data"]
            if kind == "retrieve":
                if args.verbose:
                    print(f"[retrieve] {data['n_candidates']} candidates\n")
            elif kind == "rerank":
                if args.verbose:
                    print("[rerank] top picks:")
                    for c in data["citations"]:
                        print(
                            f"  [{c['n']}] p{c['page']} ({c['chunk_type']}) "
                            f"score={c['rerank_score']}/10"
                        )
                    print()
            elif kind == "token":
                print(data["text"], end="", flush=True)
            elif kind == "done":
                citations = data["citations"]
        print()  # final newline after streamed answer
    else:
        result = chain.answer(args.query)
        citations = result["citations"]
        print(f"\nQUERY: {args.query!r}\n")
        print(f"{bar}\n  ANSWER\n{bar}\n{result['answer']}\n")

    print(f"\n{bar}\n  CITATIONS\n{bar}")
    for i, c in enumerate(citations, start=1):
        print(
            f"  [{i}] page {c['page']:>2}  ({c['chunk_type']:>6})  "
            f"score={c.get('rerank_score', '-')}/10  {_trim(c['content'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
