"""LLM-as-judge reranker.

Why an LLM reranker on top of hybrid retrieval?
- RRF gives good *recall* — the answer is almost always in the top 20.
- But the top 1-3 are often wrong: passages sharing vocabulary with the query
  outrank passages that actually answer it.
- A small LLM (Groq llama-3.1-8b) scores relevance accurately *if* you force
  chain-of-thought + strict anti-examples in the prompt. Without that, small
  judges anchor on vocabulary overlap and over-grant 10s.

Why not a cross-encoder model (ms-marco-MiniLM, etc.)?
- Stronger in absolute terms — but adds another local model download (~150MB)
  and inference path. The 8B judge gives ~90% of the benefit with zero extra
  infra and reuses our existing Groq budget.

Failure mode handled: if the judge's output is unparseable, we fall back to
RRF order so the chain still produces an answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_client import Role, get_llm, invoke_with_retry

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = """You are a STRICT retrieval relevance judge.

You receive a user query and N numbered candidate passages. Score EACH AND EVERY candidate from 0 to 10 by how directly it answers the query.

YOU MUST OUTPUT EXACTLY ONE LINE PER CANDIDATE — do NOT skip any. Do NOT add preamble, summary, or anything outside the per-candidate lines.

LINE FORMAT (exact):
<index>: <score>/10 - <≤12-word reason>

SCORING GUIDE:
- 10: passage directly contains the answer (definition, number, explicit statement).
- 7-9: passage discusses the precise topic with substantive detail.
- 4-6: passage is on-topic but peripheral — context, motivation, or a related concept.
- 1-3: passage shares vocabulary but does not address the query.
- 0: off-topic, citations, acknowledgements, boilerplate, or References-section text.

RULES:
- A related-but-different concept (e.g. "scaled dot-product attention" when asked about "multi-head attention") scores at most 5.
- Section lead-ins that introduce the topic (e.g. "we must inject some information about position...") score 5-7 — they motivate but don't fully answer.
- Be strict with 9-10. Most queries have only 1-2 candidates that deserve 8+."""

# Captures "<idx>: <score>/10" anywhere on the line, tolerating leading whitespace
# or list markers the model occasionally inserts.
_SCORE_RE = re.compile(r"(?:^|\n)\s*(\d+)\s*:\s*(\d+)\s*/\s*10\b")


def _format_candidate(idx: int, c: dict[str, Any], *, max_chars: int = 400) -> str:
    content = " ".join(c["content"].split())
    if len(content) > max_chars:
        content = content[:max_chars] + "…"
    return f"[{idx}] ({c['chunk_type']}, p{c['page']}) {content}"


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 6,
    model_role: Role = Role.FAST,
) -> list[dict[str, Any]]:
    """Score candidates with a Groq judge and return the top_k.

    model_role=FAST (8B) is the default — cheap and fast. Bump to SMART (70B)
    if the eval shows the judge is unreliable on your domain.
    """
    if not candidates:
        return []

    block = "\n\n".join(_format_candidate(i, c) for i, c in enumerate(candidates))
    n = len(candidates)
    user_msg = (
        f"Query: {query}\n\n"
        f"Candidates (0..{n - 1}):\n{block}\n\n"
        f"Output exactly {n} lines, one per candidate, in index order. "
        f"Format: <index>: <score>/10 - <≤12-word reason>"
    )
    messages = [
        SystemMessage(content=_RERANK_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    llm = get_llm(model_role, temperature=0.0, max_tokens=1500)
    response = invoke_with_retry(llm, messages)
    raw = str(response.content)
    logger.info("Rerank judge raw output:\n%s", raw)

    scores: dict[int, int] = {}
    for match in _SCORE_RE.finditer(raw):
        idx = int(match.group(1))
        score = max(0, min(10, int(match.group(2))))
        if 0 <= idx < len(candidates):
            scores[idx] = score

    if not scores:
        logger.warning(
            "Reranker returned no parseable scores; falling back to RRF order. "
            "Raw response (first 300 chars): %r",
            raw[:300],
        )

    # Sort by LLM score, then by RRF as tiebreaker so unscored items still rank sensibly.
    ranked_idx = sorted(
        range(len(candidates)),
        key=lambda i: (scores.get(i, -1), candidates[i].get("rrf_score", 0)),
        reverse=True,
    )[:top_k]

    return [{**candidates[i], "rerank_score": scores.get(i, -1)} for i in ranked_idx]
