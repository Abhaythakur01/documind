"""Answer synthesizer: 70B reads top-K passages and writes a cited answer.

Design choices worth defending in an interview:

1. **Numbered citations [1][2] mapped to passage order**, not raw chunk IDs.
   - The LLM can keep these tight in working memory.
   - The UI resolves [N] → {page, bbox} and renders clickable highlights.

2. **Synthesis uses 70B, not 8B.**
   - Reranking can use 8B because relevance is a classification task.
   - Synthesis requires composition + faithfulness, where small models
     hallucinate. The cost is one 70B call per query — well within free tier.

3. **Refusal is explicit.** If the passages don't contain the answer, the
   prompt forces a refusal rather than letting the model invent one. This is
   what makes the answers actually trustworthy for a hiring-manager demo.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_client import Role, get_llm, invoke_with_retry, stream_with_retry

_SYNTHESIZER_SYSTEM = """You are a precise document analyst. You answer questions about a document using ONLY the numbered passages provided to you.

RULES:
1. Every factual claim must end with a citation in square brackets, like [1] or [2][3]. Numbers refer to the passage list shown below.
2. If the passages do NOT contain a clear answer, reply EXACTLY: "The provided passages do not contain enough information to answer this question." — do not speculate, do not draw on outside knowledge.
3. ALSO refuse with the same exact line if ANY of these hold:
   - The question is too vague to identify a specific topic (e.g. "what is this?", "tell me about it").
   - The retrieved passages are incoherent fragments, unrelated to each other, or appear to be example/visualization content rather than substantive document content.
   - You cannot find at least one passage that directly addresses the question.
4. Do not summarize what the passages "appear to be about" — either answer the specific question with citations or refuse.
5. Quote exact numbers, equations, and proper nouns verbatim from the passages.
6. Be concise. Aim for 2–4 sentences unless the question genuinely requires a long answer.
7. Do not include a "Sources" or "References" footer — citations are inline only.
8. Do not narrate your reasoning ("Based on passage [2]..."). State the answer directly with the citation at the end of the supporting sentence.

EXAMPLES OF GOOD CITATION:
- "The Transformer (big) achieved a BLEU score of 28.4 on English-to-German [3]."
- "Positional encoding uses sine and cosine functions of different frequencies [1][2]."

EXAMPLES OF BAD ANSWERS:
- "Based on the passages, it seems like..." (don't hedge)
- "The provided text appears to be about..." (don't describe the passages — answer or refuse)
- "The Transformer achieved 28.4 BLEU." (no citation)
- "The answer is in passage [1]." (don't reference passage numbers as objects)"""


def _format_passages(citations: list[dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        header = f"[{i}] (page {c['page']}, {c['chunk_type']})"
        blocks.append(f"{header}\n{c['content']}")
    return "\n\n".join(blocks)


def _build_messages(query: str, citations: list[dict[str, Any]]) -> list:
    passages = _format_passages(citations)
    return [
        SystemMessage(content=_SYNTHESIZER_SYSTEM),
        HumanMessage(
            content=(
                f"Question: {query}\n\n"
                f"Passages:\n{passages}\n\n"
                "Answer (concise, with inline [N] citations):"
            )
        ),
    ]


def synthesize(
    query: str,
    citations: list[dict[str, Any]],
    *,
    model_role: Role = Role.SMART,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Generate a cited answer from the reranked passages.

    Returns dict with keys 'answer' (str) and 'citations' (list[dict]).
    """
    if not citations:
        return {
            "answer": "The provided passages do not contain enough information to answer this question.",
            "citations": [],
        }

    messages = _build_messages(query, citations)
    llm = get_llm(model_role, temperature=0.0, max_tokens=max_tokens)
    response = invoke_with_retry(llm, messages)
    return {
        "answer": str(response.content).strip(),
        "citations": citations,
    }


def stream_synthesize(
    query: str,
    citations: list[dict[str, Any]],
    *,
    model_role: Role = Role.SMART,
    max_tokens: int = 512,
) -> Iterator[str]:
    """Token-by-token streaming version for FastAPI SSE."""
    if not citations:
        yield "The provided passages do not contain enough information to answer this question."
        return

    messages = _build_messages(query, citations)
    llm = get_llm(model_role, temperature=0.0, max_tokens=max_tokens, streaming=True)
    for chunk in stream_with_retry(llm, messages):
        text = str(chunk.content) if chunk.content else ""
        if text:
            yield text
