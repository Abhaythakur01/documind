"""Groq LLM client with model routing, retry, and streaming.

Free-tier survival kit:
- Route the smart 70B model only to roles that need it (synthesis, judge).
- Use the 8B model for everything else — sub-agents, reranking, query rewriting.
- Wrap invocations with exponential backoff so a transient 429 doesn't kill a run.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterator

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings


class Role(str, Enum):
    """Logical model role — maps to a concrete Groq model via settings."""

    SMART = "smart"
    FAST = "fast"
    VISION = "vision"


_MODEL_MAP: dict[Role, str] = {
    Role.SMART: settings.groq_model_smart,
    Role.FAST: settings.groq_model_fast,
    Role.VISION: settings.groq_model_vision,
}


def get_llm(
    role: Role = Role.FAST,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> ChatGroq:
    """Return a configured ChatGroq client for the given logical role."""
    return ChatGroq(
        model=_MODEL_MAP[role],
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        **kwargs,
    )


def _is_retryable(exc: BaseException) -> bool:
    """Match transient Groq errors worth retrying."""
    msg = str(exc).lower()
    return any(
        signal in msg
        for signal in (
            "rate_limit",
            "rate limit",
            "429",
            "too many requests",
            "timeout",
            "connection",
            "service unavailable",
            "502",
            "503",
        )
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def invoke_with_retry(llm: ChatGroq, messages: list[BaseMessage]) -> BaseMessage:
    """Invoke an LLM with bounded exponential backoff on transient errors."""
    return llm.invoke(messages)


def stream_with_retry(
    llm: ChatGroq, messages: list[BaseMessage]
) -> Iterator[BaseMessage]:
    """Stream tokens from an LLM.

    Only the initial connection attempt is retried — partial streams are not
    resumed (would require token-position bookkeeping not worth the complexity
    here).
    """

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        reraise=True,
    )
    def _open_stream() -> Iterator[BaseMessage]:
        return llm.stream(messages)

    yield from _open_stream()
