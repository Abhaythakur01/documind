"""Quick end-to-end check that the Groq client wrapper works.

Run from the project root:
    python -m app.llm.smoke_test
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_client import Role, get_llm, invoke_with_retry


def main() -> None:
    bar = "─" * 60
    print(bar)
    print(" DocuMind — Groq smoke test")
    print(bar)

    for role in (Role.FAST, Role.SMART):
        llm = get_llm(role, temperature=0.0, max_tokens=64)
        messages = [
            SystemMessage(content="You answer in exactly one short sentence."),
            HumanMessage(content="What is retrieval-augmented generation?"),
        ]
        response = invoke_with_retry(llm, messages)
        print(f"\n[{role.value:>5}]  {response.content}")

    print("\n✓ Groq client wrapper is working.")


if __name__ == "__main__":
    main()
