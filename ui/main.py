"""DocuMind — Document Forensics Terminal.

A deliberately aggressive aesthetic: black canvas, JetBrains Mono UI chrome,
Newsreader serif for AI answers, signal-yellow (#D4FF3E) accent. The visual
language echoes the product positioning — a precision instrument for reading
dense technical documents.

All custom CSS lives in ui/styles.py to keep this file readable.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
from streamlit_pdf_viewer import pdf_viewer  # noqa: E402

from app.chains.rag_chain import build_default_chain  # noqa: E402
from app.config import settings  # noqa: E402
from ui.bootstrap import ensure_indexes  # noqa: E402
from ui.styles import CUSTOM_CSS  # noqa: E402


# ─── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind // Document Forensics Terminal",
    page_icon="◢",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Session state ───────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],          # list of {role, content, citations?, id, latency_ms?}
        "pdf_page": 1,
        "pending_query": None,
        "doc_info": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─── Bootstrap & helpers ─────────────────────────────────────────────────────
# Build indexes from data/pdfs/*.pdf if missing (HF Spaces / fresh-clone path).
ensure_indexes()


@st.cache_resource(show_spinner=False)
def get_chain():
    return build_default_chain()


def find_pdf_path() -> str | None:
    pdfs = sorted(settings.pdf_dir.glob("*.pdf"))
    return str(pdfs[0]) if pdfs else None


def load_doc_info():
    chunks_dir = settings.cache_dir / "chunks"
    files = sorted(chunks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    with files[0].open(encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    return {
        "doc_id": data["doc_id"],
        "pages": data["num_pages"],
        "text": sum(1 for c in chunks if c["chunk_type"] == "text"),
        "tables": sum(1 for c in chunks if c["chunk_type"] == "table"),
        "figures": sum(1 for c in chunks if c["chunk_type"] == "figure"),
        "name": Path(data["source_path"]).name,
    }


def jump_to_page(page: int) -> None:
    st.session_state.pdf_page = max(1, int(page))


def queue_query(q: str) -> None:
    st.session_state.pending_query = q


def style_inline_citations(text: str) -> str:
    """Wrap [N] markers in the answer text with styled chips."""
    return re.sub(r"\[(\d+)\]", r'<span class="cit-inline">[\1]</span>', text)


def render_user_message(content: str) -> None:
    st.markdown(
        f"""
        <div class="msg msg-user">
          <div class="msg-label"><span class="role-q">// QUERY</span></div>
          <div class="msg-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_message(content: str, meta: str = "", streaming: bool = False) -> str:
    styled = style_inline_citations(content)
    cursor = '<span class="stream-cursor"></span>' if streaming else ""
    label_extra = f'<span class="meta">· {meta}</span>' if meta else ""
    return f"""
    <div class="msg msg-ai">
      <div class="msg-label"><span class="role-a">// RESPONSE</span>{label_extra}</div>
      <div class="msg-body">{styled}{cursor}</div>
    </div>
    """


def render_citation_card(cit: dict) -> str:
    preview = " ".join(cit["content"].split())[:200]
    score = cit.get("rerank_score", -1)
    score_str = f"{score:02d}/10" if score >= 0 else "  —  "
    chunk_type = cit["chunk_type"].upper()
    return f"""
    <div class="cit-card">
      <div class="cit-tag">
        [{cit['n']:02d}]<span class="delim"> · </span>P.{cit['page']:02d}<span class="delim"> · </span><span class="type">{chunk_type}</span><span class="delim"> · </span><span class="score">{score_str}</span>
      </div>
      <div class="cit-body">{preview}…</div>
    </div>
    """


# Lazy-load doc info once
if st.session_state.doc_info is None:
    st.session_state.doc_info = load_doc_info()
info = st.session_state.doc_info


# ─── STATUS BAR ──────────────────────────────────────────────────────────────
doc_name = info["name"] if info else "no document indexed"
page_label = (
    f"P.{st.session_state.pdf_page:02d}/{info['pages']:02d}" if info else ""
)
st.markdown(
    f"""
    <div class="docu-statusbar">
      <div class="docu-brand">
        <span class="ident">// DOCUMIND</span>
        <span class="sub">v0.1 · multi-modal RAG terminal</span>
      </div>
      <div class="docu-status-right">
        <span><span class="docu-dot"></span>SYSTEM ONLINE</span>
        <span style="color:var(--text-faint);">·</span>
        <span>DOC <span style="color:var(--text);">{doc_name}</span></span>
        {f'<span style="color:var(--text-faint);">·</span><span>{page_label}</span>' if info else ''}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── HEADLINE ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="reveal reveal-1">
      <div class="docu-headline">An <em>instrument</em> for reading dense documents.</div>
      <div class="docu-sub">Hybrid retrieval · LLM-as-judge rerank · cited synthesis · zero hallucination</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── STAT STRIP ──────────────────────────────────────────────────────────────
if info:
    st.markdown(
        f"""
        <div class="docu-stats reveal reveal-2">
          <span><span class="key">PAGES /</span> <span class="val">{info["pages"]:02d}</span></span>
          <span><span class="key">TEXT /</span> <span class="val">{info["text"]:03d}</span></span>
          <span><span class="key">TABLES /</span> <span class="val">{info["tables"]:02d}</span></span>
          <span><span class="key">FIGURES /</span> <span class="val">{info["figures"]:02d}</span></span>
          <span><span class="key">DOC_ID /</span> <span class="val-accent">{info["doc_id"][-12:]}</span></span>
          <span style="margin-left:auto;"><span class="key">EMBED /</span> <span class="val">bge-large-en-v1.5</span></span>
          <span><span class="key">LLM /</span> <span class="val">llama-3.3-70b</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── SAMPLE QUERY CHIPS ──────────────────────────────────────────────────────
st.markdown(
    '<div class="docu-label reveal reveal-3"><span>// QUERY ▸ TRY ONE</span><span class="rule"></span></div>',
    unsafe_allow_html=True,
)
SAMPLES = [
    "what is multi-head attention?",
    "how is positional encoding computed?",
    "what BLEU score on English-to-German?",
]
chip_cols = st.columns(3, gap="small")
for col, q in zip(chip_cols, SAMPLES):
    with col:
        if st.button(q, key=f"sample-{hash(q)}", use_container_width=True):
            queue_query(q)

st.markdown('<div class="streak"></div>', unsafe_allow_html=True)


# ─── TWO-COLUMN MAIN ─────────────────────────────────────────────────────────
chat_col, pdf_col = st.columns([1, 1], gap="large")


# ============================================================================
# CHAT COLUMN
# ============================================================================
with chat_col:
    st.markdown(
        '<div class="col-head"><span>▸ DIALOGUE</span><span class="accent">↓ STREAM</span></div>',
        unsafe_allow_html=True,
    )

    chat_box = st.container(height=600, border=False)

    with chat_box:
        # Empty state
        if not st.session_state.messages and not st.session_state.pending_query:
            st.markdown(
                """
                <div class="empty-state">
                  <span class="marker">◢ ◣</span>
                  No queries logged. Pick a chip above, or type below.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Render history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                render_user_message(msg["content"])
            else:
                meta = f"{msg.get('latency_ms', '—')} ms" if msg.get("latency_ms") else ""
                st.markdown(render_ai_message(msg["content"], meta=meta), unsafe_allow_html=True)
                if msg.get("citations"):
                    with st.expander(f"⊟ SOURCES · {len(msg['citations']):02d} CITED", expanded=False):
                        for cit in msg["citations"]:
                            cit_cols = st.columns([0.82, 0.18])
                            with cit_cols[0]:
                                st.markdown(render_citation_card(cit), unsafe_allow_html=True)
                            with cit_cols[1]:
                                st.markdown('<div class="jump-btn">', unsafe_allow_html=True)
                                st.button(
                                    f"↗ P.{cit['page']:02d}",
                                    key=f"jump-{msg['id']}-{cit['n']}",
                                    on_click=jump_to_page,
                                    args=(cit["page"],),
                                    use_container_width=True,
                                )
                                st.markdown('</div>', unsafe_allow_html=True)

    # ─── Input ───
    typed = st.chat_input("Type a question · answers will cite the source page")
    query = typed or st.session_state.pending_query
    if st.session_state.pending_query and not typed:
        st.session_state.pending_query = None

    if query:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "id": f"u{len(st.session_state.messages)}",
            }
        )
        with chat_box:
            render_user_message(query)

            placeholder = st.empty()
            placeholder.markdown(
                render_ai_message("", meta="initializing", streaming=True),
                unsafe_allow_html=True,
            )

            chain = get_chain()
            full_text = ""
            citations: list[dict] = []
            t0 = time.perf_counter()
            n_candidates = 0
            for event in chain.stream(query):
                kind, data = event["event"], event["data"]
                if kind == "retrieve":
                    n_candidates = data["n_candidates"]
                    placeholder.markdown(
                        render_ai_message("", meta=f"reranking {n_candidates:02d}", streaming=True),
                        unsafe_allow_html=True,
                    )
                elif kind == "rerank":
                    placeholder.markdown(
                        render_ai_message("", meta="synthesizing", streaming=True),
                        unsafe_allow_html=True,
                    )
                elif kind == "token":
                    full_text += data["text"]
                    placeholder.markdown(
                        render_ai_message(full_text, meta="streaming", streaming=True),
                        unsafe_allow_html=True,
                    )
                elif kind == "done":
                    citations = data["citations"]

            latency_ms = int((time.perf_counter() - t0) * 1000)
            placeholder.markdown(
                render_ai_message(full_text, meta=f"{latency_ms} ms"),
                unsafe_allow_html=True,
            )

        msg_id = f"a{len(st.session_state.messages)}"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_text,
                "citations": [{"n": i + 1, **c} for i, c in enumerate(citations)],
                "id": msg_id,
                "latency_ms": latency_ms,
            }
        )
        if citations:
            jump_to_page(citations[0]["page"])
        st.rerun()


# ============================================================================
# PDF COLUMN
# ============================================================================
with pdf_col:
    pdf_path = find_pdf_path()
    page_label = (
        f"P.{st.session_state.pdf_page:02d}" + (f"/{info['pages']:02d}" if info else "")
    )
    st.markdown(
        f'<div class="col-head"><span>▸ DOCUMENT</span><span class="accent">{page_label}</span></div>',
        unsafe_allow_html=True,
    )

    if pdf_path is None:
        st.markdown(
            """
            <div class="empty-state">
              <span class="marker">◢</span>
              NO DOCUMENT INDEXED
              <br>
              <code>python -m app.ingest.run data/pdfs/your.pdf</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="pdf-frame-shell">', unsafe_allow_html=True)
        pdf_viewer(
            pdf_path,
            width=700,
            height=620,
            scroll_to_page=st.session_state.pdf_page,
            key="pdf_viewer",
        )
        st.markdown("</div>", unsafe_allow_html=True)
