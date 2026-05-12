"""All custom CSS for DocuMind — kept out of app.py so the layout file stays
readable. The aesthetic is 'Document Forensics Terminal': black canvas,
mono UI chrome, serif content, signal-yellow accent."""

CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@200;300;400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,200..700;1,6..72,200..700&display=swap" rel="stylesheet">

<style>
:root {
  --bg: #0A0A0A;
  --surface: #111111;
  --surface-2: #161614;
  --hair: #26261E;
  --hair-soft: #1A1A14;
  --text: #F4F4DC;
  --text-dim: #8A8A7E;
  --text-faint: #4A4A42;
  --accent: #D4FF3E;
  --accent-soft: rgba(212, 255, 62, 0.08);
  --accent-line: rgba(212, 255, 62, 0.25);
  --accent-glow: rgba(212, 255, 62, 0.45);
  --warn: #FF6B4A;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --serif: 'Newsreader', Georgia, 'Times New Roman', serif;
}

/* === Base layout === */
* { font-family: var(--mono); }

.stApp {
  background: var(--bg);
  color: var(--text);
  background-image:
    url("data:image/svg+xml;utf8,<svg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.95' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 0.9  0 0 0 0.06 0'/></filter><rect width='200' height='200' filter='url(%23n)'/></svg>"),
    radial-gradient(ellipse at top, #141410 0%, #0A0A0A 60%);
  background-attachment: fixed;
}

/* Hide Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
  padding: 0.5rem 2.2rem 2rem 2.2rem !important;
  max-width: 100% !important;
}

/* === Status bar === */
.docu-statusbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 0 0.75rem 0;
  margin-bottom: 1.25rem;
  border-bottom: 1px solid var(--hair);
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.docu-brand { display: flex; gap: 1.1rem; align-items: baseline; }
.docu-brand .ident {
  color: var(--accent); font-weight: 600;
  text-shadow: 0 0 12px var(--accent-glow);
}
.docu-brand .sub { color: var(--text-faint); }
.docu-status-right { display: flex; gap: 1rem; align-items: center; }
.docu-dot {
  display: inline-block;
  width: 6px; height: 6px;
  background: var(--accent);
  margin-right: 0.45rem;
  border-radius: 50%;
  box-shadow: 0 0 8px var(--accent-glow);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 0.55; } 50% { opacity: 1; } }

/* === Headline === */
.docu-headline {
  font-family: var(--serif);
  font-weight: 300;
  font-size: clamp(2rem, 4.5vw, 3rem);
  line-height: 0.95;
  letter-spacing: -0.025em;
  color: var(--text);
  margin: 0.5rem 0 0.5rem 0;
}
.docu-headline em {
  font-style: italic;
  font-weight: 200;
  color: var(--accent);
  text-shadow: 0 0 18px var(--accent-glow);
}
.docu-sub {
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin: 0.25rem 0 1.5rem 0;
}

/* === Stat strip === */
.docu-stats {
  display: flex;
  gap: 2.4rem;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
  padding: 0.6rem 0;
  margin: 0 0 1.25rem 0;
  border-top: 1px solid var(--hair);
  border-bottom: 1px solid var(--hair);
}
.docu-stats .key { color: var(--text-faint); }
.docu-stats .val { color: var(--text); font-weight: 500; }
.docu-stats .val-accent { color: var(--accent); font-weight: 500; }

/* === Section labels === */
.docu-label {
  font-size: 0.62rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin: 1rem 0 0.6rem 0;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.docu-label .rule {
  flex: 1;
  height: 1px;
  background: var(--hair);
}

/* === Streamlit buttons → terminal chips === */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 0.74rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.01em !important;
  background: transparent !important;
  color: var(--text-dim) !important;
  border: 1px solid var(--hair) !important;
  border-radius: 0 !important;
  padding: 0.75rem 1rem !important;
  text-align: left !important;
  white-space: normal !important;
  height: auto !important;
  min-height: 0 !important;
  line-height: 1.4 !important;
  transition: all 160ms ease;
  position: relative;
}
.stButton > button::before {
  content: "$ ";
  color: var(--text-faint);
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-soft) !important;
  transform: translateX(2px);
  box-shadow: -2px 0 0 var(--accent);
}
.stButton > button:hover::before { color: var(--accent); }
.stButton > button:focus { outline: none !important; box-shadow: none !important; }

/* Jump-to-page buttons get a different treatment (shorter, right-aligned) */
.jump-btn .stButton > button {
  text-align: center !important;
  padding: 0.5rem 0.6rem !important;
  font-size: 0.7rem !important;
}
.jump-btn .stButton > button::before { content: ""; }

/* === Two-column section header === */
.col-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 0 0.55rem 0;
  margin-bottom: 0.75rem;
  border-top: 1px solid var(--hair);
  border-bottom: 1px solid var(--hair);
  font-size: 0.64rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text-faint);
}
.col-head .accent { color: var(--accent); }

/* === Messages (custom HTML, bypass st.chat_message) === */
.msg {
  padding: 1rem 0 1.1rem 0;
  border-bottom: 1px solid var(--hair-soft);
  animation: reveal 320ms ease-out backwards;
}
.msg-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 0.5rem;
  display: flex;
  gap: 0.6rem;
  align-items: center;
}
.msg-label .role-q { color: var(--accent); }
.msg-label .role-a { color: var(--text-dim); }
.msg-label .meta { color: var(--text-faint); }

.msg-user .msg-body {
  font-family: var(--mono);
  font-size: 0.88rem;
  color: var(--text);
  line-height: 1.55;
  border-left: 2px solid var(--accent);
  padding-left: 0.85rem;
}
.msg-ai .msg-body {
  font-family: var(--serif);
  font-size: 1.05rem;
  font-weight: 350;
  color: var(--text);
  line-height: 1.6;
  letter-spacing: -0.005em;
}
.msg-ai .msg-body p { margin: 0 0 0.7rem 0; }
.msg-ai .msg-body p:last-child { margin-bottom: 0; }

/* Inline citation tag inside answer text */
.cit-inline {
  display: inline-block;
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 500;
  color: var(--accent);
  background: var(--accent-soft);
  border: 1px solid var(--accent-line);
  padding: 0.05rem 0.35rem;
  margin: 0 0.1rem;
  vertical-align: 0.18em;
  line-height: 1.2;
  letter-spacing: 0.04em;
  transition: all 120ms ease;
}
.cit-inline:hover {
  background: var(--accent);
  color: var(--bg);
}

/* Streaming cursor */
.stream-cursor {
  display: inline-block;
  width: 0.5em;
  height: 1em;
  background: var(--accent);
  vertical-align: -0.15em;
  margin-left: 0.1em;
  animation: blink 1.1s steps(2) infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* === Citation cards === */
.cit-list { margin-top: 0.5rem; }
.cit-card {
  display: grid;
  grid-template-columns: minmax(170px, auto) 1fr;
  gap: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--hair-soft);
}
.cit-card:last-child { border-bottom: none; }
.cit-tag {
  font-family: var(--mono);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.05em;
  color: var(--accent);
  white-space: nowrap;
}
.cit-tag .delim { color: var(--text-faint); font-weight: 400; }
.cit-tag .type { color: var(--text-dim); }
.cit-tag .score { color: var(--text); }
.cit-body {
  font-family: var(--mono);
  font-size: 0.76rem;
  color: var(--text-dim);
  line-height: 1.55;
}

/* === Expanders === */
[data-testid="stExpander"] {
  background: transparent !important;
  border: 1px solid var(--hair) !important;
  border-radius: 0 !important;
  margin-top: 0.65rem;
}
[data-testid="stExpander"] details summary {
  font-family: var(--mono) !important;
  font-size: 0.66rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
  padding: 0.55rem 0.8rem !important;
}
[data-testid="stExpander"] details summary:hover { color: var(--accent) !important; }
[data-testid="stExpander"] details > div { padding: 0.5rem 0.8rem 0.8rem 0.8rem !important; }

/* === Status widget (during streaming) === */
[data-testid="stStatus"] {
  background: transparent !important;
  border: 1px dashed var(--hair) !important;
  border-radius: 0 !important;
}
[data-testid="stStatus"] [data-testid="stStatusContent"] *,
[data-testid="stStatus"] summary {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.12em !important;
  color: var(--text-dim) !important;
}

/* === Chat input === */
[data-testid="stChatInput"] {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
  border-radius: 0 !important;
  transition: border-color 160ms ease;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent-line);
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.92rem !important;
  caret-color: var(--accent) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text-faint) !important;
  font-family: var(--mono) !important;
  letter-spacing: 0.02em;
}
[data-testid="stChatInput"] button {
  background: transparent !important;
  color: var(--accent) !important;
}

/* === Dividers === */
hr {
  border: none !important;
  border-top: 1px solid var(--hair) !important;
  margin: 1.25rem 0 !important;
}
.streak {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--hair), var(--accent-line), var(--hair), transparent);
  margin: 1.25rem 0;
}

/* === PDF frame === */
.pdf-frame-shell {
  position: relative;
  padding: 8px;
  border: 1px solid var(--hair);
  background: var(--surface);
}
.pdf-frame-shell::before, .pdf-frame-shell::after {
  content: "";
  position: absolute;
  width: 14px; height: 14px;
  pointer-events: none;
}
.pdf-frame-shell::before {
  top: -1px; left: -1px;
  border-top: 1.5px solid var(--accent);
  border-left: 1.5px solid var(--accent);
}
.pdf-frame-shell::after {
  bottom: -1px; right: -1px;
  border-bottom: 1.5px solid var(--accent);
  border-right: 1.5px solid var(--accent);
}

/* === Empty state === */
.empty-state {
  border: 1px dashed var(--hair);
  padding: 2.2rem 1rem;
  text-align: center;
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--text-dim);
  margin-top: 1rem;
  letter-spacing: 0.04em;
}
.empty-state .marker {
  font-size: 1.6rem;
  color: var(--text-faint);
  display: block;
  margin-bottom: 0.6rem;
  letter-spacing: 0.3em;
}
.empty-state code {
  display: inline-block;
  margin-top: 0.7rem;
  padding: 0.3rem 0.6rem;
  background: var(--surface);
  border: 1px solid var(--hair);
  color: var(--accent);
  font-size: 0.72rem;
}

/* === Scrollbar === */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--hair); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

/* === Reveal animation === */
@keyframes reveal {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.reveal { animation: reveal 380ms ease-out backwards; }
.reveal-1 { animation-delay: 80ms; }
.reveal-2 { animation-delay: 160ms; }
.reveal-3 { animation-delay: 240ms; }
.reveal-4 { animation-delay: 320ms; }

/* Streamlit container hosting our chat scroll area */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 0 !important;
}

/* Caption (used for muted notes) */
[data-testid="stCaptionContainer"] p {
  font-family: var(--mono) !important;
  font-size: 0.66rem !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-faint) !important;
}
</style>
"""
