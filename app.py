"""
Virtual Manish — redesigned studio UI.
Run: streamlit run app.py
"""
import sys
import asyncio

# Suppress harmless Windows asyncio ConnectionResetError noise
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import json
import re
import os
from pathlib import Path
from datetime import datetime

import streamlit as st

import Creative_DNA as backend


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "creative_examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Virtual Manish",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add this right after st.set_page_config
#st.set_option('server.fileWatcherType', 'none')

# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
/* ============ RESET & BASE ============ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background: #0a0a0d;
    color: #c8c8d0;
}
.block-container {
    padding: 1.4rem 2rem 3rem 2rem !important;
    max-width: 100% !important;
}

/* ============ TEXT HIERARCHY ============ */
h1, h2, h3, h4 {
    color: #f5f5f7;
    font-weight: 600;
    letter-spacing: -0.015em;
    margin: 0;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding-bottom: 0.4rem;
}
.brand-icon {
    font-size: 1.6rem;
    line-height: 1;
    filter: drop-shadow(0 0 12px rgba(255,169,77,0.35));
}
.brand-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f5f5f7;
    letter-spacing: -0.02em;
}
.brand-tag {
    font-size: 0.82rem;
    color: #7a7a85;
    margin-bottom: 1.8rem;
    letter-spacing: 0.005em;
}

/* ============ SECTION HEADERS ============ */
.section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #7a7a85;
    font-weight: 700;
    margin: 0 0 0.9rem 0;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #2a2a32, transparent);
}
.section-count {
    color: #ffa94d;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

/* ============ SCENE ROWS ============ */
.scene-list {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

/* ============ PROMPT ============ */
.prompt-label {
    font-size: 0.72rem;
    color: #7a7a85;
    margin-bottom: 0.5rem;
    letter-spacing: 0.02em;
}

/* ============ RETRIEVAL ============ */
.retrieval-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    background: #131318;
    border: 1px solid #1e1e26;
    margin-bottom: 0.35rem;
    font-size: 0.82rem;
}
.retrieval-item.high { border-left: 3px solid #4ade80; }
.retrieval-item.mid { border-left: 3px solid #ffa94d; }
.retrieval-item.low { border-left: 3px solid #3a3a45; opacity: 0.55; }
.retrieval-file {
    font-family: 'SF Mono', 'Menlo', monospace;
    color: #b0b0ba;
    font-size: 0.78rem;
}
.retrieval-score {
    font-family: 'SF Mono', 'Menlo', monospace;
    font-size: 0.74rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.retrieval-score.high { color: #4ade80; }
.retrieval-score.mid  { color: #ffa94d; }
.retrieval-score.low  { color: #6a6a75; }

/* ============ GENERATION CARDS ============ */
.gen-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: #16161b;
    border: 1px solid #22222a;
    border-bottom: none;
    border-radius: 10px 10px 0 0;
}
.gen-time {
    font-size: 0.72rem;
    color: #7a7a85;
    font-family: 'SF Mono', 'Menlo', monospace;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.gen-badge {
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 3px 9px;
    border-radius: 4px;
    background: rgba(255,169,77,0.12);
    color: #ffa94d;
    border: 1px solid rgba(255,169,77,0.25);
}
.gen-body {
    background: #131318;
    border: 1px solid #22222a;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1rem;
    margin-bottom: 0.9rem;
}

/* ============ BUTTONS ============ */
.stButton > button {
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.14s ease !important;
    border: 1px solid #2a2a32 !important;
    background: #16161b !important;
    color: #d0d0d8 !important;
    height: 2.35rem !important;
    padding: 0 0.9rem !important;
}
.stButton > button:hover {
    background: #1e1e26 !important;
    border-color: #3a3a45 !important;
    color: #f5f5f7 !important;
}
.stButton > button:active {
    transform: scale(0.98);
}
/* Primary — unmissable */
.stButton > button[kind="primary"] {
    background: #ffa94d !important;
    color: #0a0a0d !important;
    border-color: #ffa94d !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 0 24px rgba(255,169,77,0.2) !important;
    height: 2.6rem !important;
}
.stButton > button[kind="primary"]:hover {
    background: #ffbe6b !important;
    border-color: #ffbe6b !important;
    box-shadow: 0 0 32px rgba(255,169,77,0.35) !important;
}

/* ============ INPUTS ============ */
.stTextArea textarea:disabled,
.stTextArea textarea[disabled] {
    color: #c8c8d0 !important;
    -webkit-text-fill-color: #c8c8d0 !important;
    opacity: 1 !important;
    background: #0f0f14 !important;
    cursor: default !important;
    border: 1px solid #22222a !important;
}
.stTextArea textarea, .stTextInput input {
    background: #0f0f14 !important;
    border: 1px solid #22222a !important;
    color: #d0d0d8 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    padding: 0.7rem !important;
    transition: border-color 0.14s ease, box-shadow 0.14s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #ffa94d !important;
    box-shadow: 0 0 0 3px rgba(255,169,77,0.12) !important;
    outline: none !important;
}
.stTextArea textarea {
    font-family: 'SF Mono', 'Menlo', 'JetBrains Mono', monospace !important;
    line-height: 1.6 !important;
    font-size: 0.83rem !important;
}

/* ============ MISC ============ */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
.block-container {
    padding-top: 2rem !important;
}
.empty-state {
    text-align: center;
    color: #4a4a55;
    font-size: 0.83rem;
    padding: 2.5rem 1rem;
    font-style: italic;
    line-height: 1.6;
}
.stSpinner > div { border-top-color: #ffa94d !important; }
hr { border-color: #22222a !important; margin: 1.2rem 0 !important; }
.stCaption, .stMarkdown small { color: #6a6a75 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ============ SEPARATOR BETWEEN COLUMNS ============ */
div[data-testid="column"]:not(:last-child) {
    border-right: 1px solid #16161b;
    padding-right: 1.6rem !important;
}
div[data-testid="column"]:not(:first-child) {
    padding-left: 1.6rem !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STATE
# ============================================================

if "generations" not in st.session_state:
    st.session_state.generations = []
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "retrieval_preview" not in st.session_state:
    st.session_state.retrieval_preview = []
if "last_preview_prompt" not in st.session_state:
    st.session_state.last_preview_prompt = ""


# ============================================================
# HELPERS
# ============================================================

def register_class(reg):
    r = reg.lower()
    if "comedy" in r: return "reg-comedy"
    if "thriller" in r: return "reg-thriller"
    if "drama" in r: return "reg-drama"
    if "romance" in r: return "reg-romance"
    if "tragedy" in r: return "reg-tragedy"
    return "reg-untagged"


def list_scenes():
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(EXAMPLES_DIR.glob("*.txt"))


def read_scene(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def register_of(text):
    m = re.search(r"\[REGISTER:\s*([^\]]+)\]", text, re.IGNORECASE)
    return m.group(1).strip() if m else "untagged"


def word_count(text):
    return len(text.split())


def preview_retrieval(prompt):
    """Preview the top matches, deduplicated by source file (max 3)."""
    if not prompt.strip() or len(prompt.strip()) < 12:
        return []
    try:
        from rag_memory import search_memory
        results = search_memory(prompt, limit=10) # fetch more to allow dedup
        
        seen = set()
        deduped = []
        for r in results:
            src = r.get("source", "unknown")
            if src in seen:
                continue
            seen.add(src)
            deduped.append({
                "file": src,
                "score": float(r.get("score", 0.0)),
            })
            if len(deduped) >= 3:
                break
        return deduped
    except Exception:
        return []


import concurrent.futures

def run_generation(prompt):
    import io
    import contextlib
    import traceback

    buf = io.StringIO()
    result = None

    # We wrap the backend call in a ThreadPoolExecutor to enforce a hard timeout.
    def _call_backend():
        try:
            return backend.chat_with_ai(prompt)
        except Exception as e:
            return f"[CRITICAL ERROR]\n{traceback.format_exc()}"

    print("\n[UI] Calling backend...")
    
    try:
        with contextlib.redirect_stdout(buf):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_call_backend)
                # Wait max 90 seconds. If it takes longer, kill it.
                result = future.result(timeout=90)
    except concurrent.futures.TimeoutError:
        result = "[CRITICAL ERROR] Generation timed out after 90 seconds."
    except Exception as e:
        result = f"[CRITICAL ERROR]\n{traceback.format_exc()}"

    captured = buf.getvalue()
    
    # Print the backend output to the terminal so we can see it
    print("\n=== BACKEND LOG ===")
    print(captured)
    print("===================\n")
    
    if not result or len(result.strip()) < 10 or "[ERROR]" in result or "[CRITICAL" in result:
        print("\n[GENERATION FAILED]")
        print(result)
        print("------------------\n")
        return "Generation failed or timed out. Check the terminal for details.", []

    retrieval = []
    for line in captured.splitlines():
        m = re.match(r"\s*-\s+(.+?)\s+\(score ([\d.]+)\)", line)
        if m:
            retrieval.append({"file": m.group(1).strip(), "score": float(m.group(2))})

    return result.strip(), retrieval

def score_class(score):
    if score >= 0.65: return "high"
    if score >= 0.55: return "mid"
    return "low"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand"><span class="brand-icon">🎬</span>'
    '<span class="brand-title">Virtual Manish</span></div>'
    '<div class="brand-tag">A writing studio that learns your voice.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LAYOUT
# ============================================================

col_left, col_center, col_right = st.columns([1.05, 1.55, 2.4], gap="medium")

# ============ LEFT: Corpus ============
with col_left:
    scenes = list_scenes()
    st.markdown(
        f'<div class="section-title">Corpus'
        f'<span class="section-count">{len(scenes)}</span></div>',
        unsafe_allow_html=True,
    )

    if not scenes:
        st.markdown(
            '<div class="empty-state">No scenes yet.<br>'
            'Add a .txt file to creative_examples/.</div>',
            unsafe_allow_html=True,
        )
    else:
        for f in scenes:
            text = read_scene(f)
            reg = register_of(text)
            wc = word_count(text)
            is_active = st.session_state.selected_file == f.name
            marker = "●" if is_active else "○"

            label = f"{marker}  {f.stem}   ·   {wc}w"
            if st.button(
                label,
                key=f"open_{f.name}",
                use_container_width=True,
            ):
                st.session_state.selected_file = (
                    None if is_active else f.name
                )
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.selected_file:
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        p = EXAMPLES_DIR / st.session_state.selected_file
        if p.exists():
            raw = read_scene(p)
            lines = raw.splitlines()
            body_lines = []
            in_header = True
            for line in lines:
                stripped = line.strip()
                if in_header and (stripped.startswith("[") and stripped.endswith("]")):
                    continue
                if in_header and stripped == "":
                    continue
                in_header = False
                body_lines.append(line)
            preview_text = "\n".join(body_lines).strip()

            st.text_area(
                "preview",
                preview_text,
                height=280,
                label_visibility="collapsed",
                disabled=True,
                key="preview_area",
            )

    if st.button("⟳  Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding…"):
            try:
                import subprocess
                subprocess.run(
                    ["python", "rag_memory.py", "--build"],
                    capture_output=True, timeout=180,
                )
                st.success("Index rebuilt.")
            except Exception as e:
                st.error(f"Failed: {e}")


# ============ CENTER: Prompt ============
with col_center:
    st.markdown('<div class="section-title">Workspace</div>', unsafe_allow_html=True)
    
    # Create two tabs
    tab_write, tab_diagnose = st.tabs(["✍️ Write Scene", "🩺 Diagnose & Character Fit"])
    
    # --- TAB 1: WRITE SCENE ---
    with tab_write:
        st.markdown('<div class="prompt-label">Quick Templates</div>', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        
        with t1:
            if st.button("🎭 Quiet Drama", use_container_width=True):
                st.session_state.last_prompt = (
                    "Write a quiet family drama scene. Use subtext and physical "
                    "actions rather than explicit dialogue to show the tension. "
                    "End on a small physical gesture.\n\n[REGISTER: quiet realism]"
                )
                st.rerun()
                
        with t2:
            if st.button("😂 Comedy", use_container_width=True):
                st.session_state.last_prompt = (
                    "Write a comedy scene where characters' personalities clash. "
                    "The humor should come from their bad decisions and reactions, "
                    "not from punchlines.\n\n[REGISTER: feel-good comedy]"
                )
                st.rerun()
                
        with t3:
            if st.button("🕵️ Thriller", use_container_width=True):
                st.session_state.last_prompt = (
                    "Write a psychological thriller scene where the protagonist "
                    "notices something is wrong, but can't prove it. Build dread "
                    "through small inconsistencies.\n\n[REGISTER: thriller]"
                )
                st.rerun()
                
        with t4:
            if st.button("🌱 Slice of Life", use_container_width=True):
                st.session_state.last_prompt = (
                    "Write a slice-of-life scene. Nothing dramatic happens. "
                    "Focus on ordinary people, lived-in details, and warmth.\n\n"
                    "[REGISTER: slice-of-life]"
                )
                st.rerun()
                
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        prompt = st.text_area(
            "prompt",
            value=st.session_state.last_prompt,
            height=150,
            placeholder="Select a template above, or type your own idea...",
            label_visibility="collapsed",
        )
        st.session_state.last_prompt = prompt

        if prompt.strip() and prompt != st.session_state.last_preview_prompt:
            if len(prompt.strip()) >= 12:
                st.session_state.retrieval_preview = preview_retrieval(prompt)
                st.session_state.last_preview_prompt = prompt

        if st.session_state.retrieval_preview:
            st.markdown('<div class="section-title" style="margin-top:1.2rem;">Would retrieve</div>', unsafe_allow_html=True)
            for r in st.session_state.retrieval_preview:
                sc = score_class(r["score"])
                st.markdown(
                    f'<div class="retrieval-item {sc}">'
                    f'<span class="retrieval-file">{r["file"]}</span>'
                    f'<span class="retrieval-score {sc}">{r["score"]:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.4, 1])
        with c1:
            generate = st.button("✨  Generate Scene", type="primary", use_container_width=True)
        with c2:
            clear = st.button("Clear", use_container_width=True)

        if clear:
            st.session_state.last_prompt = ""
            st.session_state.retrieval_preview = []
            st.session_state.last_preview_prompt = ""
            st.rerun()

        if generate and prompt.strip():
            with st.spinner("Writing… 30–60 seconds."):
                result, retrieval = run_generation(prompt)
            
            # Check if it failed silently
            if not result or len(result.strip()) < 10:
                st.error("The backend returned an empty response. Check the terminal for the actual error.")
            else:
                st.session_state.generations.insert(0, {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "prompt": prompt,
                    "text": result,
                    "retrieval": retrieval,
                })
                st.rerun()

    # --- TAB 2: DIAGNOSE ---
    with tab_diagnose:
        st.markdown('<div class="prompt-label">Paste the script you are stuck on:</div>', unsafe_allow_html=True)
        
        stuck_script = st.text_area(
            "script_context",
            height=200,
            placeholder="Paste the half-finished scene here...",
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="prompt-label">What are you stuck on?</div>', unsafe_allow_html=True)
        
        stuck_prompt = st.text_input(
            "stuck_block",
            placeholder="e.g., I need a funny uncle to walk in, but I don't want him to ruin the sadness.",
            label_visibility="collapsed"
        )
        
        if st.button("🩺 Analyze Scene & Suggest Character", type="primary", use_container_width=True):
            if stuck_script.strip() and stuck_prompt.strip():
                with st.spinner("Searching Character Bible & analyzing tone..."):
                    diagnosis = backend.diagnose_stuck_scene(stuck_script, stuck_prompt)
                
                st.markdown("### Diagnosis")
                st.info(diagnosis)
            else:
                st.warning("Please provide both the script context and what you are stuck on.")


# ============ RIGHT: Generations ============
with col_right:
    n = len(st.session_state.generations)
    st.markdown(
        f'<div class="section-title">Output'
        f'<span class="section-count">{n}</span></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.generations:
        st.markdown(
            '<div class="empty-state">Generated scenes appear here.<br>'
            'Each generation stays as a card you can edit, compare, and save.</div>',
            unsafe_allow_html=True,
        )
    else:
        for idx, gen in enumerate(st.session_state.generations):
            st.markdown(
                f'<div class="gen-card-header">'
                f'<span class="gen-time">#{n - idx} · {gen["timestamp"]}</span>'
                f'<span class="gen-badge">generated</span>'
                f'</div>'
                f'<div class="gen-body">',
                unsafe_allow_html=True,
            )

            edited = st.text_area(
                f"gen_{idx}",
                value=gen["text"],
                height=380,
                label_visibility="collapsed",
                key=f"edit_{idx}",
            )
            st.session_state.generations[idx]["text"] = edited

            st.markdown("</div>", unsafe_allow_html=True)

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("💾  Save", key=f"save_{idx}", use_container_width=True):
                    st.session_state[f"show_save_{idx}"] = True
            with b2:
                if st.button("📝  Note", key=f"fb_{idx}", use_container_width=True):
                    st.session_state[f"show_fb_{idx}"] = True
            with b3:
                if st.button("🗑  Drop", key=f"drop_{idx}", use_container_width=True):
                    st.session_state.generations.pop(idx)
                    st.rerun()

            if st.session_state.get(f"show_save_{idx}"):
                with st.form(f"save_form_{idx}"):
                    fname = st.text_input("Filename (no .txt)", value="new_scene")
                    reg = st.text_input("Register", value="feel-good comedy")
                    if st.form_submit_button("Confirm save", type="primary"):
                        safe = re.sub(r"[^a-zA-Z0-9_\- ]", "", fname).strip().replace(" ", "_")
                        if safe:
                            target = EXAMPLES_DIR / f"{safe}.txt"
                            target.write_text(
                                f"[REGISTER: {reg}]\n\n{edited}",
                                encoding="utf-8",
                            )
                            st.success(f"Saved → {target.name}")
                            st.session_state[f"show_save_{idx}"] = False

            if st.session_state.get(f"show_fb_{idx}"):
                with st.form(f"fb_form_{idx}"):
                    note = st.text_area("What was wrong / how to fix", height=100)
                    if st.form_submit_button("Save feedback", type="primary"):
                        if note.strip():
                            try:
                                backend.save_feedback(edited, note.strip())
                                st.success("Feedback saved.")
                                st.session_state[f"show_fb_{idx}"] = False
                            except Exception as e:
                                st.error(f"Failed: {e}")

            st.markdown("<br>", unsafe_allow_html=True)