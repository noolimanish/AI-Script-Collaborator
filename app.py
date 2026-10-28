"""
Virtual Manish — CineScript Studio (With Local NLP Director).
Run: streamlit run app.py
"""
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import json
import re
import os
import base64
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

import Creative_DNA as backend
import nlp_director  # <-- NEW LOCAL NLP MODULE

# ============================================================
# CONFIG & PAGE SETUP
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "creative_examples"
POSTERS_DIR = BASE_DIR / "posters"
EXAMPLES_DIR.mkdir(exist_ok=True)
POSTERS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="CineScript Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STATE INITIALIZATION
# ============================================================
if "generations" not in st.session_state:
    st.session_state.generations = []
if "selected_script" not in st.session_state:
    st.session_state.selected_script = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0
if "last_preview_prompt" not in st.session_state:
    st.session_state.last_preview_prompt = ""
if "retrieval_preview" not in st.session_state:
    st.session_state.retrieval_preview = {}
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = ""
if "show_journal" not in st.session_state:
    st.session_state.show_journal = False

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    -webkit-font-smoothing: antialiased;
    background-color: #020205;
    color: #e2e2ec;
}
.stApp {
    background: radial-gradient(circle at top right, #150524 0%, #020205 50%, #020205 100%);
}
.block-container {
    padding: 0rem 4rem 4rem 4rem !important;
    max-width: 1600px !important;
}

header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.brand { font-size: 1.6rem; font-weight: 900; letter-spacing: 3px; color: #fff; display: flex; align-items: center; gap: 10px; }
.brand-icon { color: #ff007f; font-size: 1.8rem; filter: drop-shadow(0 0 8px #ff007f); }

.hero-title {
    font-size: 5rem; font-weight: 900; line-height: 1.0; margin-bottom: 1rem; text-transform: uppercase;
    background: linear-gradient(90deg, #00f0ff, #ff007f, #ff00ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(255,0,127,0.2); letter-spacing: -2px;
}
.hero-subtitle { font-size: 1.1rem; color: #a0a0b0; max-width: 650px; margin-bottom: 3rem; line-height: 1.7; }

.stButton > button {
    border-radius: 50px !important; font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important; 
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid #2a2a3a !important; background: #0d0d15 !important; color: #a0a0b0 !important;
    height: 3rem !important; white-space: nowrap !important; padding: 0 1rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px); border-color: #00f0ff !important; color: #00f0ff !important;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.15) !important; background: #11111a !important;
}
.stButton > button:active { transform: translateY(1px) scale(0.98) !important; box-shadow: none !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff007f, #ff00ff) !important; color: #fff !important; border: none !important;
    box-shadow: 0 4px 20px rgba(255,0,127,0.3) !important;
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 8px 30px rgba(255,0,127,0.5) !important; transform: translateY(-3px) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid #1a1a2e; }
.stTabs [data-baseweb="tab"] {
    height: 50px; background-color: transparent; border-radius: 4px 4px 0px 0px;
    font-weight: 800; color: #5a5a65; font-size: 1.1rem; text-transform: uppercase; transition: 0.3s;
}
.stTabs [data-baseweb="tab"]:hover { color: #fff; }
.stTabs [aria-selected="true"] { color: #00f0ff !important; border-bottom: 3px solid #00f0ff !important; text-shadow: 0 0 10px rgba(0,240,255,0.4); }

.stTextArea textarea, .stTextInput input {
    background: #09090e !important; border: 1px solid #1a1a2e !important; color: #e2e2ec !important;
    border-radius: 12px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important; padding: 1.2rem !important; line-height: 1.6 !important;
    transition: all 0.3s ease !important; resize: vertical !important;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color: #00f0ff !important; box-shadow: 0 0 0 3px rgba(0,240,255,0.1) !important; }

[data-testid="stMultiSelect"] {
    width: 100% !important;
    max-width: 100% !important;
}

.scripts-title-container {
    position: relative; text-align: center; margin-bottom: 3rem; margin-top: 4rem;
    display: flex; justify-content: center; align-items: center; height: 120px;
}
.scripts-bg {
    position: absolute; font-size: 8.5rem; font-weight: 900; color: transparent;
    -webkit-text-stroke: 2px #5a00ff; letter-spacing: 5px; text-transform: uppercase; z-index: 1; opacity: 0.8;
}
.scripts-fg {
    position: relative; font-size: 4rem; font-weight: 900; color: #ff007f; text-transform: uppercase;
    letter-spacing: 8px; z-index: 2; text-shadow: 0 0 20px rgba(255,0,127,0.6);
}

.markdown-box {
    background: #0b0b12; padding: 2rem; border-radius: 14px; border: 1px solid #1a1a2e;
    font-size: 0.95rem; line-height: 1.7; color: #d0d0d8; max-height: 600px; overflow-y: auto;
}

.transparency-title {
    font-size: 0.75rem; color: #ffa94d; font-weight: 900; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase;
}
.trace-item {
    font-size: 0.75rem; color: #a0a0b0; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;
}
.trace-score { color: #00f0ff; font-weight: bold; }

.journal-box {
    background: #11111a; border: 2px solid #ff007f; border-radius: 14px; padding: 1.5rem; margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_base64_image(image_path):
    if image_path.exists():
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def preview_retrieval(prompt):
    if not prompt.strip() or len(prompt.strip()) < 12:
        return {}
    try:
        from rag_memory import search_memory, search_style_references, search_structure_references, search_scene_techniques, search_failure_rules
        trace = {}
        mem = search_memory(prompt, limit=3)
        trace["voice"] = [{"source": r.get("source"), "score": round(r.get("score", 0), 3)} for r in mem]
        style = search_style_references(prompt, limit=2)
        trace["style"] = [{"source": r.get("source"), "score": round(r.get("score", 0), 3)} for r in style]
        struct = search_structure_references(prompt, limit=2)
        trace["structure"] = [{"source": r.get("source"), "score": round(r.get("score", 0), 3)} for r in struct]
        tech = search_scene_techniques(prompt, limit=2)
        trace["techniques"] = [{"source": t.get("type"), "score": "Matched Rule"} for t in tech]
        fails = search_failure_rules(prompt, limit=3)
        trace["failure_modes"] = [{"source": r.get("failure_type"), "score": round(r.get("score", 0), 3)} for r in fails]
        return trace
    except Exception as e:
        print(f"Preview error: {e}")
        return {}

def render_transparency_box(prompt, key_suffix=""):
    if prompt.strip() and prompt != st.session_state.last_preview_prompt:
        if len(prompt.strip()) >= 10:
            st.session_state.retrieval_preview = preview_retrieval(prompt)
            st.session_state.last_preview_prompt = prompt

    if st.session_state.retrieval_preview:
        with st.expander("🔍 System Transparency (What the AI will read)", expanded=True):
            trace = st.session_state.retrieval_preview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='transparency-title'>🎭 Voice (creative_examples)</div>", unsafe_allow_html=True)
                for v in trace.get("voice", []):
                    st.markdown(f"<div class='trace-item'>• {v['source']} <span class='trace-score'>({v['score']})</span></div>", unsafe_allow_html=True)
                    
                st.markdown("<div class='transparency-title' style='margin-top:12px;'>🏗️ Structure (liked_structure)</div>", unsafe_allow_html=True)
                for s in trace.get("structure", []):
                    st.markdown(f"<div class='trace-item'>• {s['source']} <span class='trace-score'>({s['score']})</span></div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='transparency-title'>📝 Style (my_style_pattern)</div>", unsafe_allow_html=True)
                for stl in trace.get("style", []):
                    st.markdown(f"<div class='trace-item'>• {stl['source']} <span class='trace-score'>({stl['score']})</span></div>", unsafe_allow_html=True)
                    
                st.markdown("<div class='transparency-title' style='margin-top:12px;'>🪝 Mechanics (scene_techniques)</div>", unsafe_allow_html=True)
                for t in trace.get("techniques", []):
                    st.markdown(f"<div class='trace-item'>• {t['source']} <span class='trace-score'>({t['score']})</span></div>", unsafe_allow_html=True)
                    
                st.markdown("<div class='transparency-title' style='margin-top:12px;'>🛑 Failure Modes</div>", unsafe_allow_html=True)
                for f in trace.get("failure_modes", []):
                    st.markdown(f"<div class='trace-item'>• {f['source']} <span class='trace-score'>({f['score']})</span></div>", unsafe_allow_html=True)

# ============================================================
# TOP NAVIGATION & HERO
# ============================================================
st.markdown('<div class="brand"><span class="brand-icon">🎬</span> CINESCRIPT</div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-title" style="margin-top: 2rem;">CINE<br>SCRIPT</div>
<div class="hero-subtitle">Where stories come to life. The ultimate AI-powered screenwriting studio.</div>
""", unsafe_allow_html=True)

st.markdown("<br><hr style='border-color: #1a1a2e;'><br>", unsafe_allow_html=True)

# ============================================================
# MAIN WORKSPACE (Tabs)
# ============================================================
tab_write, tab_brainstorm, tab_diag = st.tabs(["✍️ Write Scene", "🧠 Brainstorm", "🩺 Scene Doctor"])

# --- TAB 1: WRITE SCENE ---
with tab_write:
    st.markdown('<div id="write-scene"></div>', unsafe_allow_html=True)
    col_prompt, col_output = st.columns([1, 1.4], gap="large")
     
    with col_prompt:
        st.markdown("<div style='font-size: 0.8rem; color: #00f0ff; font-weight: bold; margin-bottom: 5px;'>1. THE PROMPT</div>", unsafe_allow_html=True)
        
        prompt = st.text_area("prompt_box", height=150, placeholder="Type your scene prompt here...", label_visibility="collapsed")

        # GENRE REGISTER (Multi-select)
        st.markdown("<div style='font-size: 0.8rem; color: #00f0ff; font-weight: bold; margin: 15px 0 5px 0;'>2. GENRE REGISTER (Select Multiple)</div>", unsafe_allow_html=True)
        
        selected_registers = st.multiselect(
            "register_box", 
            ["comedy", "suspense", "thriller", "drama", "horror", "feel-good", "slice-of-life"], 
            default=["comedy"],
            label_visibility="collapsed"
        )
        
        if selected_registers:
            reg_string = ", ".join(selected_registers)
            if f"[REGISTER: {reg_string}]" not in prompt:
                prompt = re.sub(r"\[REGISTER:[^\]]+\]", "", prompt).strip()
                prompt += f"\n\n[REGISTER: {reg_string}]"

        # DIRECTOR'S CUT (Dynamic Dropdowns)
        st.markdown("<div style='font-size: 0.8rem; color: #00f0ff; font-weight: bold; margin: 15px 0 5px 0;'>3. DIRECTOR'S CUT (OPTIONAL)</div>", unsafe_allow_html=True)
        
        techniques_data = []
        techniques_file = BASE_DIR / "scene_techniques.json"
        if techniques_file.exists():
            try:
                raw_data = json.loads(techniques_file.read_text(encoding="utf-8"))
                if isinstance(raw_data, dict):
                    techniques_data = raw_data.get("techniques", [])
                elif isinstance(raw_data, list):
                    techniques_data = raw_data
                techniques_data = [t for t in techniques_data if isinstance(t, dict)]
            except Exception as e:
                print(f"Error loading techniques: {e}")

        opening_options = ["[Auto-Select]"] + [t.get("type", "Unknown") for t in techniques_data if t.get("position") == "opening"]
        closing_options = ["[Auto-Select]"] + [t.get("type", "Unknown") for t in techniques_data if t.get("position") == "closing"]

        col_tech1, col_tech2 = st.columns(2)
        with col_tech1:
            chosen_open = st.selectbox("Opening Technique", opening_options)
        with col_tech2:
            chosen_close = st.selectbox("Closing Technique", closing_options)

        final_gen_prompt = prompt
        if chosen_open != "[Auto-Select]":
            final_gen_prompt += f"\n[REQUIREMENT: Must open using technique: '{chosen_open}']"
        if chosen_close != "[Auto-Select]":
            final_gen_prompt += f"\n[REQUIREMENT: Must close using technique: '{chosen_close}']"

        render_transparency_box(final_gen_prompt, key_suffix="write")

        # GENERATE BUTTON
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("✨ GENERATE SCREENPLAY", type="primary", use_container_width=True):
            if final_gen_prompt.strip():
                # Live Status Box
                status_box = st.status("🎬 Starting generation...", expanded=True)
                
                try:
                    status_box.write("🔍 Searching RAG context, DNA, and Style Patterns...")
                    status_box.write("✍️ Drafting screenplay (Gemini 3.8 Flash)...")
                    
                    result = backend.chat_with_ai(final_gen_prompt, mode="creative")
                    
                    if not result or len(result.strip()) < 10:
                        status_box.update(label="❌ Generation failed (Empty response)", state="error", expanded=False)
                        st.error("The model returned an empty response or hit rate limits.")
                    else:
                        status_box.update(label="✅ Scene Generated & Reviewed!", state="complete", expanded=False)
                        st.session_state.generations.insert(0, {
                            "timestamp": datetime.now().strftime("%H:%M:%S"), 
                            "text": result, 
                            "prompt": final_gen_prompt, 
                            "type": "scene"
                        })
                        st.rerun()
                except Exception as e:
                    status_box.update(label="⚠️ Error / Rate Limited", state="error", expanded=False)
                    st.error(f"Error: {e}. Check your Gemini API limits (5 RPM for 3.8 Flash).")
            else:
                st.warning("Please enter a prompt first.")

    with col_output:
        if not st.session_state.generations:
            st.info("Generated scenes will appear here.")
            
        for idx, gen in enumerate(st.session_state.generations):
            if gen.get("type") == "scene":
                # --- LOCAL NLP DIRECTOR DASHBOARD ---
                try:
                    audit = nlp_director.audit_scene(gen["text"])
                    d_score = audit["dialect"]["authenticity_score"]
                    d_ratio = audit["pacing"]["dialogue_ratio"]
                    a_ratio = audit["pacing"]["action_ratio"]
                    p_label = audit["pacing"]["pacing_label"]
                    v_status = audit["voice"]["status"]
                    
                    st.markdown(f'<div class="poster-card" style="height: 50px; background: #0b0b12; border-radius: 8px 8px 0 0; border:1px solid #1a1a2e; padding: 15px;"><h4 style="margin:0; font-size:1rem; color:#fff;">Generated Scene · {gen["timestamp"]} <span style="font-size:0.7rem; color:#00f0ff; margin-left:10px;">{p_label}</span></h4></div>', unsafe_allow_html=True)
                    
                    st.markdown(
                        f"""
                        <div style="background:#0e0e13; border:1px solid #1c1c28; border-left:4px solid #00f0ff; border-radius:8px; padding:8px 12px; margin-bottom:10px; font-size:0.75rem; font-family:'JetBrains Mono', monospace;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                <span>🗣️ <b>Cadence:</b> {d_score}% ({audit['dialect']['verdict']})</span>
                                <span>⏱️ <b>Rhythm:</b> {d_ratio}% Dial / {a_ratio}% Act</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; color:#8a8a98;">
                                <span>👥 <b>Voice Contrast:</b> {v_status}</span>
                                <span>📍 <b>Particles:</b> {audit['dialect']['regional_particle_count']} found</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if audit["dialect"]["bookish_terms_detected"]:
                        st.warning(f"⚠️ Formal words detected: {', '.join(audit['dialect']['bookish_terms_detected'])}")
                    if audit["pacing"]["monologue_flags"]:
                        st.info(f"⏳ Long monologue alerts: {', '.join(audit['pacing']['monologue_flags'])}")
                except Exception as e:
                    print(f"NLP Director skipped: {e}")
                # ------------------------------------

                edited_text = st.text_area(f"edit_{idx}", value=gen["text"], height=400, label_visibility="collapsed")
                st.session_state.generations[idx]["text"] = edited_text
                
                st.markdown("<div style='font-size: 0.75rem; color: #ffa94d; margin-top: 10px; font-weight: bold;'>DECISION CAPTURE: Did you edit the scene? Tell the AI why so it learns your logic.</div>", unsafe_allow_html=True)
                decision_note = st.text_input(f"decision_note_{idx}", placeholder="e.g., 'I changed the ending because he should never confess.'", label_visibility="collapsed", key=f"dec_{idx}")
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("✏️ Refine", key=f"r_{idx}", use_container_width=True):
                        st.session_state[f"show_refine_{idx}"] = not st.session_state.get(f"show_refine_{idx}", False)
                with c2:
                    if st.button("💾 Save", key=f"s_{idx}", use_container_width=True):
                        st.session_state[f"show_save_{idx}"] = not st.session_state.get(f"show_save_{idx}", False)
                with c3:
                    if st.button("📝 Save Logic", key=f"n_{idx}", type="primary", use_container_width=True):
                        if decision_note.strip():
                            backend.save_feedback(edited_text, decision_note)
                            st.success("Logic saved to AI memory.")
                        else:
                            st.warning("Please type why you edited it first.")
                with c4:
                    if st.button("🗑️ Drop", key=f"d_{idx}", use_container_width=True):
                        st.session_state.generations.pop(idx)
                        st.rerun()

                if st.session_state.get(f"show_refine_{idx}"):
                    with st.form(f"refine_form_{idx}"):
                        refine_instr = st.text_input("What to change?")
                        if st.form_submit_button("Regenerate"):
                            with st.spinner("Refining..."):
                                res = backend.chat_with_ai(f"Original:\n{edited_text}\n\nRefinement:\n{refine_instr}", mode="creative")
                                st.session_state.generations[idx]["text"] = res
                                st.session_state[f"show_refine_{idx}"] = False
                                st.rerun()

                if st.session_state.get(f"show_save_{idx}"):
                    with st.form(f"save_form_{idx}"):
                        f_name = st.text_input("Filename", value=f"scene_draft_{idx}")
                        if st.form_submit_button("Save to Corpus"):
                            (EXAMPLES_DIR / f"{f_name}.txt").write_text(edited_text, encoding="utf-8")
                            st.success(f"Saved {f_name}.txt")
                            st.session_state[f"show_save_{idx}"] = False

# --- TAB 2: BRAINSTORM ---
with tab_brainstorm:
    st.markdown('<div id="brainstorm"></div>', unsafe_allow_html=True)
    col_brainstorm_in, col_brainstorm_out = st.columns([1, 1], gap="large")
    
    with col_brainstorm_in:
        st.markdown("<div style='font-size: 0.8rem; color: #00f0ff; font-weight: bold; margin-bottom: 5px;'>WHAT DO YOU NEED IDEAS FOR?</div>", unsafe_allow_html=True)
        brainstorm_mode = st.selectbox(
            "brainstorm_mode",
            [
                "General Plot / Ending Ideas",
                "🪝 Scene Openings (Sensory Shocks, etc.)",
                "🔚 Scene Closings (Buttons, Irony, etc.)"
            ],
            label_visibility="collapsed"
        )
        
        brainstorm_idea = st.text_area("brainstorm_box", height=150, placeholder="e.g., I need an ending for a scene where a kid gets caught stealing...", label_visibility="collapsed")
        
        st.markdown("<div style='font-size: 0.8rem; color: #00f0ff; font-weight: bold; margin: 15px 0 5px 0;'>PASTE FULL SCRIPT CONTEXT</div>", unsafe_allow_html=True)
        context_script = st.text_area("context_box", height=250, placeholder="Paste the full script here...", label_visibility="collapsed")

        render_transparency_box(brainstorm_idea, key_suffix="brainstorm")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 BRAINSTORM WITH AI", type="primary", use_container_width=True):
            if brainstorm_idea.strip():
                if brainstorm_mode == "🪝 Scene Openings (Sensory Shocks, etc.)":
                    final_prompt = f"[TASK: Brainstorming Scene Openings]\n\nCONTEXT:\n{context_script}\n\nSCENE SITUATION:\n{brainstorm_idea}\n\nGive me 3 completely different visual/action ways to OPEN this scene based on Manish's style."
                elif brainstorm_mode == "🔚 Scene Closings (Buttons, Irony, etc.)":
                    final_prompt = f"[TASK: Brainstorming Scene Closings]\n\nCONTEXT:\n{context_script}\n\nSCENE SITUATION:\n{brainstorm_idea}\n\nGive me 3 completely different visual/action ways to CLOSE this scene based on Manish's style."
                else:
                    final_prompt = f"[TASK: Brainstorming]\n\nCONTEXT:\n{context_script}\n\nSTUCK ON:\n{brainstorm_idea}"
                
                with st.spinner("Brainstorming..."):
                    result = backend.chat_with_ai(final_prompt, mode="brainstorm")
                if result and len(result.strip()) > 10:
                    st.session_state.generations.insert(0, {"timestamp": datetime.now().strftime("%H:%M:%S"), "text": result, "type": "brainstorm"})
                    st.session_state.show_journal = False
                    st.rerun()
            else:
                st.warning("Please enter what you want to brainstorm.")
                
    with col_brainstorm_out:
        brainstorm_results = [g for g in st.session_state.generations if g.get("type") == "brainstorm"]
        
        if brainstorm_results:
            current_idea = brainstorm_results[0]
            st.markdown("<div style='font-size: 0.8rem; color: #00f0ff; font-weight: bold; margin-bottom: 10px;'>LATEST IDEAS</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='markdown-box'>{current_idea['text']}</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='font-size: 0.75rem; color: #ffa94d; margin-top: 15px; font-weight: bold;'>CAPTURE THIS IDEA:</div>", unsafe_allow_html=True)
            
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("✍️ Draft This Idea", use_container_width=True):
                    st.session_state.draft_prompt = f"Write a scene based on this specific concept:\n\n{current_idea['text']}"
                    st.session_state.active_tab = 0
                    st.rerun()
            with b2:
                if st.button("📓 Log Decision", type="primary", use_container_width=True):
                    st.session_state.show_journal = not st.session_state.show_journal
            with b3:
                if st.button("🗑️ Drop", key="drop_latest_brain", use_container_width=True):
                    st.session_state.generations.remove(current_idea)
                    st.session_state.show_journal = False
                    st.rerun()

            if st.session_state.show_journal:
                with st.form("journal_form"):
                    st.markdown("<div class='journal-box'>", unsafe_allow_html=True)
                    st.markdown("### 🧠 Decision Journal")
                    st.markdown("Explain your logic so the AI learns *how* you evaluate ideas.")
                    
                    chosen_concept = st.selectbox("Which concept did you choose?", ["Concept 1", "Concept 2", "Concept 3"])
                    
                    why_chosen = st.text_area("Why did you choose it? (Emotion, vibe, structure)", 
                                              placeholder="e.g., I chose this because the moral outrage hides his theft. It keeps the vibe chaotic but grounded.")
                    
                    why_rejected = st.text_area("Why did you reject the other two?", 
                                                placeholder="e.g., Concept 1 was too passive. Concept 2 relied on a coincidence.")
                    
                    if st.form_submit_button("Save Decision Journal", type="primary"):
                        if why_chosen.strip() and why_rejected.strip():
                            journal_entry = {
                                "reasoning": f"CHOSEN: {chosen_concept}\n\nWHY CHOSEN: {why_chosen}\n\nWHY REJECTED OTHERS: {why_rejected}"
                            }
                            backend.save_feedback(current_idea['text'], journal_entry)
                            st.success("Decision Journal saved to AI memory!")
                            st.session_state.show_journal = False
                            st.rerun()
                        else:
                            st.warning("Please fill out both fields.")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Your brainstormed ideas will appear here.")

# --- TAB 3: DOCTOR ---
with tab_diag:
    st.markdown('<div id="scene-doctor"></div>', unsafe_allow_html=True)
    st.markdown("#### 🩺 Scene Doctor")
    c1, c2 = st.columns(2)
    with c1: stuck_script = st.text_area("Paste script...", height=200)
    with c2: stuck_query = st.text_area("What's the problem?", height=200)
    if st.button("Run Diagnosis", type="primary"):
        with st.spinner("Diagnosing..."):
            diag = backend.diagnose_stuck_scene(stuck_script, stuck_query)
            st.markdown(f"<div class='markdown-box' style='border-left: 4px solid #00f0ff;'>{diag}</div>", unsafe_allow_html=True)

# ============================================================
# THE TRUE ANIMATED JAVASCRIPT CAROUSEL
# ============================================================
st.markdown("""
<div class="scripts-title-container">
    <div class="scripts-bg">SCRIPTS</div>
    <div class="scripts-fg">THE SCRIPTS</div>
</div>
""", unsafe_allow_html=True)

scenes = sorted(EXAMPLES_DIR.glob("*.txt"))

if scenes:
    scene_data = []
    for f in scenes:
        text = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"\[REGISTER:\s*([^\]]+)\]", text, re.IGNORECASE)
        reg = m.group(1).strip().upper() if m else "UNTAGGED"
        
        poster_jpg = POSTERS_DIR / f"{f.stem}.jpg"
        poster_png = POSTERS_DIR / f"{f.stem}.png"
        img_base64 = get_base64_image(poster_jpg) or get_base64_image(poster_png)
        bg_image = f"url('data:image/jpeg;base64,{img_base64}')" if img_base64 else "linear-gradient(180deg, #140a24 0%, #0b0b12 100%)"
        
        scene_data.append({
            "title": f.stem.replace('_', ' ').upper(),
            "register": reg,
            "image": bg_image,
            "filename": f.name
        })
        
    json_data = json.dumps(scene_data)

    carousel_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
        body {{ margin: 0; background: transparent; font-family: 'Outfit', sans-serif; overflow: hidden; }}
        
        .carousel-container {{
            position: relative; width: 100%; height: 480px;
            display: flex; justify-content: center; align-items: flex-start; padding-top: 20px;
        }}
        .carousel-track {{
            position: relative; width: 100%; height: 380px;
            display: flex; justify-content: center; align-items: center;
        }}
        .card {{
            position: absolute; width: 250px; height: 360px;
            background-color: #0b0b12; border-radius: 16px; border: 1px solid #1a1a2e;
            display: flex; flex-direction: column; transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            cursor: pointer; box-shadow: 0 10px 25px rgba(0,0,0,0.6); overflow: hidden;
            opacity: 0; pointer-events: none;
        }}
        
        .card-image {{
            height: 240px; background-size: cover; background-position: center; position: relative;
        }}
        .card-image::after {{
            content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 60px;
            background: linear-gradient(to top, #0b0b12 0%, transparent 100%);
        }}

        .read-action {{
            position: absolute; top: 12px; right: 12px;
            width: 32px; height: 32px; border-radius: 50%;
            background: rgba(11, 11, 18, 0.7); backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            display: flex; justify-content: center; align-items: center;
            color: #fff; font-size: 0.8rem; transition: all 0.3s ease;
            opacity: 0; transform: translateY(-5px); z-index: 10;
        }}
        .card.center:hover .read-action {{
            opacity: 1; transform: translateY(0);
            background: #ff007f; border-color: #ff007f; box-shadow: 0 0 15px rgba(255,0,127,0.6);
        }}

        .card-info {{
            height: 120px; padding: 1rem 1.2rem; background: #0b0b12;
            display: flex; flex-direction: column; justify-content: space-between;
            border-top: 1px solid #1a1a2e; position: relative;
        }}
        
        .card-title {{ 
            margin: 0; font-size: 0.95rem; font-weight: 900; color: #fff; 
            text-transform: uppercase; line-height: 1.25; 
            display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; 
            overflow: hidden; text-shadow: 0 1px 4px rgba(0,0,0,0.8);
        }}
        
        .card-bottom-row {{
            display: flex; justify-content: space-between; align-items: center;
            width: 100%; margin-top: auto;
        }}

        .genre-container {{
            display: flex; align-items: center; gap: 6px;
            font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #a0a0b0; font-weight: 700;
            overflow: hidden; white-space: nowrap; max-width: 140px;
        }}
        
        .badge {{ background: #f5c518; color: #000; padding: 2px 5px; border-radius: 3px; font-weight: 900; flex-shrink: 0; }}
        
        .time-meta {{
            display: flex; align-items: center; gap: 4px;
            font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #a0a0b0; font-weight: 700;
            flex-shrink: 0;
        }}

        .card.center {{
            transform: translateX(0) translateZ(100px) scale(1.15);
            opacity: 1; pointer-events: auto; z-index: 10;
            border-color: #ff007f; box-shadow: 0 0 40px rgba(255,0,127,0.3);
        }}
        .card.left-1 {{ transform: translateX(-160px) translateZ(0) scale(0.9); opacity: 0.4; pointer-events: auto; z-index: 5; }}
        .card.right-1 {{ transform: translateX(160px) translateZ(0) scale(0.9); opacity: 0.4; pointer-events: auto; z-index: 5; }}
        .card.left-2 {{ transform: translateX(-300px) translateZ(-100px) scale(0.75); opacity: 0.1; z-index: 1; }}
        .card.right-2 {{ transform: translateX(300px) translateZ(-100px) scale(0.75); opacity: 0.1; z-index: 1; }}
        .card.left-1:hover, .card.right-1:hover {{ opacity: 0.8; border-color: #00f0ff; }}
        
        .controls {{
            position: absolute; bottom: 0px; left: 50%; transform: translateX(-50%);
            display: flex; gap: 40px; z-index: 20;
        }}
        .nav-btn {{
            background: #0b0b12; border: 1px solid #2a2a3a; color: #a0a0b0; border-radius: 50%;
            width: 45px; height: 45px; font-size: 1.1rem; cursor: pointer; transition: 0.3s;
            display: flex; justify-content: center; align-items: center;
        }}
        .nav-btn:hover {{ background: #ff007f; border-color: #ff007f; color: #fff; transform: scale(1.1); }}
    </style>
    </head>
    <body>
        <div class="carousel-container">
            <div class="carousel-track" id="track"></div>
            <div class="controls">
                <button class="nav-btn" onclick="slide(-1)">◀</button>
                <button class="nav-btn" onclick="slide(1)">▶</button>
            </div>
        </div>

        <script>
            const data = {json_data};
            let currentIndex = 0;
            const track = document.getElementById('track');

            function init() {{
                data.forEach((item, i) => {{
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.onclick = () => {{
                        if (card.classList.contains('center')) {{
                            window.parent.postMessage({{ isStreamlitMessage: true, type: "OPEN_SCRIPT", filename: item.filename }}, "*");
                        }} else {{
                            currentIndex = i;
                            updateClasses();
                        }}
                    }};
                    
                    card.innerHTML = `
                        <div class="card-image" style="background-image: ${{item.image}};">
                            <div class="read-action">▶</div>
                        </div>
                        <div class="card-info">
                            <h4 class="card-title">${{item.title}}</h4>
                            <div class="card-bottom-row">
                                <div class="genre-container"><span class="badge">PRO</span> ${{item.register}}</div>
                                <div class="time-meta">⏱ 2h 15m</div>
                            </div>
                        </div>
                    `;
                    track.appendChild(card);
                }});
                updateClasses();
            }}

            function slide(dir) {{
                currentIndex = (currentIndex + dir + data.length) % data.length;
                updateClasses();
            }}

            function updateClasses() {{
                const cards = document.querySelectorAll('.card');
                const total = cards.length;
                
                cards.forEach((card, i) => {{
                    card.className = 'card';
                    let diff = i - currentIndex;
                    if (diff < -Math.floor(total/2)) diff += total;
                    if (diff > Math.floor(total/2)) diff -= total;
                    
                    if (diff === 0) card.classList.add('center');
                    else if (diff === -1) card.classList.add('left-1');
                    else if (diff === 1) card.classList.add('right-1');
                    else if (diff === -2) card.classList.add('left-2');
                    else if (diff === 2) card.classList.add('right-2');
                }});
            }}

            init();
        </script>
    </body>
    </html>
    """
    
    components.html(carousel_html, height=520)

# ============================================================
# THE BRIDGE: LISTEN FOR CLICK & OPEN SCRIPT
# ============================================================
query_params = st.query_params
if "open_script" in query_params:
    filename = query_params["open_script"]
    target_file = EXAMPLES_DIR / filename
    if target_file.exists():
        st.session_state.selected_script = target_file
        st.query_params.clear()
        st.rerun()

# ============================================================
# SCRIPT READER MODAL
# ============================================================
if st.session_state.selected_script:
    st.markdown("<br><hr style='border-color: #1a1a2e;'><br>", unsafe_allow_html=True)
    with st.container():
        st.markdown(f"### 📖 Reading: {st.session_state.selected_script.stem}")
        content = st.session_state.selected_script.read_text(encoding="utf-8", errors="replace")
        st.text_area("reader", content, height=500, disabled=True, label_visibility="collapsed")
        if st.button("Close Reader"):
            st.session_state.selected_script = None
            st.rerun()