"""
Local NLP Director Engine for CineScript Studio.
Analyzes screenplay drafts in real-time with ZERO API overhead:
  1. Dialect & Cadence Analytics (Hyderabad Tanglish CSR, regional markers, bookish Telugu detection).
  2. Rhythm & Pacing Profiling (Dialogue-to-Action ratio, staccato retorts, monologue drag).
  3. Character Voice Divergence (Semantic distance per speaker via local Vyakyarth embeddings).
  4. Narrative Continuity & State Ledger (Props, active lies, dramatic secrets across scenes).
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "narrative_state.json"

# Regional particles common in conversational Hyderabad Telugu
REGIONAL_MARKERS = {
    "ra", "arra", "bey", "rey", "ga", "kada", "le", "enti", "abba",
    "ante", "kadaa", "laaga", "ippudu", "malli", "enduku", "chudu",
    "padha", "avuna", "nijama", "sare", "bava", "mama", "anna"
}

# Overly formal / Graanthika Telugu words that break natural gully dialogue
BOOKISH_TERMS = {
    "mariyu", "kaavuna", "yokka", "anduvalla", "ayitheenoo",
    "praapthinchu", "uthsavamu", "nivaasamu", "adhikaaramu", "gruhamu"
}

# Screenplay block parser patterns
CHARACTER_HEADING_RE = re.compile(r"^\s*([A-Z0-9_\s\(\)\'\.\-]+)\s*$", re.MULTILINE)
SCENE_HEADING_RE = re.compile(r"^\s*(?:INT|EXT)[\.\s]", re.IGNORECASE | re.MULTILINE)


# ============================================================
# 1. PARSING & EXTRACTION HELPERS
# ============================================================

def parse_screenplay_blocks(script_text: str) -> dict:
    """
    Splits screenplay text into Action and Dialogue components,
    and groups lines by Character Name.
    """
    lines = script_text.strip().split("\n")
    character_dialogues = {}
    action_lines = []
    dialogue_lines = []

    current_char = None
    accumulated_dialogue = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_char and accumulated_dialogue:
                spoken = " ".join(accumulated_dialogue).strip()
                character_dialogues.setdefault(current_char, []).append(spoken)
                dialogue_lines.append(spoken)
                current_char = None
                accumulated_dialogue = []
            continue

        # Skip scene headings
        if SCENE_HEADING_RE.match(stripped):
            current_char = None
            accumulated_dialogue = []
            continue

        # Check if line is character cue (Uppercase, short, no punctuation ending)
        is_char_header = (
            stripped.isupper()
            and len(stripped.split()) <= 4
            and not stripped.endswith((".", ",", "!", "?"))
            and not stripped.startswith("(")
        )

        if is_char_header:
            if current_char and accumulated_dialogue:
                spoken = " ".join(accumulated_dialogue).strip()
                character_dialogues.setdefault(current_char, []).append(spoken)
                dialogue_lines.append(spoken)
            current_char = re.sub(r"\(.*?\)", "", stripped).strip()
            accumulated_dialogue = []
        elif current_char:
            # Skip parentheticals
            if not (stripped.startswith("(") and stripped.endswith(")")):
                accumulated_dialogue.append(stripped)
        else:
            action_lines.append(stripped)

    # Flush tail
    if current_char and accumulated_dialogue:
        spoken = " ".join(accumulated_dialogue).strip()
        character_dialogues.setdefault(current_char, []).append(spoken)
        dialogue_lines.append(spoken)

    return {
        "character_dialogues": character_dialogues,
        "action_lines": action_lines,
        "dialogue_lines": dialogue_lines,
    }


# ============================================================
# 2. DIALECT & CADENCE ENGINE
# ============================================================

def analyze_dialect_and_cadence(dialogue_lines: list[str]) -> dict:
    """Calculates Tanglish ratio, regional particle density, and checks for bookish terms."""
    full_text = " ".join(dialogue_lines).lower()
    tokens = re.findall(r"[a-z0-9\']+", full_text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return {
            "authenticity_score": 50,
            "regional_particle_count": 0,
            "bookish_terms_detected": [],
            "verdict": "Insufficient dialogue",
        }

    # Count regional particle markers
    particle_count = sum(1 for tok in tokens if tok in REGIONAL_MARKERS)

    # Detect bookish / Graanthika terms
    bookish_found = [tok for tok in tokens if tok in BOOKISH_TERMS]

    # Calculate particle density per 100 words
    density = (particle_count / total_tokens) * 100

    # Score calculation (Target: 4% to 12% particle density)
    if 4.0 <= density <= 15.0:
        auth_score = min(98, 80 + int(density * 1.5))
    elif density > 15.0:
        auth_score = 75  # Overly exaggerated slang
    else:
        auth_score = max(35, int(density * 15))

    if bookish_found:
        auth_score = max(20, auth_score - (len(bookish_found) * 15))

    if auth_score >= 80:
        verdict = "Authentic Hyderabad Cadence"
    elif auth_score >= 60:
        verdict = "Acceptable Conversational Flow"
    else:
        verdict = "Overly Formal or Mechanical Dialogue"

    return {
        "authenticity_score": auth_score,
        "regional_particle_count": particle_count,
        "bookish_terms_detected": list(set(bookish_found)),
        "verdict": verdict,
    }


# ============================================================
# 3. PACING & RHYTHM PROFILER
# ============================================================

def analyze_pacing_and_rhythm(parsed_script: dict) -> dict:
    """Measures Dialogue vs Action balance, monologue drag, and retort speed."""
    dialogue_words = sum(len(line.split()) for line in parsed_script["dialogue_lines"])
    action_words = sum(len(line.split()) for line in parsed_script["action_lines"])
    total_words = dialogue_words + action_words

    if total_words == 0:
        return {
            "dialogue_ratio": 50,
            "action_ratio": 50,
            "pacing_label": "Empty Scene",
            "monologue_flags": [],
        }

    d_ratio = int((dialogue_words / total_words) * 100)
    a_ratio = 100 - d_ratio

    # Flag dialogue turns that drag over 35 words (comedic timing killer)
    monologue_warnings = []
    for char, turns in parsed_script["character_dialogues"].items():
        for idx, turn in enumerate(turns):
            w_len = len(turn.split())
            if w_len > 35:
                monologue_warnings.append(f"{char} Turn #{idx+1} ({w_len} words)")

    # Pacing characterization
    if d_ratio >= 75:
        pacing_label = "Rapid-Fire Banter / Argument"
    elif 50 <= d_ratio < 75:
        pacing_label = "Balanced Scene Dynamics"
    elif 30 <= d_ratio < 50:
        pacing_label = "Visual / Tension-Driven"
    else:
        pacing_label = "Atmospheric / Heavy Action"

    return {
        "dialogue_ratio": d_ratio,
        "action_ratio": a_ratio,
        "pacing_label": pacing_label,
        "monologue_flags": monologue_warnings,
    }


# ============================================================
# 4. CHARACTER VOICE DIVERGENCE (via Local Vyakyarth)
# ============================================================

def analyze_character_voice_separation(character_dialogues: dict) -> dict:
    """
    Computes semantic distance between primary speakers to detect
    the 'mono-voice' trap using local Vyakyarth embeddings.
    """
    # Filter speakers with at least 2 lines
    active_speakers = {
        char: " ".join(lines)
        for char, lines in character_dialogues.items()
        if len(lines) >= 2 and len(" ".join(lines).split()) >= 15
    }

    if len(active_speakers) < 2:
        return {
            "status": "Single Primary Speaker",
            "separation_score": 100,
            "details": "Not enough speakers to calculate voice contrast.",
        }

    try:
        from rag_memory import embed_texts, cosine_similarity

        names = list(active_speakers.keys())[:3]  # Compare top speakers
        texts = [active_speakers[n] for n in names]
        vectors = embed_texts(texts)

        # Pairwise comparison
        similarities = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = cosine_similarity(vectors[i], vectors[j])
                similarities.append((names[i], names[j], sim))

        avg_similarity = sum(s[2] for s in similarities) / len(similarities)
        divergence_score = max(0, int((1.0 - avg_similarity) * 100))

        if avg_similarity >= 0.88:
            verdict = "Mono-Voice Risk (Characters sound too similar)"
        elif avg_similarity >= 0.70:
            verdict = "Moderate Contrast"
        else:
            verdict = "Strong Distinct Voices"

        return {
            "status": verdict,
            "separation_score": divergence_score,
            "avg_similarity": round(avg_similarity, 3),
            "pairs": [f"{s[0]} vs {s[1]}: {s[2]:.2f}" for s in similarities],
        }

    except Exception as exc:
        return {
            "status": "Divergence Engine Idle",
            "separation_score": 70,
            "details": f"Embedding comparison skipped: {exc}",
        }


# ============================================================
# 5. NARRATIVE STATE LEDGER
# ============================================================

def load_narrative_state() -> dict:
    if not STATE_FILE.exists():
        default_state = {
            "active_lies": [],
            "carried_props": [],
            "unresolved_conflicts": []
        }
        STATE_FILE.write_text(json.dumps(default_state, indent=2), encoding="utf-8")
        return default_state
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"active_lies": [], "carried_props": [], "unresolved_conflicts": []}


def save_narrative_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def add_narrative_state_item(category: str, detail: str):
    state = load_narrative_state()
    if category in state and detail.strip():
        if detail.strip() not in state[category]:
            state[category].append(detail.strip())
            save_narrative_state(state)


# ============================================================
# 6. MASTER AUDIT FUNCTION
# ============================================================

def audit_scene(script_text: str) -> dict:
    """Executes the complete local directorial NLP suite in under 0.05 seconds."""
    parsed = parse_screenplay_blocks(script_text)
    dialect_metrics = analyze_dialect_and_cadence(parsed["dialogue_lines"])
    pacing_metrics = analyze_pacing_and_rhythm(parsed)
    voice_metrics = analyze_character_voice_separation(parsed["character_dialogues"])
    narrative_state = load_narrative_state()

    return {
        "dialect": dialect_metrics,
        "pacing": pacing_metrics,
        "voice": voice_metrics,
        "state": narrative_state,
        "speaker_count": len(parsed["character_dialogues"]),
    }