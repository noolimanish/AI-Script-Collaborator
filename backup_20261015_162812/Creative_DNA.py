"""
Virtual Manish — Creative DNA
Personal creative collaborator for Manish.

Layout of this file:
  1. Configuration and constants
  2. System instructions (the AI's role)
  3. File / database initialization
  4. Memory: preferences, films, feedback
  5. Semantic duplicate detection and consolidation
  6. Learning: scene study (produces observations, not preferences)
  7. Context building for the generation prompt
  8. Retrieval (semantic only; no keyword fallback)
  9. Spelling-correction mode
 10. Creative chat with generate -> review -> retry loop
 11. Command handling
 12. Main loop
"""

import difflib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from google import genai

# Optional modules — imported lazily inside functions where used,
# so a missing module does not break startup.

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROFILE_FILE = BASE_DIR / "creative_profile.json"
FILM_FILE = BASE_DIR / "film_library.json"
FEEDBACK_FILE = BASE_DIR / "feedback.json"
DATABASE_FILE = BASE_DIR / "virtual_manish.db"
EXAMPLES_DIR = BASE_DIR / "creative_examples"
HISTORY_FILE = BASE_DIR / "creative_scene_history.json"

# Model name is configurable. Verify this exists in your SDK version.
MODEL = os.getenv("VM_MODEL", "gemini-2.0-flash")

# Cap pending lessons so the review queue cannot grow without bound.
MAX_PENDING_LESSONS = 15

# How many prior accepted scenes to compare against.
HISTORY_COMPARE_WINDOW = 40
HISTORY_SAVE_LIMIT = 40

# Number of generation attempts (initial + retries).
MAX_GENERATION_ATTEMPTS = 3

DEBUG = os.getenv("VM_DEBUG", "0") == "1"

client = genai.Client()


# ============================================================
# 2. SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are Virtual Manish, Manish's personal creative collaborator.

YOUR MAIN PURPOSE
Understand how Manish thinks, writes, creates characters, develops scenes,
evaluates dialogue, and makes creative decisions.

Your job is not merely to produce writing that sounds vaguely similar to
Manish. Learn from his actual writing, understand the creative mechanisms
behind it, and help him develop original work that fits his taste.

MANISH'S LANGUAGE AND DIALOGUE STYLE
- Preserve his natural Hyderabad-style Telugu and Telugu-English mixing.
- Correct spelling when requested, but do not rewrite dialogue unnecessarily.
- Do not replace his slang with formal Telugu.
- Preserve sentence rhythm, word connections, expressions, and conversational flow.
- Do not translate English into Telugu mechanically.
- Do not make dialogue sound like generic movie dialogue.
- Do not invent slang and claim it is Manish's style.
- Treat his actual writing samples and explicit corrections as evidence of his voice.
- Different characters must have different ways of speaking, based on personality,
  background, relationships, confidence, education, and situation.
- Do not make every character speak like Manish or like the same generic hero.

USING CREATIVE EXAMPLES
- Retrieved passages are evidence and references, not templates to copy.
- Learn the underlying mechanism: character motivation, blind spots, behaviour,
  conflict, timing, reactions, relationships, and consequences.
- Never assume that matching the surface details means matching Manish's style.
- Do not reuse a reference scene's names, distinctive lines, props, excuses,
  jokes, or sequence of events in an unrelated scene.
- Do not reuse a memorable phrase just because it appeared in a retrieved passage.
- Reuse specific wording only when Manish explicitly requests a continuation,
  quotation, revision, or callback.
- When a reference contains a character exposing a flaw through misplaced pride,
  invent a different flaw, situation, and behaviour for the new character.
- If an example is irrelevant to the current request, ignore it.
- An English screenplay may teach storytelling and visual description. It does not
  automatically define Manish's Telugu dialogue style.
- Distinguish Manish's original writing, his explicit feedback, and AI-generated
  suggestions. AI-generated suggestions are not automatically Manish's preferences.

CHARACTER-DRIVEN COMEDY
When developing comedy, consider these questions internally:

1. What does this character want right now?
2. What does the character want other people to believe about them?
3. What flaw, blind spot, insecurity, habit, or selfish priority contradicts that image?
4. What specific situation puts pressure on that contradiction?
5. How does the character's own behaviour accidentally expose the truth?
6. How do the other characters react, and what does that reveal about their relationships?

Use these questions as creative guidance, not as a rigid formula.

- Prefer comedy emerging naturally from behaviour, flawed reasoning, relationships,
  misplaced confidence, awkward decisions, and attempts to recover.
- A character may genuinely believe their own excuse. They do not need to know
  that they are being funny.
- Let actions, pauses, expressions, interruptions, and reactions carry meaning.
- Do not explain a joke after the audience can already understand it.
- Avoid excessive dialogue that spells out the contradiction.
- Do not add a punchline to every exchange.
- Do not force idioms, exaggerated slang, or witty one-liners into every scene.
- Do not automatically add alcohol, rent problems, cigarettes, or familiar
  middle-class struggles just because they appeared in an earlier example.
- Choose details because they belong to this particular character and situation.
- A new scene should have its own dramatic situation, conflict, behaviour, and payoff.

CREATIVE ORIGINALITY CHECK
Before responding to a request for an original scene, silently check:

- Am I repeating the same situation or joke from a retrieved example?
- Am I borrowing a distinctive line instead of learning from its underlying mechanism?
- Is the character's behaviour specific to this person?
- Is the contradiction visible through behaviour rather than explained repeatedly?
- Are the reactions believable for these particular relationships?
- Can I remove a line without losing anything important?

If the draft is too close to a reference, change the situation, character flaw,
actions, dialogue, and payoff before presenting it.

CREATIVE COLLABORATION
- Develop Manish's ideas rather than replacing them with generic alternatives.
- Avoid generic rom-com characters, predictable templates, and forced punchlines.
- Consider story potential, conflict, relationships, character choices, and consequences.
- When Manish asks for evaluation, explain what works, what feels generic, and why.
- Give specific constructive feedback instead of praising everything automatically.
- If Manish is exploring an idea, collaborate before writing an entire scene.
- Ask one clear question at a time when an important detail is missing.
- If you ask a question, wait for the answer instead of silently choosing an assumption.
- If enough context exists to proceed, make a reasonable choice and state it briefly
  only when that assumption materially affects the result.
- Learn from Manish's explicit feedback and corrections.
- Never claim to remember information that is not available in your context.

LEARNING AND MEMORY
- A conversation with Manish is not automatically proof of a permanent preference.
- Your own generated dialogue, examples, suggestions, and interpretations are not
  evidence that Manish wrote or approved those ideas.
- Treat explicit corrections and confirmed preferences as stronger evidence than
  inferred preferences.
- Never claim a lesson is approved unless the application's memory status confirms it.
- When uncertain about Manish's intent, ask instead of inventing a preference.

RESPONSE STYLE
Respond naturally and directly.
Use the amount of detail the task requires.
Do not expose internal analysis or turn every creative response into a checklist.
When writing a scene, prioritize the scene itself over explanations about how it works.
"""


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def now():
    return datetime.now().isoformat(timespec="seconds")


def normalize_text(text):
    text = str(text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def get_text(value):
    """Return a text representation of a preference-like object."""
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return str(
            value.get("text")
            or value.get("note")
            or value.get("preference")
            or value.get("lesson")
            or ""
        )

    return str(value)


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{path.name} contains invalid JSON: {exc}. "
            "Restore it from your backup before continuing."
        )


def save_json(path, data):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    temporary_path.replace(path)


def response_text(response):
    """
    Extract text from a Gemini response. Handles multiple SDK shapes so
    a change in the SDK does not silently break every call.
    """
    if response is None:
        raise RuntimeError("Gemini returned None.")

    # Newer SDK shape.
    output = getattr(response, "output_text", None)
    if isinstance(output, str) and output.strip():
        return output.strip()

    # Classic google-genai shape.
    candidates = getattr(response, "candidates", None)
    if candidates:
        try:
            parts = candidates[0].content.parts
            collected = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    collected.append(text)
            if collected:
                return "\n".join(collected).strip()
        except (AttributeError, IndexError, TypeError):
            pass

    # Last resort: stringify, but log in debug mode.
    if DEBUG:
        print("[DEBUG] Unexpected response shape:")
        print(repr(response))

    raise RuntimeError(
        "Gemini returned no extractable text. "
        "Set VM_DEBUG=1 to inspect the raw response."
    )


def extract_json(text):
    """Extract valid JSON from a model response, tolerating code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(match.group(0))


def call_model(prompt, system_instruction=None):
    """Single entry point for all model calls."""
    kwargs = {
        "model": MODEL,
        "input": prompt,
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    response = client.interactions.create(**kwargs)
    return response_text(response)


# ============================================================
# 4. FILE / DATABASE INITIALIZATION
# ============================================================

def _migrate_preferences(profile):
    """
    Older preferences stored a 'note' field. Canonicalize to 'text'.
    Preserve created_at and any other metadata.
    """
    preferences = profile.get("preferences", [])
    migrated = []

    for item in preferences:
        if isinstance(item, str):
            migrated.append({"text": item, "created_at": now(), "source": "migrated"})
        elif isinstance(item, dict):
            text = item.get("text") or item.get("note") or item.get("preference")
            if not text:
                continue
            new_item = dict(item)
            new_item["text"] = str(text)
            new_item.pop("note", None)
            new_item.pop("preference", None)
            new_item.setdefault("created_at", now())
            new_item.setdefault("source", "migrated")
            migrated.append(new_item)

    profile["preferences"] = migrated
    return profile


def initialize_files():
    defaults = [
        (PROFILE_FILE, {"preferences": [], "preference_evidence": []}),
        (FILM_FILE, {"films": []}),
        (FEEDBACK_FILE, {"examples": []}),
    ]

    for path, default in defaults:
        if not path.exists():
            save_json(path, default)

    profile = load_json(PROFILE_FILE, {"preferences": [], "preference_evidence": []})
    profile.setdefault("preferences", [])
    profile.setdefault("preference_evidence", [])
    profile = _migrate_preferences(profile)
    save_json(PROFILE_FILE, profile)

    films = load_json(FILM_FILE, {"films": []})
    films.setdefault("films", [])
    save_json(FILM_FILE, films)

    feedback = load_json(FEEDBACK_FILE, {"examples": []})
    feedback.setdefault("examples", [])
    save_json(FEEDBACK_FILE, feedback)

    EXAMPLES_DIR.mkdir(exist_ok=True)


def get_connection():
    return sqlite3.connect(DATABASE_FILE)


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                lesson TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                confidence REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL
            )
        """)
    deduplicate_existing_lessons()


def deduplicate_existing_lessons():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, lesson, status FROM lessons ORDER BY id"
        ).fetchall()

        seen = {}
        for lesson_id, lesson, status in rows:
            key = normalize_text(lesson)
            if not key:
                continue
            if key not in seen:
                seen[key] = (lesson_id, status)
                continue
            original_id, original_status = seen[key]
            if status == "active" and original_status != "active":
                connection.execute("DELETE FROM lessons WHERE id = ?", (original_id,))
                seen[key] = (lesson_id, status)
            else:
                connection.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))


def pending_lesson_count():
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM lessons WHERE status = 'pending'"
        ).fetchone()
    return row[0] if row else 0


def save_conversation(role, message):
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO conversations (role, message, created_at) VALUES (?, ?, ?)",
            (role, message, now()),
        )


def save_lesson(category, lesson, source, status, confidence):
    """Insert a lesson. Returns new id, or None if a duplicate exists."""
    key = normalize_text(lesson)

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, lesson FROM lessons WHERE status IN ('active', 'pending')"
        ).fetchall()

        for _, existing in rows:
            if normalize_text(existing) == key:
                return None

        if status == "pending" and pending_lesson_count() >= MAX_PENDING_LESSONS:
            print(
                f"[Memory] Pending queue is full ({MAX_PENDING_LESSONS}). "
                "Approve or reject pending lessons with /pending first."
            )
            return None

        cursor = connection.execute(
            """
            INSERT INTO lessons
                (category, lesson, source, status, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (category, lesson, source, status, confidence, now()),
        )
        return cursor.lastrowid


# ============================================================
# 5. SEMANTIC DUPLICATE DETECTION AND CONSOLIDATION
# ============================================================

CONCEPT_GROUPS = {
    "comedy": {
        "comedy", "funny", "humor", "humour", "laugh", "laughs", "comic", "comedic",
    },
    "character": {
        "character", "characters", "personality", "personalities",
        "behavior", "behaviour", "flaw", "flaws",
    },
    "dialogue": {
        "dialogue", "dialog", "conversation", "conversations",
        "speech", "speaking", "talking",
    },
    "punchline": {
        "punchline", "punchlines", "joke", "jokes", "witty",
        "one-liner", "one-liners",
    },
    "naturalness": {
        "natural", "naturally", "realistic", "real",
        "authentic", "spontaneous", "organic",
    },
    "writing": {
        "writing", "writer", "writers", "story", "stories",
        "screenplay", "scene", "scenes",
    },
    "relationships": {
        "relationship", "relationships", "friendship",
        "friends", "awkward", "awkwardness",
    },
}


def concept_tags(text):
    words = set(re.findall(r"[a-z]+", text.casefold()))
    tags = set()
    for concept, vocabulary in CONCEPT_GROUPS.items():
        if words.intersection(vocabulary):
            tags.add(concept)
    return tags


def local_similarity_candidates(new_text, existing_texts):
    new_words = set(re.findall(r"[a-z]+", new_text.casefold()))
    new_tags = concept_tags(new_text)

    stop_words = {
        "i", "me", "my", "the", "a", "an", "and", "or", "but",
        "is", "are", "be", "to", "of", "in", "on", "for", "with",
        "that", "this", "it", "not", "rather", "than", "prefer",
        "like", "want", "should", "can", "from", "based",
    }

    meaningful_new = new_words - stop_words
    candidates = []

    for index, existing_text in enumerate(existing_texts):
        existing_words = set(re.findall(r"[a-z]+", existing_text.casefold()))
        meaningful_existing = existing_words - stop_words

        shared = meaningful_new & meaningful_existing
        shared_tags = new_tags & concept_tags(existing_text)

        union = meaningful_new | meaningful_existing
        jaccard = len(shared) / len(union) if union else 0

        if shared_tags or len(shared) >= 2 or jaccard >= 0.20:
            candidates.append({
                "index": index,
                "text": existing_text,
                "shared_concepts": sorted(shared_tags),
                "word_similarity": round(jaccard, 3),
            })

    return candidates


def compare_memory_meaning(new_text, existing_text):
    prompt = f"""
Compare these two creative preferences from the same person.

NEW PREFERENCE:
{new_text}

EXISTING PREFERENCE:
{existing_text}

Determine whether they express the SAME CORE PREFERENCE.

- If two preferences share the same core action or rule, return
  same_core_preference: true even if the wording differs significantly.
- Only return false if the new preference adds a genuinely new rule,
  exception, or distinct dimension.
- Treat paraphrases and restatements as duplicates.
- If uncertain, return false.
- Do not rewrite, merge, or delete anything.

Return JSON only:
{{
  "same_core_preference": true,
  "reason": "Brief explanation",
  "new_information": "Additional information, if any"
}}
"""

    raw = call_model(
        prompt,
        system_instruction=(
            "Compare creative memories conservatively. "
            "Return valid JSON only."
        ),
    )
    result = extract_json(raw)

    return {
        "same": result.get("same_core_preference") is True,
        "reason": str(result.get("reason", "")),
        "new_information": str(result.get("new_information", "")),
    }


def remember_preference(preference, source="manual"):
    preference = preference.strip()
    if not preference:
        print("Please provide a preference to remember.")
        return

    profile = load_json(PROFILE_FILE, {"preferences": [], "preference_evidence": []})
    profile.setdefault("preferences", [])
    profile.setdefault("preference_evidence", [])

    preferences = profile["preferences"]
    existing_texts = [get_text(item) for item in preferences]
    normalized_new = normalize_text(preference)

    exact_match = next(
        (t for t in existing_texts if normalize_text(t) == normalized_new),
        None,
    )

    if exact_match:
        print("This exact preference is already saved.")
        existing_texts = [
            t for t in existing_texts if normalize_text(t) != normalized_new
        ]

    candidates = local_similarity_candidates(preference, existing_texts)

    try:
        for candidate in candidates:
            comparison = compare_memory_meaning(preference, candidate["text"])
            if comparison["same"]:
                profile["preference_evidence"].append({
                    "existing_preference": candidate["text"],
                    "new_wording": preference,
                    "source": source,
                    "reason": comparison["reason"],
                    "new_information": comparison["new_information"],
                    "created_at": now(),
                })
                save_json(PROFILE_FILE, profile)
                print("This appears to express an existing preference.")
                print("The original preference was preserved.")
                print("The new wording was saved as evidence.")
                if comparison["new_information"]:
                    print("Additional detail:", comparison["new_information"])
                return
    except Exception as error:
        print("Semantic comparison failed. No existing preference was deleted.")
        print(f"Details: {error}")
        if exact_match:
            print("The exact preference already exists; no change made.")
            return

    if exact_match:
        print("No semantic duplicate was confirmed.")
        print("The exact preference is already saved.")
        return

    preferences.append({
        "text": preference,
        "created_at": now(),
        "source": source,
    })
    save_json(PROFILE_FILE, profile)
    print("Saved to your Creative DNA.")


def consolidate_preferences(preferences, max_groups=8):
    """
    Collapse preferences that share enough concept tags into a single
    representative (the longest wording in each cluster).

    This prevents build_context() from injecting six paraphrases of the
    same rule into the prompt.
    """
    entries = []
    for item in preferences:
        text = get_text(item).strip()
        if not text:
            continue
        entries.append({
            "text": text,
            "tags": concept_tags(text),
            "length": len(text),
        })

    if not entries:
        return []

    clusters = []

    for entry in entries:
        placed = False
        for cluster in clusters:
            # Cluster membership: shares at least 2 concept tags AND
            # has meaningful word overlap (>0.35 Jaccard on content words).
            shared_tags = entry["tags"] & cluster["tags"]
            if len(shared_tags) < 2:
                continue

            a = set(re.findall(r"[a-z]+", entry["text"].casefold()))
            b = set(re.findall(r"[a-z]+", cluster["representative"].casefold()))
            union = a | b
            jaccard = len(a & b) / len(union) if union else 0

            if jaccard >= 0.20:
                cluster["members"].append(entry)
                cluster["tags"] |= entry["tags"]
                if entry["length"] > len(cluster["representative"]):
                    cluster["representative"] = entry["text"]
                placed = True
                break

        if not placed:
            clusters.append({
                "representative": entry["text"],
                "tags": set(entry["tags"]),
                "members": [entry],
            })

    # Sort clusters by number of members (most evidence first), then cap.
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    return [c["representative"] for c in clusters[:max_groups]]


# ============================================================
# 6. FILM ANALYSIS AND FEEDBACK
# ============================================================

def save_film_analysis(title, analysis):
    title = title.strip()
    analysis = analysis.strip()
    if not title or not analysis:
        print("Provide both a film title and analysis.")
        return

    library = load_json(FILM_FILE, {"films": []})
    library.setdefault("films", [])

    for film in library["films"]:
        if (
            normalize_text(film.get("title", "")) == normalize_text(title)
            and normalize_text(film.get("analysis", "")) == normalize_text(analysis)
        ):
            print("This film analysis is already saved.")
            return

    library["films"].append({
        "title": title,
        "analysis": analysis,
        "created_at": now(),
    })
    save_json(FILM_FILE, library)
    print(f"Saved film analysis: {title}")


def save_feedback(ai_response, correction):
    ai_response = ai_response.strip()
    correction = correction.strip()
    if not ai_response or not correction:
        print("Both the AI response and your correction are required.")
        return

    feedback = load_json(FEEDBACK_FILE, {"examples": []})
    feedback.setdefault("examples", [])

    for example in feedback["examples"]:
        if (
            normalize_text(example.get("ai_response", "")) == normalize_text(ai_response)
            and normalize_text(example.get("correction", "")) == normalize_text(correction)
        ):
            print("This feedback example is already saved.")
            return

    feedback["examples"].append({
        "ai_response": ai_response,
        "correction": correction,
        "created_at": now(),
    })
    save_json(FEEDBACK_FILE, feedback)
    print("Feedback saved.")


# ============================================================
# 7. SCENE STUDY — produces OBSERVATIONS, not preferences
# ============================================================

def study_scene(scene_text, scene_title):
    """
    Analyse Manish's original scene. Lessons are saved as pending
    OBSERVATIONS about the scene, not claims about Manish's preferences.
    """
    scene_text = scene_text.strip()
    scene_title = scene_title.strip() or "Untitled scene"

    if not scene_text:
        print("The scene is empty. Nothing was studied.")
        return

    if pending_lesson_count() >= MAX_PENDING_LESSONS:
        print(
            f"Cannot study more scenes: pending queue is full "
            f"({MAX_PENDING_LESSONS}). Use /pending to review."
        )
        return

    prompt = f"""
Study the following original screenplay scene by Manish.

SCENE TITLE:
{scene_title}

ORIGINAL SCENE:
<scene>
{scene_text}
</scene>

Analyse the writing as a careful creative-writing student.

Look for evidence of:
- Character-specific humour and behaviour.
- Dialogue rhythm and natural Telugu-English mixing.
- Character relationships and reactions.
- Scene construction, entrances, exits, and visual actions.
- Comedy arising from personality, flaws, and situations.

RULES:
- Analyse only what the scene actually supports.
- Do NOT claim these are Manish's permanent preferences. These are
  observations about THIS scene only.
- Do not invent biographical details.
- Do not rewrite the scene.
- Keep each observation specific, useful, and concise.
- Return 3 to 5 observations when the scene provides enough evidence.
- Each observation must describe an observable writing technique
  used IN THIS SCENE.

Return valid JSON only:
{{
    "observations": [
        {{
            "category": "technique_or_observation",
            "observation": "A specific observation supported by the scene.",
            "confidence": 0.75
        }}
    ]
}}
"""

    try:
        raw = call_model(
            prompt,
            system_instruction=(
                "You are a careful screenplay analyst. "
                "Describe only what the scene shows. "
                "Return valid JSON only."
            ),
        )
        result = extract_json(raw)
        observations = result.get("observations", [])

        if not isinstance(observations, list):
            print("The scene analysis returned an invalid observation list.")
            return

        saved_count = 0
        for item in observations:
            if not isinstance(item, dict):
                continue

            category = str(item.get("category", "scene_study")).strip()
            observation = str(item.get("observation", "")).strip()
            if not observation:
                continue

            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            lesson_id = save_lesson(
                category=category,
                lesson=f"[Scene observation: {scene_title}] {observation}",
                source=f"scene_study:{scene_title}",
                status="pending",
                confidence=confidence,
            )
            if lesson_id is not None:
                saved_count += 1
                print(f"Pending observation #{lesson_id}: {observation}")

        print(f"\nScene study finished: {scene_title}")
        print(f"New observations awaiting review: {saved_count}")
        print("Use /pending to review. /approve ID or /reject ID to decide.")

    except Exception as error:
        print(f"Scene study failed: {error}")
        print("Your original scene file was not modified.")


def study_scene_file(filename):
    filename = filename.strip().strip('"').strip("'")
    if not filename:
        print("Provide a scene filename.")
        return

    source_path = EXAMPLES_DIR / filename
    if not source_path.suffix:
        source_path = source_path.with_suffix(".txt")

    if not source_path.is_file():
        print(f"Scene file not found: {source_path}")
        print("Check the filename in creative_examples.")
        return

    try:
        scene_text = source_path.read_text(encoding="utf-8")
        study_scene(scene_text=scene_text, scene_title=source_path.stem)
    except OSError as error:
        print(f"Could not read the scene file: {error}")


# ============================================================
# 8. CONTEXT BUILDING
# ============================================================

def _score_text_against_query(text, query_words):
    if not query_words:
        return 0.0
    text_words = set(re.findall(r"[a-z0-9]+", text.casefold()))
    if not text_words:
        return 0.0
    shared = query_words & text_words
    return len(shared) / len(query_words)


def _query_words(message):
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "with", "is", "are", "i", "my", "me", "how", "what", "can",
        "we", "this", "that", "please", "give", "make", "write",
    }
    words = set(re.findall(r"[a-z0-9]+", message.casefold()))
    return words - stop


def build_context(user_message):
    """
    Build the memory context for the generation prompt.

    - Preferences are consolidated so paraphrases do not repeat.
    - Films and feedback are retrieved by relevance to the query,
      then capped.
    """
    profile = load_json(PROFILE_FILE, {"preferences": []})
    film_library = load_json(FILM_FILE, {"films": []})
    feedback = load_json(FEEDBACK_FILE, {"examples": []})

    # Preferences: consolidate paraphrases.
    raw_preferences = profile.get("preferences", [])
    consolidated = consolidate_preferences(raw_preferences, max_groups=8)

    # Films: retrieve top 3 relevant.
    query_words = _query_words(user_message)
    films = film_library.get("films", [])
    scored_films = sorted(
        films,
        key=lambda f: _score_text_against_query(
            f"{f.get('title', '')} {f.get('analysis', '')}", query_words
        ),
        reverse=True,
    )[:3]

    # Feedback: retrieve top 4 relevant, prefer recent ties.
    examples = feedback.get("examples", [])
    scored_examples = sorted(
        examples,
        key=lambda e: _score_text_against_query(
            f"{e.get('ai_response', '')} {e.get('correction', '')}", query_words
        ),
        reverse=True,
    )[:4]

    # Active lessons from SQLite.
    with get_connection() as connection:
        lessons = connection.execute("""
            SELECT category, lesson, confidence
            FROM lessons
            WHERE status = 'active'
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()

        conversations = connection.execute("""
            SELECT role, message
            FROM conversations
            ORDER BY id DESC
            LIMIT 12
        """).fetchall()

    conversations.reverse()

    return f"""
MANISH'S CONSOLIDATED CREATIVE PREFERENCES:
{json.dumps(consolidated, ensure_ascii=False, indent=2)}

RELEVANT FILM ANALYSES:
{json.dumps(scored_films, ensure_ascii=False, indent=2)}

RELEVANT FEEDBACK EXAMPLES:
{json.dumps(scored_examples, ensure_ascii=False, indent=2)}

ACTIVE LEARNED LESSONS:
{json.dumps(lessons, ensure_ascii=False, indent=2)}

RECENT CONVERSATION:
{json.dumps(conversations, ensure_ascii=False, indent=2)}
"""


# ============================================================
# 9. RETRIEVAL — SEMANTIC ONLY
# ============================================================

def retrieve_approved_lessons(query, limit=5):
    """Retrieve active lessons relevant to the query. Uses fixed schema."""
    try:
        with sqlite3.connect(str(DATABASE_FILE)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, category, lesson, confidence FROM lessons "
                "WHERE status = 'active'"
            ).fetchall()
    except (sqlite3.Error, OSError) as error:
        print(f"[Lesson retrieval] Warning: {error}")
        return []

    query_words = _query_words(query)
    candidates = []

    for row in rows:
        lesson_text = str(row["lesson"] or "").strip()
        if not lesson_text:
            continue

        lesson_words = set(re.findall(r"[a-z0-9]+", lesson_text.casefold()))
        shared = query_words & lesson_words
        score = len(shared) / len(query_words) if query_words else 0.0

        candidates.append({
            "id": row["id"],
            "category": row["category"],
            "text": lesson_text,
            "score": score,
            "matched_words": sorted(shared),
        })

    candidates.sort(key=lambda i: (i["score"], len(i["matched_words"])), reverse=True)

    relevant = [c for c in candidates if c["score"] > 0][:limit]
    if not relevant:
        relevant = candidates[: min(3, limit)]
    return relevant


def retrieve_creative_context(user_message):
    """
    Retrieve relevant creative examples via semantic search ONLY.

    Keyword fallback is intentionally not used in production: keyword
    matches produce systematically different ranking behaviour, and a
    silent fallback makes the system non-deterministic. Set
    VM_ALLOW_KEYWORD_FALLBACK=1 to enable the fallback for debugging.
    """
    examples = []

    try:
        from rag_memory import search_memory
        results = search_memory(user_message, limit=10)
        for item in results:
            examples.append({
                "source": item.get("source", item.get("filename", "Unknown source")),
                "score": float(item.get("score", item.get("similarity", 0.0))),
                "text": item.get("text", item.get("content", "")),
            })
    except Exception as error:
        print(f"[Creative Context] Semantic search unavailable: {error}")

    if not examples and os.getenv("VM_ALLOW_KEYWORD_FALLBACK") == "1":
        try:
            from creative_memory import search_creative_examples
            results = search_creative_examples(user_message, limit=6)
            for item in results:
                if isinstance(item, dict):
                    examples.append({
                        "source": item.get("filename", item.get("source", "Unknown")),
                        "score": float(item.get("score", 0.0)),
                        "text": item.get("text", ""),
                    })
                else:
                    examples.append({"source": "Creative example", "score": 0.0, "text": str(item)})
        except Exception as error:
            print(f"[Creative Context] Keyword fallback unavailable: {error}")

    examples.sort(key=lambda item: item["score"], reverse=True)

    # Keep only the best passage per source file.
    selected_examples = []
    seen_sources = set()
    for item in examples:
        source = item["source"]
        text = item["text"].strip()
        if not text or source in seen_sources:
            continue
        seen_sources.add(source)
        selected_examples.append(item)
        if len(selected_examples) >= 3:
            break

    try:
        lessons = retrieve_approved_lessons(user_message, limit=5)
    except Exception as error:
        print(f"[Creative Context] Lesson retrieval unavailable: {error}")
        lessons = []

    sections = []
    for item in selected_examples:
        sections.append(f"CREATIVE EXAMPLE — {item['source']}\n{item['text']}")

    if lessons:
        lesson_lines = []
        for lesson in lessons:
            if isinstance(lesson, dict):
                category = lesson.get("category", "general")
                text = lesson.get("text", lesson.get("lesson", ""))
                if text:
                    lesson_lines.append(f"- [{category}] {text}")
            else:
                lesson_lines.append(f"- {lesson}")
        if lesson_lines:
            sections.append("APPROVED CREATIVE LESSONS\n" + "\n".join(lesson_lines))

    if not sections:
        return ""

    return (
        "\n\n===== VIRTUAL MANISH RETRIEVED CREATIVE CONTEXT =====\n\n"
        + "\n\n".join(sections)
        + "\n\n===== END RETRIEVED CREATIVE CONTEXT ====="
    )


# ============================================================
# 10. SPELLING-CORRECTION MODE
# ============================================================

CORRECTION_PHRASES = (
    "correct spelling only",
    "only correct spelling",
    "fix spelling only",
    "spelling only",
    "proofread only",
    "don't change wording",
    "do not change wording",
    "don't rewrite",
    "do not rewrite",
    "no rewriting",
    "preserve original dialogue",
    "keep my original dialogue",
    "only fix spelling",
)


def is_correction_request(message):
    text = message.casefold()
    return any(phrase in text for phrase in CORRECTION_PHRASES)


def correct_spelling_only(original_text):
    if not original_text.strip():
        return "No source text was provided."

    prompt = f"""
Correct spelling mistakes in the source text below.

STRICT RULES:
- Correct spelling only when the intended word is clear.
- Preserve every original dialogue line and its order.
- Preserve Telugu slang and Telugu-English mixing.
- Preserve sentence structure, expressions, and wording.
- Do not rewrite, polish, translate, or formalize sentences.
- Do not add or remove dialogue, words, jokes, or descriptions.
- Do not change a word just because another word sounds better.
- If a word is ambiguous, leave it unchanged.
- Return only the corrected text, without explanations.

SOURCE TEXT:
<source>
{original_text}
</source>
"""

    return call_model(
        prompt,
        system_instruction=(
            "You are a conservative proofreader. "
            "Correct clear spelling mistakes only. "
            "Never creatively rewrite source material."
        ),
    )


# ============================================================
# 11. CREATIVE CHAT — generate, review, retry
# ============================================================

def _load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            return []
        return [
            item for item in data
            if isinstance(item, dict) and isinstance(item.get("scene"), str)
        ]
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[History warning] {exc}")
        return []


def _save_history(items):
    try:
        temporary_path = str(HISTORY_FILE) + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(items[-HISTORY_SAVE_LIMIT:], file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, HISTORY_FILE)
        return True
    except OSError as exc:
        print(f"[History save error] {exc}")
        return False


def _needs_creative_review(message):
    return bool(re.search(
        r"\b(scene|screenplay|script|dialogue|comedy|"
        r"write|develop|create|draft|story|character)\b",
        message,
        re.IGNORECASE,
    ))


def _normalize_for_compare(value):
    value = str(value or "").casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def _compare_history(scene, beats, history):
    scene_normalized = _normalize_for_compare(scene)
    beats_normalized = _normalize_for_compare(" ".join(beats))

    for item in history[-HISTORY_COMPARE_WINDOW:]:
        old_scene = _normalize_for_compare(item.get("scene", ""))
        old_beats = _normalize_for_compare(" ".join(item.get("beat_summary", [])))
        if not old_scene:
            continue

        scene_ratio = difflib.SequenceMatcher(None, scene_normalized, old_scene).ratio()
        if scene_ratio >= 0.84:
            return "The scene is too similar to a previously accepted scene."

        if beats_normalized and old_beats:
            beat_ratio = difflib.SequenceMatcher(None, beats_normalized, old_beats).ratio()
            if beat_ratio >= 0.76:
                return "The scene repeats the major beats of a previously accepted scene."

    return None


def _review_scene(scene, original_request, history):
    review_prompt = f"""
You are a strict screenplay editor.

ORIGINAL USER REQUEST:
{original_request}

SCENE TO REVIEW:
{scene}

PREVIOUSLY ACCEPTED SCENE BEATS:
{json.dumps([item.get("beat_summary", []) for item in history[-5:]], ensure_ascii=False)}

Review the ACTUAL SCENE, not the author's intentions.

Reject if:
- It violates an explicit restriction in the original request.
- It repeats a previous scene's causal progression with superficial changes.
- Character behaviour exists only to force a joke or romantic moment.
- The main conflict is abandoned for an unrelated interruption.
- The ending is an unearned generic punchline or humiliation.

Judge the causal sequence, not isolated words. Isolated presence of a
character or a topic is NOT a violation on its own. Violations require the
SEQUENCE to match a forbidden progression or an explicit instruction to be broken.

Return only JSON:
{{
  "verdict": "PASS or FAIL",
  "violations": ["specific problems"],
  "beat_summary": ["actual chronological beat 1", "actual beat 2"],
  "retry_direction": "If FAIL: a specific, structurally different direction for a new attempt. If PASS: empty string."
}}

PASS requires an empty violations list and a non-empty chronological beat summary.
When uncertain about a genuine violation, return FAIL.
"""

    try:
        raw = call_model(review_prompt, system_instruction=SYSTEM_INSTRUCTIONS)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()
        result = json.loads(raw)

        verdict = str(result.get("verdict", "")).upper()
        violations = result.get("violations")
        beats = result.get("beat_summary")
        retry_direction = str(result.get("retry_direction", ""))

        if verdict not in ("PASS", "FAIL"):
            raise ValueError("Invalid verdict.")
        if not isinstance(violations, list) or not all(isinstance(v, str) for v in violations):
            raise ValueError("Invalid violations list.")
        if not isinstance(beats, list) or not all(isinstance(b, str) for b in beats):
            raise ValueError("Invalid beat summary.")
        if not beats:
            raise ValueError("Missing chronological story beats.")
        if verdict == "PASS" and violations:
            verdict = "FAIL"

        return {
            "verdict": verdict,
            "violations": violations,
            "beat_summary": beats,
            "retry_direction": retry_direction,
        }

    except Exception as exc:
        if DEBUG:
            print(f"[Review debug] {exc}")
        return {
            "verdict": "FAIL",
            "violations": ["Review failed or returned invalid JSON."],
            "beat_summary": [],
            "retry_direction": "",
        }


def _generate(prompt, attempt_number):
    try:
        result = call_model(prompt, system_instruction=SYSTEM_INSTRUCTIONS)
        if DEBUG:
            print(f"\n===== GENERATION ATTEMPT {attempt_number} =====\n")
            print(result)
            print("\n===== END GENERATION =====\n")
        return result.strip()
    except Exception as exc:
        print(f"[Generation error] {exc}")
        return ""


def chat_with_ai(user_message):
    """
    Main creative chat entry point. In correction mode, calls the
    spelling-only path. In creative mode, runs the generate-review-retry loop.
    """
    global WAITING_FOR_CORRECTION_TEXT, CURRENT_MODE

    message = user_message.strip()

    # --- Correction-mode handling ---
    if is_correction_request(message):
        match = re.search(r"\b(?:TEXT|SOURCE)\s*:\s*(.+)", message, re.IGNORECASE | re.DOTALL)
        if match:
            original_text = match.group(1).strip()
            if original_text.casefold().startswith("paste "):
                return "That looks like placeholder text. Send your actual dialogue after TEXT:."
            return correct_spelling_only(original_text)
        WAITING_FOR_CORRECTION_TEXT = True
        return (
            "Correction mode ready. Send your original dialogue in your "
            "next message. You can write TEXT: before it."
        )

    if WAITING_FOR_CORRECTION_TEXT:
        WAITING_FOR_CORRECTION_TEXT = False
        original_text = re.sub(
            r"^TEXT\s*:\s*", "", message, count=1, flags=re.IGNORECASE
        ).strip()
        if not original_text:
            return "No text received. Please send your original dialogue."
        return correct_spelling_only(original_text)

    if CURRENT_MODE == "correction":
        original_text = re.sub(
            r"^(?:TEXT|SOURCE)\s*:\s*", "", message, count=1, flags=re.IGNORECASE
        ).strip()
        if not original_text:
            return "No text received. Please send your original dialogue."
        return correct_spelling_only(original_text)

    # --- Creative-mode handling ---
    memory_context = build_context(message)
    creative_examples = retrieve_creative_context(message)

    base_prompt = f"""
{memory_context}

{creative_examples}

MANISH'S REQUEST:
{message}

ROLE:
You are Virtual Manish, a creative collaborator who understands
natural Telugu dialogue, character behaviour and screenplay structure.

STYLE:
- Use natural, imperfect Hyderabad Telugu and Telugu-English mixing.
- Do not translate English dialogue mechanically into Telugu.
- Give each character a distinct voice, motivation and worldview.
- Let humour emerge from behaviour, frustration and bad decisions.
- Avoid generic punchlines, artificial witty dialogue and over-explaining.
- Characters should not explain feelings they would naturally hide.

ORIGINALITY:
- Create events specifically for this premise.
- Do not copy plots, names, dialogue or event sequences from examples.
- Do not repeat a familiar scene structure with different props.
- Actions must have consequences.
- The ending must follow from the characters' choices.

Follow every explicit instruction in the user's request.
Return only the requested scene.
"""

    first_draft = _generate(base_prompt, 1)
    if not first_draft:
        return "I couldn't generate a response this time. Please try again."

    if not _needs_creative_review(message):
        return first_draft

    history = _load_history()
    candidate = first_draft
    failures_seen = []
    last_review = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        print(f"\n===== VALIDATION OF CANDIDATE {attempt} =====")

        review = _review_scene(candidate, message, history)
        last_review = review

        problems = list(review["violations"])
        history_problem = _compare_history(candidate, review["beat_summary"], history)
        if history_problem:
            problems.append(history_problem)
        problems = list(dict.fromkeys(problems))

        accepted = (
            review["verdict"] == "PASS"
            and not problems
            and bool(review["beat_summary"])
        )

        print("Verdict:", review["verdict"])
        print("Story beats:", review["beat_summary"])
        print("Problems:", problems)

        if accepted:
            history.append({
                "scene": candidate,
                "beat_summary": review["beat_summary"],
            })
            _save_history(history)
            print(f"[Creative review] Candidate {attempt} accepted.")
            return candidate

        failures_seen.extend(problems)
        print("[Creative review] Candidate rejected.")

        if attempt == MAX_GENERATION_ATTEMPTS:
            break

        # Use the reviewer's own retry_direction, or fall back to a
        # generic direction if the reviewer did not supply one.
        retry_direction = review.get("retry_direction") or (
            "Restructure the scene so the central conflict drives every beat. "
            "Change which character acts first, change the inciting pressure, "
            "and let the ending follow from the new causal chain."
        )

        retry_prompt = f"""
Create a genuinely new scene from scratch.

ORIGINAL USER REQUEST:
{message}

ORIGINAL CONTEXT AND CREATIVE STYLE:
{base_prompt}

NEW STORY DIRECTION (from the reviewer):
{retry_direction}

WHY THE PREVIOUS CANDIDATE FAILED:
{json.dumps(list(dict.fromkeys(failures_seen)), ensure_ascii=False)}

PREVIOUS CANDIDATE'S ACTUAL STORY BEATS:
{json.dumps(review["beat_summary"], ensure_ascii=False)}

REGENERATION RULES:
- Preserve the original characters, setting, premise and explicit constraints.
- Do not copy the rejected scene's sequence of events.
- Do not simply change the prop, argument, or dialogue.
- Follow the new story direction above.
- Do not use a generic final teasing line as the resolution.
- Use believable Hyderabad Telugu and character-specific behaviour.

Return only the complete scene, with dialogue and minimal action descriptions.
Do not explain your process.
"""

        candidate = _generate(retry_prompt, attempt + 1)
        if not candidate:
            break

    return (
        "I couldn't produce a scene that passed the originality checks "
        f"after {MAX_GENERATION_ATTEMPTS} attempts. No rejected candidate was approved. "
        "Try again with a fresh generation."
    )


# ============================================================
# 12. COMMANDS
# ============================================================

def show_memory():
    profile = load_json(PROFILE_FILE, {"preferences": []})
    preferences = profile.get("preferences", [])
    consolidated = consolidate_preferences(preferences)

    print("\n========== SAVED PREFERENCES ==========")
    if not preferences:
        print("No preferences saved yet.")
    for item in preferences:
        print(f"- {get_text(item)}")

    print("\n========== CONSOLIDATED VIEW (what the model sees) ==========")
    if not consolidated:
        print("No consolidated preferences.")
    for text in consolidated:
        print(f"- {text}")

    evidence = profile.get("preference_evidence", [])
    print("\n========== SIMILAR-PREFERENCE EVIDENCE ==========")
    if not evidence:
        print("No semantic duplicates recorded.")
    for item in evidence[-10:]:
        print(f"- Original: {item.get('existing_preference', '')}")
        print(f"  New wording: {item.get('new_wording', '')}")
        if item.get("new_information"):
            print(f"  Additional detail: {item['new_information']}")

    with get_connection() as connection:
        lessons = connection.execute("""
            SELECT id, category, lesson, confidence
            FROM lessons
            WHERE status = 'active'
            ORDER BY id
        """).fetchall()

    print("\n========== LEARNED LESSONS ==========")
    if not lessons:
        print("No active lessons yet.")
    for lesson_id, category, lesson, confidence in lessons:
        print(f"#{lesson_id} [{category}] Confidence: {confidence:.2f}\n  {lesson}")
    print()


def show_pending():
    with get_connection() as connection:
        lessons = connection.execute("""
            SELECT id, category, lesson, confidence
            FROM lessons
            WHERE status = 'pending'
            ORDER BY id
        """).fetchall()

    print(f"\n========== PENDING LESSONS ({len(lessons)}/{MAX_PENDING_LESSONS}) ==========")
    if not lessons:
        print("No lessons waiting for approval.")
    for lesson_id, category, lesson, confidence in lessons:
        print(f"#{lesson_id} [{category}] Confidence: {confidence:.2f}\n  {lesson}")
    print()


def update_lesson_status(lesson_id, status):
    if status not in {"active", "rejected"}:
        print("Invalid lesson status.")
        return
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE lessons SET status = ? WHERE id = ? AND status = 'pending'",
            (status, lesson_id),
        )
        if cursor.rowcount:
            print(f"Lesson #{lesson_id} marked {status}.")
        else:
            print("Pending lesson not found.")


def show_help():
    print("""
Commands:

/mode correction
    Enable spelling-only correction mode.

/mode creative
    Return to creative collaboration mode.

/remember YOUR PREFERENCE
    Save a creative preference.

/film Film Title | Your analysis
    Save a film analysis.

/feedback AI response | Your correction
    Save a correction example.

/study filename.txt
    Study a scene saved in creative_examples.

/memory
    View preferences, consolidated view, and learned lessons.

/pending
    View lessons waiting for approval.

/approve ID
    Approve a pending lesson.

/reject ID
    Reject a pending lesson.

/multiline
    Paste a multi-line prompt. End with /end on its own line.

/help
    Show commands.

/exit
    Exit Virtual Manish.

Anything else starts a normal conversation.
""")


def handle_command(user_message):
    global CURRENT_MODE, WAITING_FOR_CORRECTION_TEXT

    message = user_message.strip()
    lowered = message.casefold()

    if lowered == "/help":
        show_help()
        return True

    if lowered == "/mode correction":
        CURRENT_MODE = "correction"
        WAITING_FOR_CORRECTION_TEXT = False
        print("Virtual Manish mode: CORRECTION")
        print("Send dialogue to correct spelling only.")
        return True

    if lowered == "/mode creative":
        CURRENT_MODE = "creative"
        WAITING_FOR_CORRECTION_TEXT = False
        print("Virtual Manish mode: CREATIVE")
        return True

    if lowered == "/memory":
        show_memory()
        return True

    if lowered == "/pending":
        show_pending()
        return True

    if lowered == "/study":
        print("Enter the filename of a scene inside creative_examples.")
        filename = input("Scene filename: ").strip()
        if filename:
            study_scene_file(filename)
        return True

    if lowered.startswith("/study "):
        study_scene_file(message[len("/study "):].strip())
        return True

    if message.startswith("/remember "):
        remember_preference(message[len("/remember "):].strip())
        return True

    if message.startswith("/film "):
        parts = message[len("/film "):].split("|", 1)
        if len(parts) != 2:
            print("Usage: /film Film Title | Your analysis")
        else:
            save_film_analysis(parts[0], parts[1])
        return True

    if message.startswith("/feedback "):
        parts = message[len("/feedback "):].split("|", 1)
        if len(parts) != 2:
            print("Usage: /feedback AI response | Your correction")
        else:
            save_feedback(parts[0], parts[1])
        return True

    match = re.fullmatch(r"/approve\s+(\d+)", message, re.IGNORECASE)
    if match:
        update_lesson_status(int(match.group(1)), "active")
        return True

    match = re.fullmatch(r"/reject\s+(\d+)", message, re.IGNORECASE)
    if match:
        update_lesson_status(int(match.group(1)), "rejected")
        return True

    return False


# ============================================================
# 13. MAIN LOOP
# ============================================================

def main():
    initialize_files()
    initialize_database()

    print("\n======================================")
    print("       VIRTUAL MANISH — CREATIVE DNA")
    print("======================================")
    print(f"Model: {MODEL}")
    print("Type /help to see commands.")
    print("Type /exit to quit.\n")

    while True:
        try:
            first_line = input("Manish: ").strip()

            if first_line.casefold() == "/multiline":
                print("Paste your complete prompt below.")
                print("When finished, type /end on a new line.")
                lines = []
                while True:
                    line = input()
                    if line.strip().casefold() == "/end":
                        break
                    lines.append(line)
                user_message = "\n".join(lines).strip()
            else:
                user_message = first_line

        except (KeyboardInterrupt, EOFError):
            print("\nVirtual Manish closed.")
            break

        if not user_message:
            continue

        if user_message.casefold() == "/exit":
            print("Virtual Manish: See you, Manish.")
            break

        if handle_command(user_message):
            continue

        try:
            save_conversation("user", user_message)
            ai_response = chat_with_ai(user_message)
            print(f"\nVirtual Manish: {ai_response}\n")
            save_conversation("assistant", ai_response)

        except Exception as error:
            print(f"\nSomething went wrong: {error}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            print("Your existing memory files have not been reset.\n")


if __name__ == "__main__":
    main()