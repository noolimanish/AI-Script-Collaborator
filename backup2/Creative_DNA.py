
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from google import genai
from creative_memory import search_creative_examples


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROFILE_FILE = BASE_DIR / "creative_profile.json"
FILM_FILE = BASE_DIR / "film_library.json"
FEEDBACK_FILE = BASE_DIR / "feedback.json"
DATABASE_FILE = BASE_DIR / "virtual_manish.db"

MODEL = "gemini-3.5-flash-lite"

client = genai.Client()


SYSTEM_INSTRUCTIONS = """
You are Virtual Manish, Manish's personal creative collaborator.

YOUR MAIN PURPOSE
Understand how Manish thinks, writes, creates characters, develops scenes,
evaluates dialogue, and makes creative decisions.

MANISH'S LANGUAGE AND DIALOGUE STYLE
- Preserve his natural Hyderabad-style Telugu and Telugu-English mixing.
- Spelling may be corrected when useful.
- Do not replace his slang with formal Telugu.
- Preserve his sentence rhythm, word connections, expressions, and flow.
- Do not translate English into Telugu mechanically.
- Do not make his dialogue sound like generic movie dialogue.
- Do not invent slang and claim it is Manish's style.
- Treat his actual writing samples as evidence of his voice.

USING CREATIVE EXAMPLES
- Use retrieved passages as references, not templates to copy.
- Learn from dialogue, character behavior, atmosphere, scene construction,
  emotional choices, and the way characters interact.
- Do not copy names, plot details, or dialogue into unrelated stories.
- An English screenplay may teach storytelling and visual description.
  It does not automatically define Manish's Telugu dialogue style.
- If the retrieved example is not relevant, do not force it into the answer.
- Distinguish Manish's original writing from AI-generated suggestions.

CREATIVE COLLABORATION
- Develop Manish's ideas rather than replacing them with generic ideas.
- Avoid generic rom-com characters and forced punchlines.
- Prefer comedy rooted in personality, flaws, relationships, awkwardness,
  reactions, and bad decisions when appropriate.
- Pay attention to what characters do, not just what they explain.
- If Manish is exploring an idea, collaborate before writing the whole scene.
- Ask one clear question at a time when clarification is needed.
- Learn from Manish's examples and corrections.
- Do not claim to remember information that is not in your context.

Respond naturally and directly.
"""


# ============================================================
# WRITING MODES
# ============================================================

CURRENT_MODE = "creative"

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
    """Recognize explicit spelling-only requests."""
    text = message.casefold()
    return any(phrase in text for phrase in CORRECTION_PHRASES)


def find_source_text(message):
    """
    Accept text explicitly supplied after TEXT: or SOURCE:.
    Otherwise, find a specifically named example file.
    Never guess which scene the user means.
    """
    match = re.search(
        r"(?:TEXT|SOURCE)\s*:\s*(.+)",
        message,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    examples_dir = BASE_DIR / "creative_examples"

    known_files = {
        "audition": "audition_scene.txt",
        "ashrayam": "ashrayam_short_film_screenplay.txt",
        "aaravi": "THE BOOK OF AARAVI_UPDATED.txt",
    }

    message_lower = message.casefold()

    for keyword, filename in known_files.items():
        if keyword in message_lower:
            source_path = examples_dir / filename

            if source_path.is_file():
                return source_path.read_text(encoding="utf-8")

    return None


def correct_spelling_only(original_text):
    """
    Send only the identified original text to the model.
    Do not include creative memories or unrelated conversation.
    """
    prompt = f"""
Correct spelling mistakes in the source text below.

STRICT RULES:
- Correct spelling only when the intended word is clear.
- Preserve every original dialogue line and its order.
- Preserve slang, colloquial Telugu, Telugu-English mixing,
  punctuation, expressions, and sentence structure.
- Do not rewrite, polish, translate, formalize, or improve sentences.
- Do not add or remove dialogue, words, jokes, or descriptions.
- Do not change a word just because another word sounds better.
- If a word is ambiguous, leave it unchanged.
- Do not add explanations, headings, or commentary.
- Return only the corrected text.

SOURCE TEXT:
<source>
{original_text}
</source>
"""

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
        system_instruction=(
            "You are a conservative proofreader. "
            "Perform spelling correction only. "
            "Never creatively rewrite source material."
        ),
    )

    return response_text(response)


# ============================================================
# GENERAL HELPERS
# ============================================================

def now():
    return datetime.now().isoformat(timespec="seconds")


def normalize_text(text):
    text = str(text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def get_text(value):
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return str(
            value.get("text")
            or value.get("preference")
            or value.get("note")
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

    except json.JSONDecodeError:
        raise RuntimeError(
            f"{path.name} contains invalid JSON. "
            "Restore it from your backup before continuing."
        )


def save_json(path, data):
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    temporary_path.replace(path)


def response_text(response):
    output = getattr(response, "output_text", None)

    if output:
        return output.strip()

    raise RuntimeError(
        "Gemini returned no output_text. "
        "Check the response and installed google-genai version."
    )


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("Gemini did not return valid JSON.")

        return json.loads(match.group(0))


# ============================================================
# FILE INITIALIZATION
# ============================================================

def initialize_files():
    defaults = [
        (PROFILE_FILE, {
            "preferences": [],
            "preference_evidence": [],
        }),
        (FILM_FILE, {"films": []}),
        (FEEDBACK_FILE, {"examples": []}),
    ]

    for path, default in defaults:
        if not path.exists():
            save_json(path, default)

    profile = load_json(PROFILE_FILE, {"preferences": []})
    profile.setdefault("preferences", [])
    profile.setdefault("preference_evidence", [])
    save_json(PROFILE_FILE, profile)

    films = load_json(FILM_FILE, {"films": []})
    films.setdefault("films", [])
    save_json(FILM_FILE, films)

    feedback = load_json(FEEDBACK_FILE, {"examples": []})
    feedback.setdefault("examples", [])
    save_json(FEEDBACK_FILE, feedback)


# ============================================================
# SQLITE DATABASE
# ============================================================

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
        rows = connection.execute("""
            SELECT id, lesson, status
            FROM lessons
            ORDER BY id
        """).fetchall()

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
                connection.execute(
                    "DELETE FROM lessons WHERE id = ?",
                    (original_id,),
                )
                seen[key] = (lesson_id, status)

            else:
                connection.execute(
                    "DELETE FROM lessons WHERE id = ?",
                    (lesson_id,),
                )


def save_conversation(role, message):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO conversations (role, message, created_at)
            VALUES (?, ?, ?)
        """, (role, message, now()))


def save_lesson(category, lesson, source, status, confidence):
    key = normalize_text(lesson)

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, lesson FROM lessons
            WHERE status IN ('active', 'pending')
        """).fetchall()

        for lesson_id, existing in rows:
            if normalize_text(existing) == key:
                return None

        cursor = connection.execute("""
            INSERT INTO lessons
                (category, lesson, source, status, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            category,
            lesson,
            source,
            status,
            confidence,
            now(),
        ))

        return cursor.lastrowid


# ============================================================
# SEMANTIC DUPLICATE DETECTION
# ============================================================

CONCEPT_GROUPS = {
    "comedy": {
        "comedy", "funny", "humor", "humour", "laugh",
        "laughs", "comic", "comedic",
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
        existing_words = set(
            re.findall(r"[a-z]+", existing_text.casefold())
        )
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

- Similar topic does not automatically mean duplicate.
- If the new preference adds a meaningful rule, preserve it separately.
- Treat genuine paraphrases as possible duplicates.
- If uncertain, return false.
- Do not rewrite, merge, or delete anything.

Return JSON only:
{{
  "same_core_preference": true,
  "reason": "Brief explanation",
  "new_information": "Additional information, if any"
}}
"""

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
        system_instruction=(
            "Compare creative memories conservatively. Return valid JSON only."
        ),
    )

    result = extract_json(response_text(response))

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

    profile = load_json(PROFILE_FILE, {"preferences": []})
    profile.setdefault("preferences", [])
    profile.setdefault("preference_evidence", [])

    preferences = profile["preferences"]
    existing_texts = [
        get_text(item) for item in preferences
    ]

    normalized_new = normalize_text(preference)

    exact_match = next(
        (
            text for text in existing_texts
            if normalize_text(text) == normalized_new
        ),
        None,
    )

    if exact_match:
        print("This exact preference is already saved.")
        print("Checking other saved preferences for semantic duplicates...")

        existing_texts = [
            text for text in existing_texts
            if normalize_text(text) != normalized_new
        ]

    candidates = local_similarity_candidates(
        preference,
        existing_texts,
    )

    try:
        for candidate in candidates:
            comparison = compare_memory_meaning(
                preference,
                candidate["text"],
            )

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
                    print(
                        "Additional detail:",
                        comparison["new_information"],
                    )

                return

    except Exception as error:
        print(
            "Semantic comparison failed. "
            "No existing preference was deleted."
        )
        print(f"Details: {error}")

        if exact_match:
            print("The exact preference already exists; no change made.")
            return

    if exact_match:
        print("No semantic duplicate was confirmed.")
        print("The exact preference is already saved.")
        return

    if preferences and isinstance(preferences[0], dict):
        preferences.append({
            "text": preference,
            "created_at": now(),
            "source": source,
        })
    else:
        preferences.append(preference)

    save_json(PROFILE_FILE, profile)
    print("Saved to your Creative DNA.")


# ============================================================
# FILM ANALYSIS AND FEEDBACK
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
            and normalize_text(film.get("analysis", ""))
            == normalize_text(analysis)
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
            normalize_text(example.get("ai_response", ""))
            == normalize_text(ai_response)
            and normalize_text(example.get("correction", ""))
            == normalize_text(correction)
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
# AUTOMATIC LEARNING
# ============================================================

def learn_from_exchange(user_message, ai_response):
    prompt = f"""
Identify one durable lesson about Manish's creative preferences or reasoning
from this exchange.

USER:
{user_message}

VIRTUAL MANISH:
{ai_response}

Rules:
- Do not invent a preference.
- Do not turn a one-time request into a permanent rule.
- If there is no useful durable lesson, return "useful": false.
- Keep lessons specific and concise.
- Confidence must be between 0 and 1.
- Importance must be "routine" or "core".
- Core lessons must remain pending for approval.

Return JSON only:
{{
  "useful": true,
  "category": "writing_style",
  "lesson": "Specific lesson",
  "confidence": 0.85,
  "importance": "routine"
}}
"""

    try:
        response = client.interactions.create(
            model=MODEL,
            input=prompt,
            system_instruction=(
                "Extract evidence-based durable lessons. Return valid JSON only."
            ),
        )

        result = extract_json(response_text(response))

        if not result.get("useful", False):
            return

        lesson = str(result.get("lesson", "")).strip()
        category = str(result.get("category", "general")).strip()
        confidence = float(result.get("confidence", 0.0))
        importance = str(result.get("importance", "core")).casefold()

        if not lesson:
            return

        confidence = max(0.0, min(1.0, confidence))
        status = (
            "active"
            if confidence >= 0.70 and importance == "routine"
            else "pending"
        )

        lesson_id = save_lesson(
            category=category,
            lesson=lesson,
            source="automatic_learning",
            status=status,
            confidence=confidence,
        )

        if lesson_id is not None:
            print(f"[Learning] Lesson #{lesson_id} saved as {status}.")

    except Exception as error:
        print(f"[Learning] Skipped this exchange: {error}")


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context():
    profile = load_json(PROFILE_FILE, {"preferences": []})
    film_library = load_json(FILM_FILE, {"films": []})
    feedback = load_json(FEEDBACK_FILE, {"examples": []})

    preferences = get_text_list(
        profile.get("preferences", [])
    )[-30:]

    films = film_library.get("films", [])[-10:]
    examples = feedback.get("examples", [])[-10:]

    with get_connection() as connection:
        lessons = connection.execute("""
            SELECT category, lesson, confidence
            FROM lessons
            WHERE status = 'active'
            ORDER BY id DESC
        """).fetchall()

        conversations = connection.execute("""
            SELECT role, message
            FROM conversations
            ORDER BY id DESC
            LIMIT 12
        """).fetchall()

    conversations.reverse()

    return f"""
MANISH'S SAVED CREATIVE PREFERENCES:
{json.dumps(preferences, ensure_ascii=False, indent=2)}

FILM ANALYSES:
{json.dumps(films, ensure_ascii=False, indent=2)}

FEEDBACK EXAMPLES:
{json.dumps(examples, ensure_ascii=False, indent=2)}

ACTIVE LEARNED LESSONS:
{json.dumps(lessons, ensure_ascii=False, indent=2)}

RECENT CONVERSATION:
{json.dumps(conversations, ensure_ascii=False, indent=2)}
"""


def get_text_list(items):
    return [
        get_text(item)
        for item in items
        if get_text(item).strip()
    ]


def retrieve_creative_context(user_message):
    """
    Search Manish's writing and return relevant passages for Gemini.
    The source files remain unchanged.
    """
    try:
        results = search_creative_examples(user_message, limit=3)

    except Exception as error:
        print(f"[Creative memory] Search unavailable: {error}")
        return "No creative examples were retrieved."

    if not results:
        return (
            "No directly matching passages were found in the creative library. "
            "Do not pretend a matching example exists."
        )

    passages = []

    for result in results:
        passages.append(
            f"""
SOURCE FILE: {result['filename']}
PASSAGE: {result['passage_number']}
MATCHED KEYWORDS: {', '.join(result['matched_words'])}

{result['text']}
"""
        )

    return "\n\n--- RETRIEVED CREATIVE EXAMPLES ---\n".join(passages)


# ============================================================
# MEMORY COMMANDS
# ============================================================

def show_memory():
    profile = load_json(PROFILE_FILE, {"preferences": []})
    preferences = get_text_list(profile.get("preferences", []))

    print("\n========== SAVED PREFERENCES ==========")

    if not preferences:
        print("No preferences saved yet.")

    for preference in preferences:
        print(f"- {preference}")

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
        print(
            f"#{lesson_id} [{category}] Confidence: {confidence:.2f}\n"
            f"  {lesson}"
        )

    print()


def show_pending():
    with get_connection() as connection:
        lessons = connection.execute("""
            SELECT id, category, lesson, confidence
            FROM lessons
            WHERE status = 'pending'
            ORDER BY id
        """).fetchall()

    print("\n========== PENDING LESSONS ==========")

    if not lessons:
        print("No lessons waiting for approval.")

    for lesson_id, category, lesson, confidence in lessons:
        print(
            f"#{lesson_id} [{category}] Confidence: {confidence:.2f}\n"
            f"  {lesson}"
        )

    print()


def update_lesson_status(lesson_id, status):
    if status not in {"active", "rejected"}:
        print("Invalid lesson status.")
        return

    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE lessons
            SET status = ?
            WHERE id = ? AND status = 'pending'
        """, (status, lesson_id))

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
    Enable creative collaboration mode.
    
/remember YOUR PREFERENCE
    Save a creative preference.

/film Film Title | Your analysis
    Save a film analysis.

/feedback AI response | Your correction
    Save a correction example.

/memory
    View preferences and learned lessons.

/pending
    View lessons waiting for approval.

/approve ID
    Approve a pending lesson.

/reject ID
    Reject a pending lesson.

/help
    Show commands.

/exit
    Exit Virtual Manish.

Anything else starts a normal conversation.
""")


# ============================================================
# CHAT WITH CREATIVE RETRIEVAL
# ============================================================


def chat_with_ai(user_message):
    global CURRENT_MODE

    # An explicit spelling-only request takes priority.
    if is_correction_request(user_message):
        CURRENT_MODE = "correction"

    # Correction mode must not receive creative memories or examples.
    if CURRENT_MODE == "correction":
        original_text = find_source_text(user_message)

        if not original_text:
            return (
                "I couldn't identify the exact source text safely. "
                "Please send the original text using:\n\n"
                "TEXT: paste your original dialogue here\n\n"
                "I'll correct clear spelling mistakes only."
            )

        return correct_spelling_only(original_text)

    # Normal Creative Mode: retain the existing memory features.
    memory_context = build_context()
    creative_examples = retrieve_creative_context(user_message)

    prompt = f"""
{memory_context}

{creative_examples}

CURRENT MODE: CREATIVE

MANISH'S CURRENT MESSAGE:
{user_message}

HOW TO USE THE RETRIEVED EXAMPLES:
- Determine whether they are relevant to the request.
- Learn from Manish's creative choices without copying unrelated scenes.
- Preserve his natural Telugu slang and Telugu-English mixing.
- Do not assume a request to develop a scene means changing his
  original dialogue without making the changes clear.
- Distinguish original writing from newly suggested material.
- If Manish explicitly requests spelling-only correction, do not
  creatively rewrite the text.
- If no example is relevant, use the current request and saved
  preferences instead.

Respond as Manish's creative collaborator.
"""

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
        system_instruction=SYSTEM_INSTRUCTIONS,
    )

    return response_text(response)


# ============================================================
# COMMAND HANDLING
# ============================================================

def handle_command(user_message):
    if user_message == "/help":
        show_help()
        return True

    
    global CURRENT_MODE

    if user_message.casefold() in {
        "/mode correction",
        "/mode creative",
    }:
        CURRENT_MODE = user_message.split()[-1].casefold()
        print(f"Virtual Manish mode: {CURRENT_MODE}")
        return True
    
    if user_message == "/memory":
        show_memory()
        return True

    if user_message == "/pending":
        show_pending()
        return True

    if user_message.startswith("/remember "):
        remember_preference(
            user_message[len("/remember "):].strip()
        )
        return True

    if user_message.startswith("/film "):
        parts = user_message[len("/film "):].split("|", 1)

        if len(parts) != 2:
            print("Usage: /film Film Title | Your analysis")
        else:
            save_film_analysis(parts[0], parts[1])

        return True

    if user_message.startswith("/feedback "):
        parts = user_message[len("/feedback "):].split("|", 1)

        if len(parts) != 2:
            print("Usage: /feedback AI response | Your correction")
        else:
            save_feedback(parts[0], parts[1])

        return True

    match = re.fullmatch(r"/approve\s+(\d+)", user_message)

    if match:
        update_lesson_status(int(match.group(1)), "active")
        return True

    match = re.fullmatch(r"/reject\s+(\d+)", user_message)

    if match:
        update_lesson_status(int(match.group(1)), "rejected")
        return True

    return False


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    initialize_files()
    initialize_database()

    print("\n======================================")
    print("       VIRTUAL MANISH — CREATIVE DNA")
    print("======================================")
    print("Your personal creative collaborator.")
    print("Type /help to see commands.")
    print("Type /exit to quit.\n")

    while True:
        try:
            user_message = input("Manish: ").strip()

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

            learn_from_exchange(user_message, ai_response)

        except Exception as error:
            print(f"\nSomething went wrong: {error}")
            print("Your existing memory files have not been intentionally reset.\n")


if __name__ == "__main__":
    main()