
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from google import genai


# ---------- CONFIGURATION ----------

BASE_DIR = Path(__file__).parent

PROFILE_FILE = BASE_DIR / "creative_profile.json"
FILM_FILE = BASE_DIR / "film_library.json"
FEEDBACK_FILE = BASE_DIR / "feedback.json"
DB_FILE = BASE_DIR / "virtual_manish.db"

MODEL = "gemini-3.5-flash-lite"
client = genai.Client()


SYSTEM_INSTRUCTIONS = """
You are Virtual Manish: Manish's personal creative intelligence
and writing partner.

Your goal is to understand how Manish thinks, creates, communicates,
judges ideas, and develops stories.

IMPORTANT:
- Understand before suggesting.
- Do not take creative control away from Manish.
- Do not rewrite an idea unless requested.
- Ask one useful question at a time when needed.
- Keep responses clear and concise unless detail is requested.
- Distinguish Manish's ideas from your own suggestions.
- Never claim to remember something unavailable in your context.

LEARN HIS CREATIVE REASONING:
- Notice why Manish likes or rejects an idea.
- Use his examples, corrections, and past decisions.
- Prefer demonstrated patterns over generic writing advice.
- Do not turn one temporary opinion into a permanent rule.
- Do not force comedy, emotion, or dramatic moments.

WRITING AND CINEMA:
Learn from character behavior, motivations, relationships,
situations, expressions, reactions, atmosphere, pacing, subtext,
screenplay formatting, and actual examples Manish provides.

TELUGU DIALOGUE:
Use natural, character-specific Telugu and Telugu-English speech.
Respect age, background, location, relationships, and situation.
Avoid literal translations, generic slang, forced punchlines,
and dialogue that explains every emotion.

COLLABORATION:
When Manish explains an idea, develop it without replacing
his thinking with generic suggestions.
When asked to write, follow his requested format and style.
When corrected, apply the lesson in relevant future conversations.

Be specific, honest, and imaginative.
"""


# ---------- TEXT NORMALIZATION ----------

def normalize_text(text):
    """Normalize text to detect exact and punctuation-only duplicates."""
    text = str(text).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def deduplicate_preferences(preferences):
    """Keep the first occurrence of each normalized preference."""
    unique = []
    seen = set()

    for item in preferences:
        note = item.get("note", "") if isinstance(item, dict) else str(item)
        key = normalize_text(note)

        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


# ---------- DATABASE ----------

def connect_db():
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with connect_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                lesson TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

    deduplicate_existing_lessons()


def deduplicate_existing_lessons():
    """Remove duplicate active/pending lessons, retaining one copy."""
    with connect_db() as db:
        rows = db.execute("""
            SELECT id, lesson, status
            FROM lessons
            WHERE status IN ('active', 'pending')
            ORDER BY id ASC
        """).fetchall()

        groups = {}

        for row in rows:
            key = normalize_text(row["lesson"])
            if key:
                groups.setdefault(key, []).append(dict(row))

        removed = 0

        for group in groups.values():
            if len(group) < 2:
                continue

            # Prefer retaining an active lesson over a pending one.
            active = [item for item in group if item["status"] == "active"]
            keep = active[0] if active else group[0]

            for item in group:
                if item["id"] != keep["id"]:
                    db.execute(
                        "DELETE FROM lessons WHERE id = ?",
                        (item["id"],)
                    )
                    removed += 1

    if removed:
        print(f"[Memory cleanup: removed {removed} duplicate lesson(s).]")


def save_message(role, message):
    with connect_db() as db:
        db.execute("""
            INSERT INTO conversations (role, message, created_at)
            VALUES (?, ?, ?)
        """, (
            role,
            message,
            datetime.now().isoformat(timespec="seconds")
        ))


def get_recent_conversation(limit=12):
    with connect_db() as db:
        rows = db.execute("""
            SELECT role, message
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return "\n".join(
        f"{row['role']}: {row['message']}"
        for row in reversed(rows)
    )


def get_lessons(status="active", limit=100):
    with connect_db() as db:
        rows = db.execute("""
            SELECT id, category, lesson, source, status,
                   confidence, created_at
            FROM lessons
            WHERE status = ?
            ORDER BY id DESC
            LIMIT ?
        """, (status, limit)).fetchall()

    return [dict(row) for row in reversed(rows)]


def save_lesson(category, lesson, source, status, confidence):
    new_key = normalize_text(lesson)

    if not new_key:
        return None

    with connect_db() as db:
        rows = db.execute("""
            SELECT id, lesson
            FROM lessons
            WHERE status IN ('active', 'pending')
        """).fetchall()

        for row in rows:
            if normalize_text(row["lesson"]) == new_key:
                return None

        cursor = db.execute("""
            INSERT INTO lessons
            (category, lesson, source, status, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            category,
            lesson,
            source,
            status,
            confidence,
            datetime.now().isoformat(timespec="seconds")
        ))

        return cursor.lastrowid


# ---------- JSON MEMORY ----------

def load_json(path, default):
    if not path.exists():
        save_json(path, default)
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"Could not read {path.name}. Check the file.")
        raise


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def initialize_files():
    profile = load_json(PROFILE_FILE, {"preferences": []})
    load_json(FILM_FILE, {"films": []})
    load_json(FEEDBACK_FILE, {"examples": []})

    preferences = profile.get("preferences", [])
    cleaned = deduplicate_preferences(preferences)

    if len(cleaned) != len(preferences):
        profile["preferences"] = cleaned
        save_json(PROFILE_FILE, profile)
        print(
            "[Memory cleanup: removed "
            f"{len(preferences) - len(cleaned)} duplicate preference(s).]"
        )


def remember_preference(note):
    note = note.strip()

    if not note:
        print("Enter a preference after /remember.")
        return

    profile = load_json(PROFILE_FILE, {"preferences": []})
    preferences = deduplicate_preferences(
        profile.get("preferences", [])
    )
    new_key = normalize_text(note)

    for item in preferences:
        existing = (
            item.get("note", "")
            if isinstance(item, dict)
            else str(item)
        )

        if normalize_text(existing) == new_key:
            profile["preferences"] = preferences
            save_json(PROFILE_FILE, profile)
            print("That preference is already saved. No duplicate added.")
            return

    preferences.append({
        "note": note,
        "created_at": datetime.now().isoformat(timespec="seconds")
    })

    profile["preferences"] = preferences
    save_json(PROFILE_FILE, profile)
    print("Saved to your Creative DNA.")


def save_film_analysis(title, notes):
    title = title.strip()
    notes = notes.strip()
    library = load_json(FILM_FILE, {"films": []})

    for film in library.get("films", []):
        same_title = normalize_text(film.get("title", "")) == normalize_text(title)
        same_notes = normalize_text(film.get("your_analysis", "")) == normalize_text(notes)

        if same_title and same_notes:
            print("That film analysis is already saved.")
            return

    library["films"].append({
        "title": title,
        "your_analysis": notes,
        "created_at": datetime.now().isoformat(timespec="seconds")
    })

    save_json(FILM_FILE, library)
    print(f"Saved your analysis of {title}.")


def save_feedback(previous_answer, correction):
    feedback = load_json(FEEDBACK_FILE, {"examples": []})

    for item in feedback.get("examples", []):
        same_answer = normalize_text(item.get("ai_response", "")) == normalize_text(previous_answer)
        same_correction = normalize_text(item.get("your_correction", "")) == normalize_text(correction)

        if same_answer and same_correction:
            print("That feedback is already saved.")
            return

    feedback["examples"].append({
        "ai_response": previous_answer,
        "your_correction": correction,
        "created_at": datetime.now().isoformat(timespec="seconds")
    })

    save_json(FEEDBACK_FILE, feedback)
    print("Saved your correction for future reference.")


# ---------- AUTOMATIC LEARNING ----------

def learn_from_exchange(user_message, ai_answer):
    learning_prompt = f"""
Analyze this exchange for a useful, reusable lesson about Manish.

USER MESSAGE:
{user_message}

AI RESPONSE:
{ai_answer}

Return ONLY one JSON object:
{{
  "useful_lesson": true,
  "category": "writing_style",
  "lesson": "A concise reusable lesson",
  "confidence": 0.85,
  "importance": "routine"
}}

Rules:
- useful_lesson must be true or false.
- Categories: preference, reasoning_pattern, writing_style,
  project_fact, correction.
- confidence must be between 0 and 1.
- importance must be routine or important.
- Only use evidence supported by Manish's words.
- Prefer explicit explanations and corrections over guesses.
- Do not infer personality traits from isolated messages.
- Do not treat temporary story details as permanent preferences.
- Mark changes to core preferences, conflicts, and uncertain
  long-term conclusions as important.
- Never store passwords, API keys, or other secrets.
- Keep the lesson short and specific.
- Do not create a lesson just to restate a generic writing principle.
"""

    try:
        response = client.interactions.create(
            model=MODEL,
            input=learning_prompt,
            system_instruction=(
                "You are a careful personal-memory analyst. "
                "Return valid JSON only. Extract evidence-supported "
                "lessons and do not invent personal facts."
            )
        )

        raw = response.output_text.strip()
        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1:
            return

        result = json.loads(raw[start:end + 1])

        if not result.get("useful_lesson"):
            return

        lesson = str(result.get("lesson", "")).strip()
        category = str(result.get("category", "preference")).strip()
        confidence = float(result.get("confidence", 0))
        importance = result.get("importance", "important")

        allowed_categories = {
            "preference", "reasoning_pattern", "writing_style",
            "project_fact", "correction"
        }

        if not lesson or not 0 <= confidence <= 1 or confidence < 0.70:
            return

        if category not in allowed_categories:
            category = "preference"

        status = "active" if importance == "routine" else "pending"

        lesson_id = save_lesson(
            category, lesson, user_message, status, confidence
        )

        if lesson_id is None:
            return

        if status == "active":
            print(f"[Learned automatically: {category}]")
        else:
            print(
                f"[Lesson #{lesson_id} needs approval. "
                "Use /pending to review.]"
            )

    except Exception as error:
        print(f"[Learning skipped: {error}]")


# ---------- MEMORY INSPECTION ----------

def show_pending():
    lessons = get_lessons("pending", limit=100)

    if not lessons:
        print("No lessons are waiting for approval.")
        return

    for item in lessons:
        print(
            f"\n#{item['id']} [{item['category']}] "
            f"Confidence: {item['confidence']:.2f}"
        )
        print(item["lesson"])
        print(f"Approve: /approve {item['id']}")
        print(f"Reject:  /reject {item['id']}")


def show_memory():
    profile = load_json(PROFILE_FILE, {"preferences": []})
    library = load_json(FILM_FILE, {"films": []})
    feedback = load_json(FEEDBACK_FILE, {"examples": []})

    preferences = deduplicate_preferences(
        profile.get("preferences", [])
    )

    if len(preferences) != len(profile.get("preferences", [])):
        profile["preferences"] = preferences
        save_json(PROFILE_FILE, profile)

    print("\n========== SAVED PREFERENCES ==========")
    if preferences:
        for item in preferences:
            note = item.get("note", "") if isinstance(item, dict) else item
            print(f"- {note}")
    else:
        print("No saved preferences.")

    print("\n========== LEARNED LESSONS ==========")
    lessons = get_lessons("active")

    if lessons:
        for item in lessons:
            print(
                f"\n#{item['id']} [{item['category']}] "
                f"Confidence: {item['confidence']:.2f}"
            )
            print(item["lesson"])
    else:
        print("No active lessons.")

    print("\n========== FILM LIBRARY ==========")
    films = library.get("films", [])
    if films:
        for film in films:
            print(
                f"- {film.get('title', 'Untitled')}: "
                f"{film.get('your_analysis', '')}"
            )
    else:
        print("No film analyses saved.")

    print("\n========== SAVED FEEDBACK ==========")
    examples = feedback.get("examples", [])
    if examples:
        for item in examples:
            print(f"- Correction: {item.get('your_correction', '')}")
    else:
        print("No corrections saved.")

    print("\n========== PENDING APPROVAL ==========")
    show_pending()


def update_lesson_status(lesson_id, status):
    with connect_db() as db:
        cursor = db.execute("""
            UPDATE lessons
            SET status = ?
            WHERE id = ? AND status = 'pending'
        """, (status, lesson_id))

    if cursor.rowcount:
        print(f"Lesson #{lesson_id}: {status}.")
    else:
        print("Pending lesson not found. Check /pending.")


# ---------- BUILD CONTEXT ----------

def build_context():
    profile = load_json(PROFILE_FILE, {"preferences": []})
    library = load_json(FILM_FILE, {"films": []})
    feedback = load_json(FEEDBACK_FILE, {"examples": []})
    lessons = get_lessons("active")

    preferences = deduplicate_preferences(
        profile.get("preferences", [])
    )

    return f"""
MANISH'S SAVED PREFERENCES:
{json.dumps(preferences[-30:], ensure_ascii=False, indent=2)}

FILM AND SCENE ANALYSES:
{json.dumps(library.get('films', [])[-10:],
             ensure_ascii=False, indent=2)}

PAST CORRECTIONS:
{json.dumps(feedback.get('examples', [])[-10:],
             ensure_ascii=False, indent=2)}

LEARNED PATTERNS:
{json.dumps(lessons, ensure_ascii=False, indent=2)}

Use memories only when relevant.
Explicit preferences and corrections take priority.
Do not treat uncertain inferences as established facts.
"""


# ---------- CHAT ----------

def ask_ai(question):
    prompt = f"""
{build_context()}

RECENT CONVERSATION:
{get_recent_conversation()}

MANISH'S LATEST MESSAGE:
{question}
"""

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
        system_instruction=SYSTEM_INSTRUCTIONS
    )

    return response.output_text


# ---------- MAIN ----------

def main():
    initialize_files()
    initialize_database()

    last_answer = ""

    print("\n" + "=" * 48)
    print("             VIRTUAL MANISH")
    print("=" * 48)
    print("Chat naturally. Your conversation is saved locally.")
    print("\nCommands:")
    print("  /remember your preference")
    print("  /film Film Title | your analysis")
    print("  /feedback your correction")
    print("  /memory")
    print("  /pending")
    print("  /approve ID")
    print("  /reject ID")
    print("  /exit\n")

    while True:
        try:
            question = input("Manish: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nYour conversations and memories are saved.")
            break

        if not question:
            continue

        command = question.lower()

        if command == "/exit":
            print("Your conversations and memories are saved.")
            break

        if command == "/memory":
            show_memory()
            continue

        if command == "/pending":
            show_pending()
            continue

        if command.startswith("/approve ") or command.startswith("/reject "):
            parts = question.split(maxsplit=1)

            try:
                lesson_id = int(parts[1])
                status = "active" if command.startswith("/approve ") else "rejected"
                update_lesson_status(lesson_id, status)
            except (ValueError, IndexError):
                print("Use /approve ID or /reject ID.")
            continue

        if command.startswith("/remember "):
            remember_preference(question[len("/remember "):].strip())
            continue

        if command.startswith("/film "):
            details = question[len("/film "):].split("|", 1)

            if len(details) != 2 or not all(part.strip() for part in details):
                print("Use: /film Film Title | your analysis")
                continue

            save_film_analysis(details[0], details[1])
            continue

        if command.startswith("/feedback "):
            correction = question[len("/feedback "):].strip()

            if not last_answer:
                print("Ask the AI something first, then give feedback.")
                continue

            if not correction:
                print("Tell me what was wrong and what you prefer.")
                continue

            save_feedback(last_answer, correction)
            '''learn_from_exchange(
                "My correction to your previous answer: " + correction,
                last_answer
            )'''
            continue

        try:
            answer = ask_ai(question)
            print(f"\nVirtual Manish: {answer}\n")

            save_message("Manish", question)
            save_message("Virtual Manish", answer)
            last_answer = answer

            #learn_from_exchange(question, answer)

        except Exception as error:
            print(f"\nAI request failed: {error}")
            print("Check your API key, model access, and internet connection.\n")


if __name__ == "__main__":
    main()