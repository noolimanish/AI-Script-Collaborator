"""
Extracts STRUCTURAL summaries from creative_references/liked/ screenplays.

These are films Manish admires for STRUCTURE — not for voice.
Used only to inform plot logic, mystery-hiding, character connection, spine.
NEVER injected as raw text — only as structured summaries.

Output: liked_structure.json
Incremental: only re-processes files whose SHA-256 changed.
"""
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from google import genai

BASE_DIR = Path(__file__).resolve().parent
LIKED_DIR = BASE_DIR / "creative_references" / "liked"
OUTPUT_FILE = BASE_DIR / "liked_structure.json"
CACHE_FILE = BASE_DIR / "liked_structure_cache.json"

MODEL = os.getenv("VM_MODEL", "gemini-3.8-flash")
client = genai.Client()


def get_file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_script_text(text, max_words=4500):
    """Sample beginning, middle, end for long screenplays."""
    words = text.split()
    if len(words) <= max_words:
        return text
    third = max_words // 3
    mid = len(words) // 2
    return (
        " ".join(words[:third])
        + "\n\n[... middle omitted ...]\n\n"
        + " ".join(words[mid - third // 2 : mid + third // 2])
        + "\n\n[... late section omitted ...]\n\n"
        + " ".join(words[-third:])
    )


def extract_structure(text, filename):
    sampled = _sample_script_text(text)
    prompt = f"""
Analyze this screenplay ({filename}) for STRUCTURE ONLY.

Extract structural insights — the kind a screenwriter studies when they admire a film's architecture.
DO NOT summarize plot in detail. DO NOT describe dialogue. DO NOT describe prose style.
Focus purely on: how the story is BUILT.

Return ONLY valid JSON:
{{
  "acts": {{
    "act_one": "What act 1 establishes — setup, premise, inciting incident, the world.",
    "act_two": "How act 2 escalates — stakes, complications, midpoint, low point.",
    "act_three": "How act 3 resolves — climax, resolution, final image."
  }},
  "mystery_hiding": "How does the film hide its mystery / reveal its truth? What is withheld, when, and how is the reveal engineered?",
  "character_connections": "How do characters connect? What conflict structures, power dynamics, or relationships drive the story? How do arcs interlock?",
  "spine": "What is this story really about — the emotional/thematic core beneath the plot? (One sentence.)",
  "key_structural_decisions": [
    "e.g. 'Opens with the ending image, then rewinds'",
    "e.g. 'The antagonist is revealed only in act 3 via a two-shot with no dialogue'"
  ],
  "scene_rhythm": "Pacing and scene-length rhythm. E.g. 'short punchy scenes interleaved with 5-minute emotional set-pieces'."
}}

SCRIPT TEXT:
{sampled}
"""

    stream = client.interactions.create(
        model=MODEL,
        input=prompt,
        system_instruction="You are a structural screenplay analyst. Return valid JSON only.",
        store=False,
        stream=True,
    )
    raw = ""
    for event in stream:
        if hasattr(event, "delta") and hasattr(event.delta, "text") and event.delta.text:
            raw += event.delta.text

    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object returned.")
    return json.loads(match.group(0))


def main():
    print("Extracting STRUCTURE from creative_references/liked/ ...")

    if not LIKED_DIR.exists():
        print(f"Folder not found: {LIKED_DIR}")
        return

    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    existing = {"last_updated": "", "films": []}
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = {"last_updated": "", "films": []}

    by_source = {
        f.get("source"): f for f in existing.get("films", []) if isinstance(f, dict)
    }
    processed = 0
    failed = 0

    for path in sorted(LIKED_DIR.glob("*.txt")):
        file_hash = get_file_hash(path)

        if cache.get(path.name) == file_hash:
            print(f"  [Cached] {path.name}")
            continue

        print(f"  [Analyzing] {path.name} ({path.stat().st_size // 1024} KB)...")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            structure = extract_structure(text, path.name)
            structure["source"] = path.name
            by_source[path.name] = structure
            cache[path.name] = file_hash
            processed += 1
        except Exception as e:
            print(f"  [!] Failed on {path.name}: {e}")
            failed += 1
            continue

    if processed == 0 and not failed:
        print(f"\nNo new/changed files. {OUTPUT_FILE.name} is up to date.")
        return

    existing["films"] = list(by_source.values())
    existing["last_updated"] = datetime.now().isoformat(timespec="seconds")

    temp = OUTPUT_FILE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(OUTPUT_FILE)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    print(f"\nDone. Processed {processed}, failed {failed}.")
    print(f"Total films in {OUTPUT_FILE.name}: {len(existing['films'])}")


if __name__ == "__main__":
    main()