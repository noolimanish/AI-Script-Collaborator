"""
Extracts FAILURE-MODE rules from creative_references/failure_modes/.

Manish drops films/scenes here that failed in a specific way, along with notes.
This script extracts transferable rules a writer should AVOID.

Output: failure_mode_rules.json
Incremental: only re-processes files whose SHA-256 changed.
Safe failure: does NOT cache on error — so retries work next run.
"""
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from google import genai

BASE_DIR = Path(__file__).resolve().parent
FAILURE_DIR = BASE_DIR / "creative_references" / "failure_modes"
OUTPUT_FILE = BASE_DIR / "failure_mode_rules.json"
CACHE_FILE = BASE_DIR / "failure_mode_cache.json"

MODEL = os.getenv("VM_MODEL", "gemini-3.8-flash")
client = genai.Client()

FAILURE_TYPES = [
    "unearned_resolution",
    "forced_emotion",
    "generic_dialogue",
    "plot_over_character",
    "sagging_middle",
    "telegraphed_twist",
    "unmotivated_character_turn",
    "wasted_subplot",
    "convenient_coincidence",
    "unclear_stakes",
    "tonal_inconsistency",
    "padded_scene",
    "unresolved_arc",
    "weak_villain",
    "other",
]


def get_file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_text(text, max_words=3500):
    words = text.split()
    if len(words) <= max_words:
        return text
    third = max_words // 3
    mid = len(words) // 2
    return (
        " ".join(words[:third])
        + "\n\n[... omitted ...]\n\n"
        + " ".join(words[mid - third // 2 : mid + third // 2])
        + "\n\n[... omitted ...]\n\n"
        + " ".join(words[-third:])
    )


def extract_failure_rules(text, filename):
    sampled = _sample_text(text)
    prompt = f"""
Manish flagged this screenplay/scene ({filename}) as a NEGATIVE example —
something about it FAILED.

If the text contains Manish's own notes (lines with "NOTE:", "WHY:", "FAIL:", or similar),
respect them — they tell you exactly why he flagged it.

Extract 1 to 4 general rules a writer should AVOID, based on what went wrong.
These rules must be TRANSFERABLE — applicable to any scene, not tied to this film's specifics.

Failure types to consider:
{json.dumps(FAILURE_TYPES)}

Return ONLY valid JSON:
{{
  "rules": [
    {{
      "failure_type": "one of the failure_types above",
      "observed_failure": "What the writer attempted in this scene/film, and where it failed. Specific.",
      "rule": "The general, transferable rule a writer should follow to avoid this failure.",
      "example_avoidance": "Concrete directive: how to write this correctly instead."
    }}
  ]
}}

Be specific and behavioral. "Don't write boring scenes" is useless.
"Don't resolve a physical confrontation with a phone call — physical stakes must be
resolved physically, or the scene breaks its own logic" is useful.

SCRIPT TEXT:
{sampled}
"""

    stream = client.interactions.create(
        model=MODEL,
        input=prompt,
        system_instruction="You are a strict failure-mode analyst. Extract transferable rules. Return valid JSON only.",
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
    return json.loads(match.group(0)).get("rules", [])


def main():
    print("Extracting FAILURE-MODE rules from creative_references/failure_modes/ ...")

    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(FAILURE_DIR.glob("*.txt"))
    if not files:
        print(f"\nFolder {FAILURE_DIR.name}/ is empty.")
        print("Add failed-film notes/excerpts there and re-run this script.")
        empty = {"last_updated": datetime.now().isoformat(timespec="seconds"), "rules": []}
        OUTPUT_FILE.write_text(json.dumps(empty, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote empty {OUTPUT_FILE.name} — ready for future rules.")
        return

    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    existing = {"last_updated": "", "rules": []}
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = {"last_updated": "", "rules": []}

    existing_rule_texts = {
        r.get("rule", "").casefold().strip()
        for r in existing.get("rules", [])
        if isinstance(r, dict)
    }
    merged = list(existing.get("rules", []))
    processed = 0
    added = 0
    failed = 0

    for path in files:
        file_hash = get_file_hash(path)
        if cache.get(path.name) == file_hash:
            print(f"  [Cached] {path.name}")
            continue

        print(f"  [Analyzing] {path.name}...")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            rules = extract_failure_rules(text, path.name)
            for r in rules:
                if not isinstance(r, dict) or not r.get("rule"):
                    continue
                key = r["rule"].casefold().strip()
                if key in existing_rule_texts:
                    continue
                r["source"] = path.name
                merged.append(r)
                existing_rule_texts.add(key)
                added += 1
            cache[path.name] = file_hash
            processed += 1
        except Exception as e:
            print(f"  [!] Failed on {path.name}: {e}")
            failed += 1
            continue

    if processed == 0 and not failed:
        print(f"\nNo new/changed files. {OUTPUT_FILE.name} is up to date.")
        return

    existing["rules"] = merged
    existing["last_updated"] = datetime.now().isoformat(timespec="seconds")
    temp = OUTPUT_FILE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(OUTPUT_FILE)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    print(f"\nDone. Processed {processed}, failed {failed}, added {added} new rules.")
    print(f"Total rules in {OUTPUT_FILE.name}: {len(merged)}")


if __name__ == "__main__":
    main()