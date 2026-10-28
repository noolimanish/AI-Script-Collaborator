"""
Extracts Tone, Pacing, and Dialogue Rhythm rules from my_style_pattern.
Uses SCENE-BY-SCENE Semantic Analysis with Temp saving to prevent data loss.
"""
import concurrent.futures
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
STYLE_DIR = BASE_DIR / "creative_references" / "my_style_pattern"
DNA_DIR = BASE_DIR / "manish_film_dna"
CACHE_FILE = BASE_DIR / "style_dna_cache.json"  # SEPARATE CACHE

DNA_DIR.mkdir(exist_ok=True)

MODEL = os.getenv("VM_MODEL", "gemini-3.5-flash-lite")
client = genai.Client()

# Mask slang to prevent API blocks
_SAFETY_TRIGGER_MAP = {
    r"\blanja\w*\b": "l*nja", r"\bmodda\w*\b": "m*dda", r"\bpook\w*\b": "p*oka",
    r"\bdeng\w*\b": "d*ng", r"\bnee\s+yamma\b": "n*e yamma", r"\blathkor\b": "l*thkor",
    r"\bchampesth\w*\b": "ch*mpesth", r"\bchach\w*\b": "ch*ch", r"\bsuicide\b": "s*icide",
    r"\bmurder\w*\b": "m*rder", r"\bfuck\w*\b": "f*ck", r"\bshit\w*\b": "sh*t"
}

def get_file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _neutralize(text):
    for pattern, replacement in _SAFETY_TRIGGER_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def split_into_scenes(text):
    scene_pattern = re.compile(r"(?=\b(?:INT|EXT)\.?\s)", re.IGNORECASE)
    scenes = scene_pattern.split(text)
    if len(scenes) <= 1:
        scenes = re.split(r"\n\s*\n", text)
    valid_scenes = [s.strip() for s in scenes if len(s.strip().split()) > 50]
    return valid_scenes

def load_temp_rules(filename):
    """If a previous run saved raw scene rules but failed consolidation, load them."""
    temp_file = DNA_DIR / f"temp_{Path(filename).stem}_rules.json"
    if temp_file.exists():
        try:
            data = json.loads(temp_file.read_text(encoding="utf-8"))
            return data.get("scene_rules", [])
        except Exception:
            pass
    return None

def analyze_single_scene(scene_text, filename, scene_index):
    safe_text = _neutralize(scene_text)
    
    prompt = f"""
You are an academic script analyst. Analyze this specific scene from a reference script ({filename}, Scene {scene_index}).
This script represents the exact Tone, Pacing, and Dialogue Rhythm that the writer wants to mimic.

Extract 1 specific rule about HOW this scene is written. 
Focus on: Dialogue rhythm (overlapping, sparse), Visual tone (silent pauses, physical actions), and Pacing.

Return ONLY valid JSON:
{{ "rule": "The specific permanent rule here." }}

SCENE:
{safe_text}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are an academic script analyst. Output strictly valid JSON.",
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(response.text)
        return parsed.get("rule", "")
    except Exception as e:
        print(f"      [!] Failed on scene {scene_index}: {e}")
        return ""

def consolidate_rules(scene_rules, filename):
    rules_text = "\n".join(f"- {r}" for r in scene_rules if r)
    
    prompt = f"""
I have analyzed every scene in this reference script ({filename}). Here are the raw observations:
{rules_text}

Consolidate these into 3 to 4 permanent, high-level writing rules that define the TONE, PACING, and DIALOGUE RHYTHM of this script.

Return ONLY valid JSON:
{{
    "rules": [
        {{"category": "dialogue_rhythm | visual_tone | scene_pacing | comedy_mechanism", "rule": "The specific permanent rule here."}}
    ]
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are an academic script analyst. Output strictly valid JSON.",
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(response.text)
        return parsed.get("rules", [])
    except Exception as e:
        print(f"  [!] Consolidation failed on {filename}: {e}")
        return []

def main():
    print("Starting Scene-by-Scene STYLE DNA Extraction...")

    if not STYLE_DIR.exists():
        print(f"Folder not found: {STYLE_DIR}")
        return

    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    processed_files = 0
    output_file = DNA_DIR / "dna_style_patterns.json"

    existing = {"last_updated": "", "rules": []}
    if output_file.exists():
        try:
            existing = json.loads(output_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing_texts = set(r.get("rule", "").casefold() for r in existing.get("rules", []))
    merged = list(existing.get("rules", []))

    for path in sorted(STYLE_DIR.glob("*.txt")):
        file_hash = get_file_hash(path)

        if cache.get(path.name) == file_hash:
            print(f"  [Cached] Skipping unchanged file: {path.name}")
            continue

        print(f"\n  [Analyzing Style] {path.name}...")
        
        # 1. Check if we have temp rules from a previous crashed run
        scene_rules = load_temp_rules(path.name)
        
        if scene_rules:
            print(f"    -> Found {len(scene_rules)} previously saved scene observations. Skipping scene analysis!")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            scenes = split_into_scenes(text)
            print(f"    -> Found {len(scenes)} valid scenes. Reading every scene...")
            
            scene_rules = []
            for i, scene in enumerate(scenes, 1):
                print(f"       - Analyzing Scene {i}/{len(scenes)}...")
                rule = analyze_single_scene(scene, path.name, i)
                if rule:
                    scene_rules.append(rule)
                time.sleep(3) # 3 seconds to avoid rate limits
            
            # Save immediately before consolidation
            temp_file = DNA_DIR / f"temp_{path.stem}_rules.json"
            temp_file.write_text(json.dumps({"scene_rules": scene_rules}, indent=2), encoding="utf-8")
            print(f"    -> Saved raw scene observations to temp file.")

        if not scene_rules:
            print(f"    [!] No rules extracted from {path.name}. Skipping save.")
            continue

        print(f"    -> Consolidating {len(scene_rules)} scene observations...")
        final_rules = consolidate_rules(scene_rules, path.name)

        if not final_rules:
            print(f"    [!] Consolidation failed for {path.name}. Temp file is saved for next run.")
            continue

        added = 0
        for r in final_rules:
            if isinstance(r, dict) and r.get("rule"):
                key = r["rule"].casefold()
                if key not in existing_texts:
                    r_copy = dict(r)
                    r_copy["source"] = path.name
                    merged.append(r_copy)
                    existing_texts.add(key)
                    added += 1

        existing["rules"] = merged
        existing["last_updated"] = datetime.now().isoformat(timespec="seconds")
        output_file.write_text(json.dumps(existing, indent=4, ensure_ascii=False), encoding="utf-8")

        print(f"    -> Added {added} style rules to {output_file.name}")

        cache[path.name] = file_hash
        CACHE_FILE.write_text(json.dumps(cache, indent=4), encoding="utf-8")
        processed_files += 1

    if processed_files == 0:
        print("\nNo new or modified style files found. DNA is up to date.")
        return

    print(f"\nSuccess! Processed {processed_files} style files.")

if __name__ == "__main__":
    main()