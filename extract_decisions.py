"""
Extracts DECISION RULES from your personal movie/script rewrites.
Place your text files in the 'my_decisions/' folder.
"""
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from google import genai

BASE_DIR = Path(__file__).resolve().parent
DECISIONS_DIR = BASE_DIR / "my_decisions"
OUTPUT_FILE = BASE_DIR / "decision_memory.json"
CACHE_FILE = BASE_DIR / "decision_cache.json"

MODEL = os.getenv("VM_MODEL", "gemini-3.8-flash")
client = genai.Client()

def get_file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def extract_rules(text, filename):
    prompt = f"""
You are an expert script analyst. I am going to give you a log of a movie or script, 
what the original creator did, and what the writer (Manish) would have done differently.

Your job is to extract the CREATIVE PRINCIPLE or DECISION RULE from Manish's reasoning.
Do not summarize the plot. Extract the underlying logic of WHY his choice is better.

Return ONLY valid JSON:
{{
  "rules": [
    {{
      "decision_type": "e.g. 'internal_vs_external_climax', 'subverted_trope', 'tragic_irony'",
      "rule": "The permanent decision-making rule Manish follows. E.g. 'Victory must be internal, not external. Keep the systemic threat intact.'",
      "reasoning": "Why this rule matters to Manish (based on his 'WHY' section).",
      "source": "{filename}"
    }}
  ]
}}

TEXT TO ANALYZE:
{text}
"""
    try:
        stream = client.interactions.create(
            model=MODEL, input=prompt,
            system_instruction="Analyze creative decisions. Return valid JSON only.",
            store=False, stream=True,
        )
        raw = ""
        for event in stream:
            if hasattr(event, "delta") and hasattr(event.delta, "text") and event.delta.text:
                raw += event.delta.text

        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group(0)).get("rules", [])
    except Exception as e:
        print(f"  [!] Failed on {filename}: {e}")
        return None

def main():
    print("Extracting Decision Rules from my_decisions/ ...")
    DECISIONS_DIR.mkdir(exist_ok=True)
    
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    existing = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else {"last_updated": "", "rules": []}
    
    existing_texts = {r.get("rule", "").casefold() for r in existing.get("rules", [])}
    merged = list(existing.get("rules", []))
    
    processed = 0
    for path in sorted(DECISIONS_DIR.glob("*.txt")):
        file_hash = get_file_hash(path)
        if cache.get(path.name) == file_hash:
            continue
            
        print(f"  [Analyzing] {path.name}...")
        text = path.read_text(encoding="utf-8", errors="replace")
        rules = extract_rules(text, path.name)
        
        if rules is not None:
            for r in rules:
                if r.get("rule") and r["rule"].casefold() not in existing_texts:
                    merged.append(r)
                    existing_texts.add(r["rule"].casefold())
            cache[path.name] = file_hash
            processed += 1

    if processed > 0:
        existing["rules"] = merged
        existing["last_updated"] = datetime.now().isoformat()
        OUTPUT_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"\nSuccess! Added new rules. Total: {len(merged)}")
    else:
        print("\nNo new/changed files found.")

if __name__ == "__main__":
    main()