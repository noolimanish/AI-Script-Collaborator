import json
import re

# The text file you uploaded
INPUT_FILE = "character_bible_character_patterns.txt"
OUTPUT_FILE = "character_bible_full.json"

def parse_character_bible(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    characters = []
    
    # We look for patterns like "A01. Reluctant Protagonist — "
    # This regex finds the ID (e.g., A01), the Name, and the core description.
    pattern = re.compile(r'^([A-Z]\d{2})\.\s+(.*?)\s+—\s+(.*?)$', re.MULTILINE)
    
    matches = list(pattern.finditer(text))
    
    for i, match in enumerate(matches):
        char_id = match.group(1)
        name = match.group(2).strip()
        core_pattern = match.group(3).strip()
        
        # Get the text block for this character (from this match to the next match)
        start_pos = match.end()
        end_pos = matches[i+1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start_pos:end_pos].strip()
        
        # Extract specific fields if they exist in the block
        wants = ""
        fears = ""
        dialogue = ""
        
        # Very basic heuristic to grab sentences if they exist in the text
        if "wants" in block_text.lower():
            wants = "See core pattern."
        if "fears" in block_text.lower():
            fears = "See core pattern."
            
        characters.append({
            "id": char_id,
            "name": name,
            "core_pattern": core_pattern,
            "context": block_text[:500], # Store the rest of the text for reference
            "tone_risk": "Unknown - needs manual review." # We leave this for you to fill
        })
        
    return {"character_archetypes": characters}

if __name__ == "__main__":
    print("Parsing character bible...")
    data = parse_character_bible(INPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Success! Converted {len(data['character_archetypes'])} characters to {OUTPUT_FILE}")