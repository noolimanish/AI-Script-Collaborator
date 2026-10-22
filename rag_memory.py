"""
Semantic retrieval for Virtual Manish's creative examples.

Build:  python rag_memory.py --build
Search: python rag_memory.py --search "awkward family conversation"
Inspect one match with score: python rag_memory.py --search "..." --verbose
"""
import hashlib
import json
import math
import os
import sys
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "creative_examples"
INDEX_FILE = BASE_DIR / "rag_index.json"

# Local model for precise Telugu-English search
MODEL = "krutrim-ai-labs/Vyakyarth"
RESULTS_LIMIT = 3
DEFAULT_MIN_SCORE = 0.30 # Lowered for Vyakyarth
SUPPORTED_EXTENSIONS = {".txt", ".md"}

# Max words to send to Gemini (the "Parent")
MAX_PARENT_WORDS = 600

print("Loading local Indic embedding model (Vyakyarth)...")
try:
    _local_model = SentenceTransformer(MODEL)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)


def load_documents():
    if not EXAMPLES_DIR.exists():
        raise FileNotFoundError(f"Creative examples folder not found: {EXAMPLES_DIR}")
    documents = []
    for path in sorted(EXAMPLES_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if content:
            documents.append((path, content))
    return documents


def split_into_parents(text):
    """
    Splits the text into full scenes (Parents). 
    We split by INT./EXT. headings, or double line breaks if no headings exist.
    Caps each parent at MAX_PARENT_WORDS to avoid context rot.
    """
    # Try splitting by standard screenplay headings
    scene_pattern = re.compile(r'(?=\b(?:INT|EXT)\.?\s)', re.IGNORECASE)
    raw_scenes = scene_pattern.split(text)
    
    # If no headings, split by double line breaks
    if len(raw_scenes) <= 1:
        raw_scenes = re.split(r'\n\s*\n', text)

    parents = []
    for scene in raw_scenes:
        scene = scene.strip()
        if not scene:
            continue
            
        # If a scene is too long, split it by sentences
        if len(scene.split()) > MAX_PARENT_WORDS:
            sentences = re.split(r'(?<=[.!?]) +', scene)
            current_parent = []
            current_words = 0
            
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_words + s_words > MAX_PARENT_WORDS and current_parent:
                    parents.append("\n\n".join(current_parent))
                    current_parent = [sentence]
                    current_words = s_words
                else:
                    current_parent.append(sentence)
                    current_words += s_words
            if current_parent:
                parents.append("\n\n".join(current_parent))
        else:
            parents.append(scene)
            
    return parents


def split_into_children(parent_text, max_words=100):
    """
    Splits a Parent scene into Child chunks for embedding (Vyakyarth).
    """
    paragraphs = re.split(r'\n\s*\n', parent_text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        p_words = len(p.split())
        
        if p_words > max_words:
            sentences = re.split(r'(?<=[.!?]) +', p)
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_word_count + s_words > max_words and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = s_words
                else:
                    current_chunk.append(sentence)
                    current_word_count += s_words
        else:
            if current_word_count + p_words > max_words and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_word_count = p_words
            else:
                current_chunk.append(p)
                current_word_count += p_words
                
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks


def file_signature(path):
    content = path.read_bytes()
    identity = str(path.resolve()).encode("utf-8")
    return hashlib.sha256(identity + b"\0" + content).hexdigest()


def read_existing_index():
    if not INDEX_FILE.exists():
        return {}
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if data.get("model") != MODEL:
            return {}
        documents = data.get("documents", {})
        return documents if isinstance(documents, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def build_index():
    documents = load_documents()
    old_documents = read_existing_index()

    new_index = {}
    pending_parents = []

    for path, content in documents:
        key = str(path.resolve())
        signature = file_signature(path)
        cached = old_documents.get(key)

        if (
            cached
            and cached.get("signature") == signature
            and cached.get("parents")
        ):
            new_index[key] = cached
            print(f"Using cached embeddings: {path.name}")
            continue

        parents = split_into_parents(content)
        
        if parents:
            pending_parents.append({
                "key": key,
                "name": path.name,
                "signature": signature,
                "parents": parents,
            })

    total_parents = sum(len(item["parents"]) for item in pending_parents)
    print(f"\nCreative files found: {len(documents)}")
    print(f"Scenes (Parents) needing embeddings: {total_parents}")

    # Embed the child chunks for each parent
    for item in pending_parents:
        embedded_parents = []
        for parent_text in item["parents"]:
            children = split_into_children(parent_text)
            child_embeddings = _local_model.encode(children, convert_to_tensor=False)
            
            embedded_parents.append({
                "parent_text": parent_text,
                "children": [
                    {"text": child, "embedding": emb.tolist()}
                    for child, emb in zip(children, child_embeddings)
                ]
            })
        new_index[item["key"]] = {
            "name": item["name"],
            "signature": item["signature"],
            "parents": embedded_parents,
        }

    payload = {"model": MODEL, "documents": new_index}
    temp_file = INDEX_FILE.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(INDEX_FILE)

    count = sum(len(p["children"]) for d in new_index.values() for p in d["parents"])
    print(f"\nIndex saved: {INDEX_FILE.name}")
    print(f"Indexed files: {len(new_index)}")
    print(f"Total child chunks: {count}")


def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b) or not vector_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def search_memory(query, limit=RESULTS_LIMIT, min_score=DEFAULT_MIN_SCORE, verbose=False):
    if not query or not query.strip():
        return []

    if not INDEX_FILE.exists():
        if verbose:
            print("No index found. Run: python rag_memory.py --build")
        return []

    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if verbose:
            print(f"Cannot read index: {exc}")
        return []

    if data.get("model") != MODEL:
        if verbose:
            print("Embedding model changed. Run: python rag_memory.py --build")
        return []

    documents = data.get("documents", {})
    if not documents:
        return []

    query_vector = _local_model.encode([query], convert_to_tensor=False)[0].tolist()

    results = []
    for document in documents.values():
        for parent in document.get("parents", []):
            # Find the best matching child inside this parent
            best_score = 0.0
            for child in parent.get("children", []):
                score = cosine_similarity(query_vector, child["embedding"])
                if score > best_score:
                    best_score = score
            
            if best_score >= min_score:
                results.append({
                    "source": document.get("name", "unknown"),
                    "score": best_score,
                    "text": parent["parent_text"], # Return the FULL SCENE (Parent)
                })

    results.sort(key=lambda item: item["score"], reverse=True)

    # Deduplicate by source so we don't send the same file twice
    selected = []
    seen_sources = set()
    for result in results:
        source = result["source"]
        if source in seen_sources:
            continue
        seen_sources.add(source)
        selected.append(result)
        if len(selected) >= max(1, limit):
            break

    if verbose:
        print(f"\nRelevant results for: {query}\n")
        if not selected:
            print(f"No passages scored at least {min_score:.2f}.")
        for rank, result in enumerate(selected, start=1):
            print(f"{rank}. {result['source']} | similarity {result['score']:.4f}\n")
            print(result["text"][:200] + "...\n[TRUNCATED FOR DISPLAY]\n")

    return selected


def main():
    if len(sys.argv) < 2:
        print("Usage:\n  python rag_memory.py --build\n  python rag_memory.py --search \"your query\"")
        return

    command = sys.argv[1].lower()
    try:
        if command == "--build":
            build_index()
        elif command == "--search":
            args = sys.argv[2:]
            verbose = "--verbose" in args
            args = [a for a in args if a != "--verbose"]
            query = " ".join(args).strip()
            if not query:
                print("Provide a query after --search.")
                return
            search_memory(query, verbose=True)
        else:
            print("Unknown command. Use --build or --search.")
    except Exception as exc:
        print(f"RAG operation failed: {exc}")


if __name__ == "__main__":
    main()