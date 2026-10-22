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

# We use a local model that understands Indic languages
MODEL = "ola-krutrim/Vyakyarth"
RESULTS_LIMIT = 5

DEFAULT_MIN_SCORE = 0.55
SUPPORTED_EXTENSIONS = {".txt", ".md"}

# Load the model once when the script starts
print("Loading local Indic embedding model (Vyakyarth)...")
try:
    local_embed_model = SentenceTransformer(MODEL)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Falling back to standard English model...")
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    local_embed_model = SentenceTransformer(MODEL)


def load_documents():
    if not EXAMPLES_DIR.exists():
        raise FileNotFoundError(
            f"Creative examples folder not found: {EXAMPLES_DIR}"
        )
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


def split_into_chunks(text, max_words=400):
    """
    Smart, format-agnostic chunking by paragraphs.
    """
    paragraphs = re.split(r'\n\s*\n', text)
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


def embed_texts(client, texts):
    """
    Local embedding using SentenceTransformers.
    The 'client' argument is ignored (kept for compatibility).
    """
    embeddings = local_embed_model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()


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
        print("Existing index is unreadable; rebuilding it.")
        return {}


def build_index():
    documents = load_documents()
    old_documents = read_existing_index()

    new_index = {}
    pending = []

    for path, content in documents:
        key = str(path.resolve())
        signature = file_signature(path)
        cached = old_documents.get(key)

        if (
            cached
            and cached.get("signature") == signature
            and cached.get("chunks")
            and all(isinstance(c["embedding"], list) for c in cached["chunks"])
        ):
            new_index[key] = cached
            print(f"Using cached embeddings: {path.name}")
            continue

        chunks = split_into_chunks(content)
        if chunks:
            pending.append({
                "key": key,
                "name": path.name,
                "signature": signature,
                "texts": chunks,
            })

    total = sum(len(item["texts"]) for item in pending)
    print(f"\nCreative files found: {len(documents)}")
    print(f"Passages needing embeddings: {total}")

    all_texts = [chunk for item in pending for chunk in item["texts"]]
    all_vectors = embed_texts(None, all_texts) if all_texts else []
    position = 0

    for item in pending:
        embedded_chunks = []
        for chunk_text in item["texts"]:
            embedded_chunks.append({
                "text": chunk_text,
                "embedding": all_vectors[position],
            })
            position += 1
        new_index[item["key"]] = {
            "name": item["name"],
            "signature": item["signature"],
            "chunks": embedded_chunks,
        }

    payload = {"model": MODEL, "documents": new_index}
    temp_file = INDEX_FILE.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(INDEX_FILE)

    count = sum(len(d["chunks"]) for d in new_index.values())
    print(f"\nIndex saved: {INDEX_FILE.name}")
    print(f"Indexed files: {len(new_index)}")
    print(f"Total passages: {count}")


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

    query_vector = embed_texts(None, [query])[0]

    results = []
    for document in documents.values():
        for chunk in document.get("chunks", []):
            vector = chunk.get("embedding", [])
            score = cosine_similarity(query_vector, vector)
            if score >= min_score:
                results.append({
                    "source": document.get("name", "unknown"),
                    "score": score,
                    "text": chunk.get("text", ""),
                })

    results.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    for result in results:
        text = result["text"].strip()
        if not text:
            continue
        duplicate = any(
            result["source"] == old["source"] and text[:100] == old["text"][:100]
            for old in selected
        )
        if not duplicate:
            selected.append(result)
        if len(selected) >= max(1, limit):
            break

    if verbose:
        print(f"\nRelevant results for: {query}\n")
        if not selected:
            print(f"No passages scored at least {min_score:.2f}.")
        for rank, result in enumerate(selected, start=1):
            print(f"{rank}. {result['source']} | similarity {result['score']:.4f}\n{result['text']}\n")

    return selected


def main():
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python rag_memory.py --build\n"
            '  python rag_memory.py --search "your query"'
        )
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