
"""Semantic retrieval for Virtual Manish's creative examples.

Build:  python rag_memory.py --build
Search: python rag_memory.py --search "awkward family conversation"
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from google import genai

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "creative_examples"
INDEX_FILE = BASE_DIR / "rag_index.json"

MODEL = "gemini-embedding-001"
CHUNK_WORDS = 220
OVERLAP_WORDS = 40
RESULTS_LIMIT = 5

# A conservative initial filter, not a universal relevance guarantee.
DEFAULT_MIN_SCORE = 0.35
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in PowerShell first."
        )

    return genai.Client(api_key=api_key)


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


def split_into_chunks(text):
    tokens = text.split()
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_WORDS, len(tokens))
        chunk = " ".join(tokens[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(tokens):
            break

        start = max(0, end - OVERLAP_WORDS)

    return chunks


def embed_texts(client, texts):
    vectors = []

    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]

        response = client.models.embed_content(
            model=MODEL,
            contents=batch,
        )

        embeddings = getattr(response, "embeddings", None)

        if not embeddings or len(embeddings) != len(batch):
            raise RuntimeError(
                "Embedding response was empty or had an unexpected size."
            )

        for embedding in embeddings:
            values = getattr(embedding, "values", None)

            if not values:
                raise RuntimeError("Google returned an empty embedding.")

            vectors.append(list(values))

        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)}")

    return vectors


def file_signature(path):
    """Hash file contents so edits are detected even if timestamps are preserved."""
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
    client = get_client()
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
            and all(
                isinstance(chunk.get("embedding"), list)
                for chunk in cached["chunks"]
            )
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

    all_texts = [
        chunk
        for item in pending
        for chunk in item["texts"]
    ]

    all_vectors = embed_texts(client, all_texts) if all_texts else []
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

    # Write atomically to avoid leaving a partially written index.
    payload = {"model": MODEL, "documents": new_index}
    temp_file = INDEX_FILE.with_suffix(".json.tmp")

    temp_file.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_file.replace(INDEX_FILE)

    count = sum(
        len(document["chunks"])
        for document in new_index.values()
    )

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


def search_memory(
    query,
    limit=RESULTS_LIMIT,
    min_score=DEFAULT_MIN_SCORE,
    verbose=False,
):
    """Return relevant passages, or [] if there are no sufficiently strong matches."""
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

    client = get_client()
    query_vector = embed_texts(client, [query])[0]
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

        # Suppress repeated overlapping chunks with identical openings.
        duplicate = any(
            result["source"] == old["source"]
            and text[:100] == old["text"][:100]
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
            print(
                f"{rank}. {result['source']} | "
                f"similarity {result['score']:.4f}\n"
                f"{result['text']}\n"
            )

    return selected


def main():
    if len(sys.argv) < 2:
        print(
            'Usage:\n'
            '  python rag_memory.py --build\n'
            '  python rag_memory.py --search "your query"'
        )
        return

    command = sys.argv[1].lower()

    try:
        if command == "--build":
            build_index()

        elif command == "--search":
            query = " ".join(sys.argv[2:]).strip()

            if not query:
                print("Provide a query after --search.")
                return

            search_memory(query, verbose=True)

        else:
            print("Unknown command. Use --build or --search.")

    except Exception as exc:
        print(f"RAG operation failed: {exc}")
        print(
            "Check GEMINI_API_KEY, network access, and "
            "whether the embedding model is available."
        )


if __name__ == "__main__":
    main()