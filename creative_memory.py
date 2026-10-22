"""
Keyword-based search over Virtual Manish's creative examples.

This module is NOT used by Creative_DNA.py in production. It exists as a
debug / inspection tool. To enable it as a fallback, set the environment
variable VM_ALLOW_KEYWORD_FALLBACK=1 before starting Creative_DNA.py.
"""
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "creative_examples"
SUPPORTED_EXTENSIONS = {".txt", ".md"}

STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "i", "my",
    "me", "we", "our", "you", "your", "it", "this", "that", "how",
    "what", "can", "could", "would", "should", "please", "scene",
    "write", "make", "create", "develop", "about", "into", "like",
}


def read_creative_files():
    documents = []
    if not EXAMPLES_DIR.exists():
        return documents

    for path in sorted(EXAMPLES_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Skipped {path.name}: {exc}")
            continue
        if content:
            documents.append({
                "filename": path.name,
                "path": str(path),
                "content": content,
            })
    return documents


def words(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def split_into_passages(content, max_words=180, overlap=35):
    paragraphs = [
        part.strip()
        for part in re.split(
            r"\n\s*\n|^\s*(?:[-*_]\s*){3,}\s*$",
            content,
            flags=re.MULTILINE,
        )
        if part.strip()
    ]
    if not paragraphs and content.strip():
        paragraphs = [content.strip()]

    passages = []
    current = []
    current_size = 0

    for paragraph in paragraphs:
        tokens = paragraph.split()

        if len(tokens) > max_words:
            if current:
                passages.append("\n\n".join(current))
                current, current_size = [], 0

            start = 0
            while start < len(tokens):
                end = min(start + max_words, len(tokens))
                passages.append(" ".join(tokens[start:end]))
                if end >= len(tokens):
                    break
                start = max(start + max_words - overlap, start + 1)
            continue

        if current and current_size + len(tokens) > max_words:
            passages.append("\n\n".join(current))
            tail, tail_size = [], 0
            for previous in reversed(current):
                size = len(previous.split())
                if tail_size + size > overlap:
                    break
                tail.insert(0, previous)
                tail_size += size
            current, current_size = tail, tail_size

        current.append(paragraph)
        current_size += len(tokens)

    if current:
        passages.append("\n\n".join(current))
    return passages


def build_passage_library():
    library = []
    for document in read_creative_files():
        passages = split_into_passages(document["content"])
        for index, passage in enumerate(passages, start=1):
            library.append({
                "filename": document["filename"],
                "passage_number": index,
                "text": passage,
                "words": set(words(passage)),
            })
    return library


def search_creative_examples(query, limit=3):
    query_words = set(words(query)) - STOP_WORDS
    if not query_words:
        return []

    results = []
    for passage in build_passage_library():
        passage_words = passage["words"] - STOP_WORDS
        shared = query_words & passage_words
        if not shared:
            continue
        coverage = len(shared) / max(1, len(query_words))
        precision = len(shared) / max(1, len(passage_words))
        score = 0.8 * coverage + 0.2 * precision
        results.append({
            "filename": passage["filename"],
            "passage_number": passage["passage_number"],
            "score": score,
            "text": passage["text"],
            "matched_words": sorted(shared),
        })

    results.sort(
        key=lambda item: (item["score"], len(item["matched_words"])),
        reverse=True,
    )
    return results[:max(1, limit)]


def main():
    documents = read_creative_files()
    print("\nVIRTUAL MANISH — CREATIVE MEMORY")
    print("=" * 38)
    print(f"Loaded {len(documents)} creative files.")
    if not documents:
        print(f"Add .txt or .md files to: {EXAMPLES_DIR}")
        return
    for document in documents:
        print(f"- {document['filename']}")

    print("\nSearch using keywords. Type /exit to quit.\n")
    while True:
        query = input("Search: ").strip()
        if query.casefold() == "/exit":
            break
        if not query:
            continue
        results = search_creative_examples(query)
        if not results:
            print("No keyword matches. Try more specific words.\n")
            continue
        for result in results:
            print(f"\n[{result['filename']} | passage {result['passage_number']} | score {result['score']:.3f}]")
            print(f"Matched words: {', '.join(result['matched_words'])}")
            print("-" * 50)
            print(result["text"])
            print()


if __name__ == "__main__":
    main()