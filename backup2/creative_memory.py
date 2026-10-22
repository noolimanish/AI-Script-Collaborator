
import re
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent / "creative_examples"

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def read_creative_files():
    """Read creative text files without changing the originals."""
    documents = []

    if not EXAMPLES_DIR.exists():
        print(f"Examples folder not found: {EXAMPLES_DIR}")
        return documents

    for path in sorted(EXAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            content = path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            print(f"Skipped {path.name}: unable to read as UTF-8.")
            continue

        if content:
            documents.append({
                "filename": path.name,
                "content": content,
            })

    return documents


def words(text):
    """Normalize text for simple local keyword matching."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def split_into_passages(content, max_words=180, overlap=35):
    """
    Split a screenplay into manageable passages.
    Keep some overlap so scene context isn't abruptly lost.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content)
        if paragraph.strip()
    ]

    passages = []
    current = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph_size = len(paragraph.split())

        if current and current_size + paragraph_size > max_words:
            passages.append("\n\n".join(current))

            # Retain a little context from the previous passage.
            tail = []
            tail_size = 0

            for previous in reversed(current):
                size = len(previous.split())

                if tail_size + size > overlap:
                    break

                tail.insert(0, previous)
                tail_size += size

            current = tail
            current_size = tail_size

        current.append(paragraph)
        current_size += paragraph_size

    if current:
        passages.append("\n\n".join(current))

    return passages


def build_passage_library():
    """Create a searchable in-memory library from the text files."""
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
    """
    Return passages sharing useful query keywords.

    This is a simple first version, not a semantic search engine.
    """
    query_words = set(words(query))

    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on",
        "for", "with", "is", "are", "i", "my", "me", "how",
        "what", "can", "we", "this", "that", "scene", "write"
    }

    query_words -= stop_words

    if not query_words:
        return []

    library = build_passage_library()
    results = []

    for passage in library:
        shared = query_words & passage["words"]

        if not shared:
            continue

        score = len(shared) / len(query_words)

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

    return results[:limit]


def main():
    documents = read_creative_files()

    print("\nVIRTUAL MANISH — CREATIVE MEMORY")
    print("=" * 38)
    print(f"Loaded {len(documents)} creative files.")

    if not documents:
        print("Add .txt or .md files to the creative_examples folder.")
        return

    for document in documents:
        print(f"- {document['filename']}")

    print("\nSearch for scenes using keywords.")
    print("Example: mother son book")
    print("Example: wines shop funny conversation")
    print("Type /exit to quit.\n")

    while True:
        query = input("Search: ").strip()

        if query.casefold() == "/exit":
            break

        if not query:
            continue

        results = search_creative_examples(query)

        if not results:
            print("No keyword matches found. Try different words.\n")
            continue

        for result in results:
            print(
                f"\n[{result['filename']} | "
                f"passage {result['passage_number']} | "
                f"score {result['score']:.2f}]"
            )
            print(f"Matched words: {', '.join(result['matched_words'])}")
            print("-" * 50)
            print(result["text"])
            print()


if __name__ == "__main__":
    main()