"""
Semantic retrieval for Virtual Manish.

LAYER MODEL:
  creative_examples/                → voice (full text)      → search_memory()
  creative_references/
    my_style_pattern/               → voice+decision (text)   → search_style_references()
    liked/                          → structure only          → search_structure_references()
                                       (via liked_structure.json — never raw text)
    failure_modes/                  → rules only              → search_failure_rules()
                                       (via failure_mode_rules.json — never raw text)

liked/ and failure_modes/ raw screenplays are NEVER injected into prompts.
Only their derived JSON is used. This prevents Hollywood prose / bad examples
from leaking into Manish's Hyderabad-flavored output.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "creative_examples"
REFERENCES_DIR = BASE_DIR / "creative_references"
STYLE_PATTERN_DIR = REFERENCES_DIR / "my_style_pattern"
LIKED_DIR = REFERENCES_DIR / "liked"
FAILURE_MODES_DIR = REFERENCES_DIR / "failure_modes"
SCENE_TECHNIQUES_FILE = BASE_DIR / "scene_techniques.json"

INDEX_FILE = BASE_DIR / "rag_index.json"
LIKED_STRUCTURE_FILE = BASE_DIR / "liked_structure.json"
FAILURE_MODE_RULES_FILE = BASE_DIR / "failure_mode_rules.json"

INDEX_VERSION = 3

MODEL = "krutrim-ai-labs/Vyakyarth"
RESULTS_LIMIT = 3
DEFAULT_MIN_SCORE = 0.30
REGISTER_MATCH_BOOST = 0.08
STYLE_PATTERN_BOOST = 0.06
SUPPORTED_EXTENSIONS = {".txt", ".md"}
MAX_PARENT_WORDS = 600

print("Loading local Indic embedding model (Vyakyarth)...")
try:
    _local_model = SentenceTransformer(MODEL)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)


def embed_texts(texts):
    if not texts:
        return []
    embeddings = _local_model.encode(texts, convert_to_tensor=False)
    return [emb.tolist() for emb in embeddings]


def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b) or not vector_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_register_and_clean(text):
    match = re.search(r"\[REGISTER:\s*([^\]]+)\]", text, re.IGNORECASE)
    register = match.group(1).strip().lower() if match else "untagged"
    cleaned = re.sub(r"\[REGISTER:\s*[^\]]+\]\s*", "", text, flags=re.IGNORECASE).strip()
    return register, cleaned


def _registers_match(query_register, doc_register):
    if not query_register or not doc_register:
        return False
    if query_register == "untagged" or doc_register == "untagged":
        return False
    if query_register == doc_register:
        return True
    q_words = set(re.findall(r"[a-z]+", query_register.lower()))
    d_words = set(re.findall(r"[a-z]+", doc_register.lower()))
    return bool(q_words & d_words)


def load_documents_from_dir(directory, required=False):
    if not directory.exists():
        if required:
            raise FileNotFoundError(f"Folder not found: {directory}")
        return []
    documents = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Skipped {path.name}: {exc}")
            continue
        if content:
            documents.append((path, content))
    return documents


def split_into_parents(text):
    scene_pattern = re.compile(r"(?=\b(?:INT|EXT)\.?\s)", re.IGNORECASE)
    raw_scenes = scene_pattern.split(text)
    if len(raw_scenes) <= 1:
        raw_scenes = re.split(r"\n\s*\n", text)
    parents = []
    for scene in raw_scenes:
        scene = scene.strip()
        if not scene:
            continue
        if len(scene.split()) > MAX_PARENT_WORDS:
            sentences = re.split(r"(?<=[.!?]) +", scene)
            current_parent, current_words = [], 0
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_words + s_words > MAX_PARENT_WORDS and current_parent:
                    parents.append("\n\n".join(current_parent))
                    current_parent, current_words = [sentence], s_words
                else:
                    current_parent.append(sentence)
                    current_words += s_words
            if current_parent:
                parents.append("\n\n".join(current_parent))
        else:
            parents.append(scene)
    return parents


def split_into_children(parent_text, max_words=100):
    paragraphs = re.split(r"\n\s*\n", parent_text)
    chunks, current_chunk, current_word_count = [], [], 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        p_words = len(p.split())
        if p_words > max_words:
            sentences = re.split(r"(?<=[.!?]) +", p)
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_word_count + s_words > max_words and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk, current_word_count = [sentence], s_words
                else:
                    current_chunk.append(sentence)
                    current_word_count += s_words
        else:
            if current_word_count + p_words > max_words and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk, current_word_count = [p], p_words
            else:
                current_chunk.append(p)
                current_word_count += p_words
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def file_signature(path):
    content = path.read_bytes()
    identity = path.name.encode("utf-8")
    return hashlib.sha256(identity + b"\0" + content).hexdigest()


def read_existing_index():
    if not INDEX_FILE.exists():
        return {"examples": {}, "references": {}}
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if data.get("model") != MODEL or data.get("version") != INDEX_VERSION:
            return {"examples": {}, "references": {}}
        return {
            "examples": data.get("documents", {}),
            "references": data.get("references", {}),
        }
    except (OSError, json.JSONDecodeError):
        return {"examples": {}, "references": {}}


def _index_corpus(documents, cached_corpus, label):
    updated_corpus = {}
    pending_items = []

    cache_by_name_sig = {}
    for _, doc_val in cached_corpus.items():
        if isinstance(doc_val, dict) and doc_val.get("name") and doc_val.get("signature"):
            cache_by_name_sig[(doc_val["name"], doc_val["signature"])] = doc_val

    for path, raw_content in documents:
        key = str(path.resolve())
        signature = file_signature(path)
        subfolder = path.parent.name.lower()
        cached = cached_corpus.get(key) or cache_by_name_sig.get((path.name, signature))

        if cached and cached.get("parents") and "register" in cached:
            cached["subfolder"] = subfolder
            updated_corpus[key] = cached
            print(f"[{label}] Using cached embeddings: {path.name} ({subfolder})")
            continue

        doc_register, clean_content = extract_register_and_clean(raw_content)
        parents = split_into_parents(clean_content)
        if parents:
            pending_items.append({
                "key": key, "name": path.name, "subfolder": subfolder,
                "signature": signature, "register": doc_register, "parents": parents,
            })

    total_parents = sum(len(i["parents"]) for i in pending_items)
    print(f"\n[{label}] Files found: {len(documents)} | New/Modified scenes needing embeddings: {total_parents}")

    for idx, item in enumerate(pending_items, start=1):
        parent_children_map, all_file_children = [], []
        for parent_text in item["parents"]:
            children = split_into_children(parent_text)
            if not children:
                continue
            start_i = len(all_file_children)
            all_file_children.extend(children)
            end_i = len(all_file_children)
            parent_children_map.append((parent_text, children, start_i, end_i))

        print(f"  -> [{idx}/{len(pending_items)}] Embedding {item['name']} "
              f"({len(item['parents'])} scenes, {len(all_file_children)} chunks)...", flush=True)

        if all_file_children:
            all_embeddings = [
                emb.tolist() for emb in _local_model.encode(
                    all_file_children, batch_size=64,
                    show_progress_bar=len(all_file_children) > 50,
                    convert_to_tensor=False,
                )
            ]
        else:
            all_embeddings = []

        embedded_parents = []
        for parent_text, children, start_i, end_i in parent_children_map:
            chunk_embs = all_embeddings[start_i:end_i]
            embedded_parents.append({
                "parent_text": parent_text,
                "children": [{"text": c, "embedding": e} for c, e in zip(children, chunk_embs)],
            })

        updated_corpus[item["key"]] = {
            "name": item["name"], "subfolder": item["subfolder"],
            "signature": item["signature"], "register": item["register"],
            "parents": embedded_parents,
        }
    return updated_corpus


def build_index():
    print("\n=== BUILD INDEX ===")
    example_docs = load_documents_from_dir(EXAMPLES_DIR, required=True)
    
    # NEW: Load the raw text from my_style_pattern to use as Voice
    style_docs = load_documents_from_dir(STYLE_PATTERN_DIR, required=False)

    existing = read_existing_index()
    new_examples = _index_corpus(example_docs, existing["examples"], "EXAMPLES")
    
    # NEW: Embed the style patterns into the "references" slot of the index
    new_references = _index_corpus(style_docs, existing["references"], "MY_STYLE_PATTERN")

    payload = {
        "version": INDEX_VERSION,
        "model": MODEL,
        "documents": new_examples,
        "references": new_references,
    }
    temp_file = INDEX_FILE.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(INDEX_FILE)

    ex_chunks = sum(len(p["children"]) for d in new_examples.values() for p in d["parents"])
    ref_chunks = sum(len(p["children"]) for d in new_references.values() for p in d["parents"])
    print(f"\nIndex saved: {INDEX_FILE.name}")
    print(f"Indexed Examples (Voice): {len(new_examples)} files ({ex_chunks} chunks)")
    print(f"Indexed Style Patterns (Voice): {len(new_references)} files ({ref_chunks} chunks)")

def _search_corpus_dict(corpus_dict, query, limit, min_score, source_prefix=""):
    if not query or not query.strip() or not corpus_dict:
        return []
    query_register, clean_query = extract_register_and_clean(query)
    if not clean_query:
        clean_query = query.strip()

    query_vector = embed_texts([clean_query])[0]
    results = []

    for doc_key, document in corpus_dict.items():
        doc_register = document.get("register", "untagged")
        subfolder = document.get("subfolder", "") or Path(doc_key).parent.name.lower()
        reg_bonus = REGISTER_MATCH_BOOST if _registers_match(query_register, doc_register) else 0.0
        style_bonus = STYLE_PATTERN_BOOST if "my_style_pattern" in subfolder else 0.0

        for parent in document.get("parents", []):
            best_raw_score = 0.0
            for child in parent.get("children", []):
                score = cosine_similarity(query_vector, child["embedding"])
                if score > best_raw_score:
                    best_raw_score = score
            final_score = (
                min(1.0, best_raw_score + reg_bonus + style_bonus)
                if best_raw_score >= min_score else 0.0
            )
            if final_score >= min_score:
                doc_name = document.get("name", "unknown")
                tier_tag = " [MY_STYLE]" if style_bonus > 0 else ""
                results.append({
                    "source": f"{source_prefix}{doc_name}{tier_tag}",
                    "register": doc_register,
                    "score": final_score,
                    "raw_score": best_raw_score,
                    "register_boosted": reg_bonus > 0,
                    "text": parent["parent_text"],
                })
    results.sort(key=lambda i: i["score"], reverse=True)
    selected, seen = [], set()
    for r in results:
        if r["source"] in seen:
            continue
        seen.add(r["source"])
        selected.append(r)
        if len(selected) >= max(1, limit):
            break
    return selected


def _load_full_index_file():
    if not INDEX_FILE.exists():
        return None
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if data.get("model") != MODEL:
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def search_memory(query, limit=RESULTS_LIMIT, min_score=DEFAULT_MIN_SCORE, verbose=False):
    """Voice reference — searches creative_examples/ full text."""
    data = _load_full_index_file()
    if not data:
        if verbose:
            print("No valid index. Run: python rag_memory.py --build")
        return []
    documents = data.get("documents", {})
    selected = _search_corpus_dict(documents, query, limit=limit, min_score=min_score)
    if verbose:
        print(f"\n[search_memory] {len(selected)} results")
        for r in selected:
            print(f"  - {r['source']} | {r['score']:.3f}")
    return selected


def search_style_references(query, limit=2, min_score=DEFAULT_MIN_SCORE):
    """Voice+decision reference — searches my_style_pattern/ full text."""
    data = _load_full_index_file()
    if not data:
        return []
    references = data.get("references", {})
    return _search_corpus_dict(
        references, query, limit=limit, min_score=min_score,
        source_prefix="STYLE: ",
    )


def search_structure_references(query, limit=2, min_score=0.25):
    """Structure reference — reads liked_structure.json. Never raw text."""
    if not LIKED_STRUCTURE_FILE.exists():
        return []
    try:
        data = json.loads(LIKED_STRUCTURE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    films = data.get("films", [])
    if not films or not query.strip():
        return []

    film_texts = []
    for film in films:
        parts = [
            film.get("spine", ""),
            film.get("mystery_hiding", ""),
            film.get("character_connections", ""),
            film.get("scene_rhythm", ""),
            " ".join(film.get("key_structural_decisions", []) or []),
        ]
        acts = film.get("acts", {})
        parts.extend([acts.get("act_one", ""), acts.get("act_two", ""), acts.get("act_three", "")])
        film_texts.append(" ".join(p for p in parts if p))

    film_embeddings = embed_texts(film_texts)
    query_vector = embed_texts([query.strip()])[0]

    scored = []
    for film, emb in zip(films, film_embeddings):
        sim = cosine_similarity(query_vector, emb)
        if sim >= min_score:
            scored.append({"score": sim, "film": film})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [
        {"source": s["film"].get("source", "unknown"), "score": s["score"], "structure": s["film"]}
        for s in scored[:limit]
    ]


def search_failure_rules(query, limit=5, min_score=0.25):
    """Failure-mode rules — reads failure_mode_rules.json. Never raw text."""
    if not FAILURE_MODE_RULES_FILE.exists():
        return []
    try:
        data = json.loads(FAILURE_MODE_RULES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    rules = data.get("rules", [])
    if not rules or not query.strip():
        return []

    rule_texts = [
        f"{r.get('rule', '')} {r.get('example_avoidance', '')} {r.get('observed_failure', '')}".strip()
        for r in rules
    ]
    if not any(rule_texts):
        return []

    rule_embeddings = embed_texts(rule_texts)
    query_vector = embed_texts([query.strip()])[0]

    scored = []
    for rule, emb in zip(rules, rule_embeddings):
        sim = cosine_similarity(query_vector, emb)
        if sim >= min_score:
            scored.append({"score": sim, "rule": rule})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [
        {
            "source": s["rule"].get("source", "unknown"),
            "score": s["score"],
            "failure_type": s["rule"].get("failure_type", "other"),
            "observed_failure": s["rule"].get("observed_failure", ""),
            "rule": s["rule"].get("rule", ""),
            "example_avoidance": s["rule"].get("example_avoidance", ""),
        }
        for s in scored[:limit]
    ]

def search_scene_techniques(query, limit=2):
    """Searches scene_techniques.json for openings and closings."""
    if not SCENE_TECHNIQUES_FILE.exists():
        return []
    try:
        data = json.loads(SCENE_TECHNIQUES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
        
    techniques = data.get("techniques", [])
    if not techniques:
        return []

    # We use simple keyword matching for techniques (no need for heavy embeddings)
    query_lower = query.lower()
    scored = []
    
    for tech in techniques:
        score = 0
        if tech.get("register", "").lower() in query_lower:
            score += 5
        if tech.get("type", "").lower() in query_lower:
            score += 3
        if tech.get("position", "").lower() in query_lower:
            score += 2
            
        # Give a base score so we always return something useful
        scored.append((score, tech))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t[1] for t in scored[:limit]]

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