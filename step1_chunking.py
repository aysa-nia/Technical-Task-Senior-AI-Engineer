import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "corpus.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"   

TARGET_CHUNK_CHARS = 1600
CHUNK_OVERLAP_SENTENCES = 1

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SECTION_TYPE_RULES = [
    (("specification",), "spec"),
    (("maintenance", "interval", "schedule", "replacement", "calibration"), "maintenance"),
    (("procedure", "checklist", "startup", "shutdown","lockout", "commissioning"),"procedure"),
    (("error code",), "error-code"),
    (("vibration",), "spec"),
    (("ordering", "spare part"), "procedure"),
    (("logging",), "reference"),
]

def infer_section_type(title: str) -> str:
    t = title.lower()
    for keywords, label in _SECTION_TYPE_RULES:
        if any(k in t for k in keywords):
            return label
    return "reference"

def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

def chunk_text(text: str, target_chars: int = TARGET_CHUNK_CHARS, overlap_sentences: int = CHUNK_OVERLAP_SENTENCES):
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent) + 1         
        if current and current_len + sent_len > target_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks

def load_docs(path: str):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs

def build_chunks(docs):
    chunks = []
    for d in docs:
        section_type = infer_section_type(d["title"])
        pieces = chunk_text(d["text"])
        n = len(pieces)
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id":       f"{d['id']}-{i}",
                "doc_id":         d["id"],
                "title":          d["title"],
                "section_type":   section_type,
                "chunk_index":    i,
                "n_chunks_in_doc": n,
                "text":    f"{d['title']}: {piece}",
                "raw_text": piece,
            })
    return chunks

def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts)
    vecs = np.asarray(vecs, dtype="float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

def retrieve(query: str, chunks, vectors, model, top_k: int = 1):
    q = model.encode([query])[0].astype("float32")
    q = q / np.linalg.norm(q)
    sims = vectors @ q
    order = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in order]

def answer(query: str, chunks, vectors, model):
    results = retrieve(query, chunks, vectors, model, top_k=1)
    hit, score = results[0]
    return f"[{hit['doc_id']}] {hit['raw_text']}"


if __name__ == "__main__":
    docs = load_docs(CORPUS_PATH)
    chunks = build_chunks(docs)
    for c in chunks:
        print(f"  {c['chunk_id']:12s} | type={c['section_type']:12s} | "f"chars={len(c['text']):4d} | chunks_in_doc={c['n_chunks_in_doc']}")
        preview = c['text'][:80].replace('\n', ' ')
        print(f"               preview: {preview!r}")
    print(f"\nTotal chunks: {len(chunks)}  (expected ≈ 16, one per doc)")
