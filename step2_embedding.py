import json
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_PATH   = "corpus.jsonl"
CHUNK_SIZE    = 400
EMBED_MODEL   = "BAAI/bge-large-en-v1.5"

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_docs(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i + size] for i in range(0, len(text), size)]


def build_baseline_chunks(docs):
    chunks = []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunks.append({"doc_id": d["id"], "title": d["title"], "text": c})
    return chunks


def build_index(chunks, model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def retrieve(query: str, chunks, vectors: np.ndarray, model: SentenceTransformer, top_k: int = 4):
    q_vec = model.encode(
        [_QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype("float32")
    sims  = vectors @ q_vec
    order = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in order]


def build_step2_pipeline(corpus_path: str = CORPUS_PATH):
    print(f"  [step2]    Loading model: {EMBED_MODEL}")
    model  = SentenceTransformer(EMBED_MODEL)

    docs   = load_docs(corpus_path)
    chunks = build_baseline_chunks(docs)       
    vectors = build_index(chunks, model)

    def query_fn(query: str, top_k: int):
        results = retrieve(query, chunks, vectors, model, top_k=top_k)
        return [(c["doc_id"], score) for c, score in results]

    meta = {
        "name":     "step2_embedding",
        "model":    EMBED_MODEL,
        "chunking": f"fixed-{CHUNK_SIZE}-chars",   
        "n_chunks": len(chunks),
    }
    return query_fn, meta

