import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH   = "corpus.jsonl"
CHUNK_SIZE    = 400
EMBED_MODEL   = "BAAI/bge-large-en-v1.5"
RRF_K         = 60          

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN_RE     = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")   


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
        for idx, c in enumerate(chunk_text(d["text"])):
            chunks.append({
                "doc_id": d["id"], 
                "chunk_id": f"{d['id']}_ch{idx}", 
                "title": d["title"], 
                "text": c
            })
    return chunks


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def build_dense_index(chunks, model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_bm25_index(chunks) -> BM25Okapi:
    return BM25Okapi([_tokenize(c["text"]) for c in chunks])


def dense_search(query: str, chunks, vectors: np.ndarray, model: SentenceTransformer, top_k: int = 10):
    q_vec = model.encode(
        [_QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype("float32")
    sims  = vectors @ q_vec
    order = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in order]


def bm25_search(query: str, chunks, bm25_index: BM25Okapi, top_k: int = 5):
    scores = bm25_index.get_scores(_tokenize(query))
    order  = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in order]


def hybrid_search(query: str, chunks, vectors: np.ndarray, bm25_index: BM25Okapi, model: SentenceTransformer, top_k_each: int = 5, fused_top_k: int = 10):
    dense_results = dense_search(query, chunks, vectors, model, top_k=top_k_each)
    bm25_results  = bm25_search(query, chunks, bm25_index, top_k=top_k_each)

    fused_scores: dict[str, float] = {}
    chunk_by_id:  dict[str, dict]  = {}

    for rank, (c, _) in enumerate(dense_results, start=1):
        cid = c["chunk_id"]
        chunk_by_id[cid]  = c
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    for rank, (c, _) in enumerate(bm25_results, start=1):
        cid = c["chunk_id"]
        chunk_by_id[cid]  = c
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(fused_scores.items(), key=lambda kv: -kv[1])[:fused_top_k]
    return [(chunk_by_id[cid], score) for cid, score in ranked]


def build_step3_pipeline(corpus_path: str = CORPUS_PATH):
    print(f"  [step3]    Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    docs    = load_docs(corpus_path)
    chunks  = build_baseline_chunks(docs)
    vectors = build_dense_index(chunks, model)
    bm25_idx = build_bm25_index(chunks)

    def query_fn(query: str, top_k: int):
        results = hybrid_search(
            query, chunks, vectors, bm25_idx, model,
            top_k_each=max(top_k * 3, 10),   
            fused_top_k=top_k,
        )
        return [(c["doc_id"], score) for c, score in results]

    meta = {
        "name": "step3_hybrid",
        "model": EMBED_MODEL,
        "chunking": f"fixed-{CHUNK_SIZE}-chars",
        "retrieval": "dense+BM25 RRF(k=60)",
        "n_chunks": len(chunks),
    }
    return query_fn, meta