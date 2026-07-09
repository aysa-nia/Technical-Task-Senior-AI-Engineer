import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from step3_hybrid import (
    load_docs, 
    build_baseline_chunks, 
    build_dense_index, 
    build_bm25_index, 
    hybrid_search, 
    CORPUS_PATH, 
    EMBED_MODEL,
    CHUNK_SIZE
)

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query: str, candidates: list, rerank_model: CrossEncoder, top_n: int = 4):
    if not candidates:
        return []
    
    texts = [c["text"] for c, _ in candidates]
    scores = rerank_model.predict([(query, t) for t in texts])
    
    scored = list(zip([c for c, _ in candidates], map(float, scores)))
    scored.sort(key=lambda cs: -cs[1])
    return scored[:top_n]


def build_step4_pipeline(corpus_path: str = CORPUS_PATH):
    print(f"  [step4]    Loading Dense Embedder: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    
    print(f"  [step4]    Loading Cross-Encoder:  {RERANK_MODEL_NAME}")
    rerank_model = CrossEncoder(RERANK_MODEL_NAME)

    docs    = load_docs(corpus_path)
    chunks  = build_baseline_chunks(docs)
    vectors = build_dense_index(chunks, model)
    bm25_idx = build_bm25_index(chunks)

    def query_fn(query: str, top_k: int):
        candidates = hybrid_search(
            query, chunks, vectors, bm25_idx, model,
            top_k_each=max(top_k * 3, 10),
            fused_top_k=max(top_k * 2, 8),
        )
        reranked_results = rerank(query, candidates, rerank_model, top_n=top_k)
        return [(c["doc_id"], score) for c, score in reranked_results]

    meta = {
        "name": "step4_rerank",
        "model": f"{EMBED_MODEL} + {RERANK_MODEL_NAME}",
        "chunking": f"fixed-{CHUNK_SIZE}-chars",
        "retrieval": "dense+BM25 RRF -> Cross-Encoder",
        "n_chunks": len(chunks),
    }
    return query_fn, meta

