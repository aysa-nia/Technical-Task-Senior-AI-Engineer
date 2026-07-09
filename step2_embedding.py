import numpy as np
from sentence_transformers import SentenceTransformer
from step1_chunking import load_docs, build_chunks   

CORPUS_PATH   = "corpus.jsonl"
EMBED_MODEL   = "BAAI/bge-large-en-v1.5"

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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
    chunks = build_chunks(docs)       
    vectors = build_index(chunks, model)

    def query_fn(query: str, top_k: int):
        results = retrieve(query, chunks, vectors, model, top_k=top_k)
        return [c["doc_id"] for c, _ in results]

    meta = {
        "name":     "step2_embedding",
        "model":    EMBED_MODEL,
        "chunking": "sentence-aware-1600chars-1sent-overlap",   
        "n_chunks": len(chunks),
    }
    return query_fn, meta



if __name__ == "__main__":
    model   = SentenceTransformer(EMBED_MODEL)
    docs    = load_docs(CORPUS_PATH)
    chunks  = build_chunks(docs)
    vectors = build_index(chunks, model)

    probe = "What is the rated output of the C-100 compressor?"
    results = retrieve(probe, chunks, vectors, model, top_k=3)

    print(f"\nQuery: {probe!r}\n")
    print(f"{'Rank':<5} {'doc_id':<10} {'score':>7}  preview")
    print("-" * 68)
    for rank, (c, score) in enumerate(results, 1):
        preview = c["text"][:70].replace("\n", " ")
        print(f"  {rank:<3} {c['doc_id']:<10} {score:>7.4f}  {preview!r}")

    top_score   = results[0][1]
    second_score = results[1][1] if len(results) > 1 else 0.0
    margin      = top_score - second_score
    top_doc     = results[0][0]["doc_id"]

    print(f"\nTop hit  : {top_doc}  (expected DOC-03)")
    print(f"Margin   : {top_score:.4f} - {second_score:.4f} = {margin:.4f}")
    if top_doc == "DOC-03" and margin > 0.03:
        print("PASS — DOC-03 is top hit with clear margin.")
    elif top_doc == "DOC-03":
        print("PARTIAL — DOC-03 is top hit but margin is narrow.")
    else:
        print("FAIL — DOC-03 is NOT the top hit.")
