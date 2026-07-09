import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import re
from step1_chunking import load_docs, build_chunks, build_index
from step2_embedding import build_step2_pipeline
from step3_hybrid import build_step3_pipeline

EVAL_SET = [
    ("What is the rated output of the C-100 compressor?",           {"DOC-03"},             True),
    ("What is the nominal flow rate of the P-200 pump?",            {"DOC-01"},             True),
    ("What is the motor power of the C-100 compressor?",            {"DOC-03"},             True),
    ("What is the inlet size of the P-200 pump?",                   {"DOC-01"},             True),
    ("What current draw check happens during M-50 motor startup?",  {"DOC-04"},             True),
    ("What is the maximum operating pressure of the P-200?",        {"DOC-01", "DOC-02"},   True),
    ("P-200 maximum operating pressure",                            {"DOC-01", "DOC-02"},   True),
    ("What is the vibration velocity limit for the F-30 fan?",      {"DOC-05", "DOC-06"},   True),
    ("F-30 vibration limit",                                        {"DOC-05", "DOC-06"},   True),
    ("What does error code E-208 mean?",                            {"DOC-07"},             True),
    ("What does error code E-207 mean?",                            {"DOC-07"},             True),
    ("What does error code E-115 mean?",                            {"DOC-08"},             True),
    ("What does error code E-120 mean?",                            {"DOC-08"},             True),
    ("What causes a dry-run condition error?",                      {"DOC-07"},             True),
    ("BRG-4410 replacement interval",                               {"DOC-09"},             True),
    ("What type of bearing is the BRG-4410?",                       {"DOC-09"},             True),
    ("What should be checked before starting the M-50 motor?",      {"DOC-04"},             True),
    ("What are the steps for lockout/tagout?",                      {"DOC-11"},             True),
    ("What should happen during an emergency shutdown?",            {"DOC-14"},             True),
    ("How do I order spare parts?",                                 {"DOC-15"},             True),
    ("How often should temperature sensors be calibrated?",         {"DOC-10"},             True),
    ("What are the preventive maintenance intervals for pumps and compressors?", {"DOC-12", "DOC-01", "DOC-02"}, True),
    ("What color is the C-100 compressor?",                         set(),                  False),
    ("What is the warranty period for the F-30 fan?",               set(),                  False),
    ("Who is the manufacturer of the M-50 motor?",                  set(),                  False),
    ("How much water can the P-200 transfer per hour?", {"DOC-01"}, True),
    ("How powerful is the C-100 compressor?", {"DOC-03"}, True),
    ("How much air does the compressor deliver?", {"DOC-03"}, True),
    ("How frequently should temperature probes be calibrated?", {"DOC-10"}, True),
    ("When should BRG-4410 be replaced?", {"DOC-09"}, True),
    ("How often should compressors receive preventive maintenance?", {"DOC-12"}, True),
    ("How often should the P-200 pump be serviced?", {"DOC-02","DOC-12"}, True),
    ("Pump intake diameter", {"DOC-01"}, True),
    ("Pump discharge diameter", {"DOC-01"}, True),
    ("Electric power of C-100", {"DOC-03"}, True),
    ("Allowed vibration for F-30", {"DOC-05","DOC-06"}, True),
    ("Emergency stop procedure", {"DOC-14"}, True),
    ("What maintenance should be performed on the P-200?", {"DOC-02","DOC-12"}, True),
    ("What should be documented during maintenance?", {"DOC-02","DOC-12"}, True),
    ("Which equipment is serviced every 2000 hours?", {"DOC-02","DOC-12"}, True),
    ("What pressure limit is specified during maintenance?", {"DOC-02"}, True),
    ("What is the specification pressure of the P-200?", {"DOC-01"}, True),
    ("Which document mentions 12 bar?", {"DOC-02"}, True),
    ("Which document mentions 16 bar?", {"DOC-01"}, True),("My pump is leaking. What should I inspect?", {"DOC-02"}, True),
    ("The controller reports E-120. What happened?", {"DOC-08"}, True),
    ("The fan vibration is too high. What could be wrong?", {"DOC-05","DOC-06"}, True),
    ("The pump shows E-208.", {"DOC-07"}, True),
    ("How do I replace BRG-4410 safely?", {"DOC-09"}, True),
    ("What is the voltage of the M-50 motor?", set(), False),
    ("Who manufactures the P-200?", set(), False),
    ("What is the efficiency of the C-100?", set(), False),
    ("How much does the fan weigh?", set(), False),
    ("What is the serial number of BRG-4410?", set(), False),
]

CORPUS_PATH = "corpus.jsonl"

def build_baseline_pipeline():
    CHUNK_SIZE = 400
    EMBED_MODEL = "all-MiniLM-L6-v2"

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

    def build_index(docs, model):
        chunks = []
        for d in docs:
            for c in chunk_text(d["text"]):
                chunks.append({"doc_id": d["id"], "title": d["title"], "text": c})
        vecs = model.encode([c["text"] for c in chunks])
        vecs = np.asarray(vecs, dtype="float32")
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return chunks, vecs

    def retrieve_fn(query, chunks, vectors, model, top_k):
        q = model.encode([query])[0].astype("float32")
        q = q / np.linalg.norm(q)
        sims = vectors @ q
        order = np.argsort(-sims)[:top_k]
        return [(chunks[i]["doc_id"], float(sims[i])) for i in order]

    print("  [baseline] Loading model: all-MiniLM-L6-v2")
    model = SentenceTransformer(EMBED_MODEL)
    docs = load_docs(CORPUS_PATH)
    chunks, vectors = build_index(docs, model)

    def query_fn(query, top_k):
        return retrieve_fn(query, chunks, vectors, model, top_k)

    return query_fn, {"name": "baseline", "model": EMBED_MODEL, "chunking": "fixed-400-chars", "n_chunks": len(chunks)}


def build_step1_pipeline():
    EMBED_MODEL = "all-MiniLM-L6-v2"
    print("  [step1]    Loading model: all-MiniLM-L6-v2  (unchanged)")
    model = SentenceTransformer(EMBED_MODEL)
    docs = load_docs(CORPUS_PATH)
    chunks = build_chunks(docs)
    vectors = build_index(chunks, model)

    def retrieve_fn(query, top_k):
        q = model.encode([query])[0].astype("float32")
        q = q / np.linalg.norm(q)
        sims = vectors @ q
        order = np.argsort(-sims)[:top_k]
        return [(chunks[i]["doc_id"], float(sims[i])) for i in order]

    return retrieve_fn, {"name": "step1_chunking", "model": EMBED_MODEL,
                        "chunking": f"sentence-aware-1600chars-1sent-overlap",
                        "n_chunks": len(chunks)}


def evaluate(query_fn, k: int = 4):
    recall_hits, mrr_sum = 0, 0.0
    n_answerable = 0
    abstain_correct, abstain_total = 0, 0
    rows = []

    for query, expected_docs, is_answerable in EVAL_SET:
        retrieved_pairs = query_fn(query, top_k=k)
        retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_pairs]

        if is_answerable:
            n_answerable += 1
            hit = bool(expected_docs & set(retrieved_doc_ids))
            recall_hits += int(hit)
            rank_of_first_hit = next(
                (i for i, d in enumerate(retrieved_doc_ids, 1) if d in expected_docs), None
            )
            mrr_sum += (1.0 / rank_of_first_hit) if rank_of_first_hit else 0.0
            rows.append((query, "recall_hit" if hit else "MISS", retrieved_pairs, is_answerable))
        else:
            abstain_total += 1
            abstained = len(retrieved_pairs) == 0
            abstain_correct += int(abstained)
            rows.append((query,
                        "correctly_abstained" if abstained else "false_answer",
                        retrieved_pairs, is_answerable))

    recall_at_k = recall_hits / n_answerable if n_answerable else float("nan")
    mrr        = mrr_sum      / n_answerable if n_answerable else float("nan")
    abstain_acc = abstain_correct / abstain_total if abstain_total else float("nan")
    return rows, recall_at_k, mrr, abstain_acc


def print_report(rows, recall_at_k, mrr, abstain_acc, meta, k):
    name = meta["name"]
    print(f"\n{'=' * 72}")
    print(f"  Pipeline : {name}")
    print(f"  Model    : {meta['model']}")
    print(f"  Chunking : {meta['chunking']}  |  total chunks: {meta['n_chunks']}")
    print(f"{'=' * 72}")
    for query, status, retrieved_pairs, is_answerable in rows:
        flag = "!!" if "MISS" in status or "false_answer" in status else "  "
        
        formatted_retrieved = ", ".join([f"{doc_id}({score:.4f})" for doc_id, score in retrieved_pairs])
        print(f"{flag} [{status:20s}] {query}")
        print(f"     retrieved: [{formatted_retrieved}]")
        
    print(f"\n{'-' * 72}")
    n_ans  = sum(1 for _, _, _, ia in rows if ia)
    n_ooc  = sum(1 for _, _, _, ia in rows if not ia)
    print(f"  Answerable queries   : {n_ans}")
    print(f"  Out-of-corpus queries: {n_ooc}")
    print(f"  Recall@{k:<2d}             : {recall_at_k:.3f}")
    print(f"  MRR                  : {mrr:.3f}")
    print(f"  Abstention accuracy  : {abstain_acc:.3f}  " f"(baseline always 0.000 — no abstention logic yet)")
    print(f"{'-' * 72}")
    return {"name": name, "recall@k": recall_at_k, "mrr": mrr, "abstain_acc": abstain_acc}


def print_delta(base_metrics, step_metrics, k):
    print(f"\n{'*' * 72}")
    print(f"  DELTA: {step_metrics['name']}  vs  {base_metrics['name']}")
    print(f"{'*' * 72}")
    for metric in ("recall@k", "mrr"):
        delta = step_metrics[metric] - base_metrics[metric]
        sign  = "+" if delta >= 0 else ""
        print(f"  {metric:14s}: {base_metrics[metric]:.3f}  →  {step_metrics[metric]:.3f}  " f"({sign}{delta:.3f})")
    print(f"{'*' * 72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="1",
                        help="Which step to evaluate vs baseline: '1', '2', '3' or 'baseline'")
    parser.add_argument("--k", type=int, default=4,
                        help="Recall@k and top-k for retrieval (default: 4)")
    args = parser.parse_args()

    K = args.k

    print(f"\nEvaluating at k={K}...\n")

    print("Building baseline pipeline...")
    base_fn, base_meta = build_baseline_pipeline()
    base_rows, base_r, base_mrr, base_abs = evaluate(base_fn, k=K)
    base_metrics = print_report(base_rows, base_r, base_mrr, base_abs, base_meta, K)

    if args.step == "baseline":
        print("(--step baseline: skipping candidate pipeline)")
    elif args.step == "1":
        print("\nBuilding Step 1 pipeline (sentence-aware chunking)...")
        step_fn, step_meta = build_step1_pipeline()
        step_rows, step_r, step_mrr, step_abs = evaluate(step_fn, k=K)
        step_metrics = print_report(step_rows, step_r, step_mrr, step_abs, step_meta, K)
        print_delta(base_metrics, step_metrics, K)
    elif args.step == "2":
        print("\nBuilding Step 2 pipeline (BGE-large-en-v1.5 embedding)...")
        step_fn, step_meta = build_step2_pipeline()
        step_rows, step_r, step_mrr, step_abs = evaluate(step_fn, k=K)
        step_metrics = print_report(step_rows, step_r, step_mrr, step_abs, step_meta, K)
        print_delta(base_metrics, step_metrics, K)
    elif args.step == "3":
        print("\nBuilding Step 3 pipeline (hybrid: BGE-large dense + BM25 via RRF)...")
        step_fn, step_meta = build_step3_pipeline(corpus_path=CORPUS_PATH)
        # NOTE: If step3 uses Reciprocal Rank Fusion (RRF), its "score" will represent 
        # the fused RRF metric rather than traditional cosine similarity.
        step_rows, step_r, step_mrr, step_abs = evaluate(step_fn, k=K)
        step_metrics = print_report(step_rows, step_r, step_mrr, step_abs, step_meta, K)
        print_delta(base_metrics, step_metrics, K)
    else:
        print(f"Unknown step '{args.step}'. Available: baseline, 1, 2, 3")