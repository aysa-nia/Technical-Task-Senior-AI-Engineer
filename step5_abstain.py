# step5_abstain.py
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from step4_rerank import build_step4_pipeline, RERANK_MODEL_NAME
from step3_hybrid import EMBED_MODEL, CHUNK_SIZE

def calibrate_threshold_automatically(step4_query_fn, eval_set):
    ooc_scores = []
    for query, _, is_answerable in eval_set:
        if not is_answerable:
            results = step4_query_fn(query, top_k=1)
            if results:
                top_score = results[0][1]  
                ooc_scores.append(top_score)
                
    if not ooc_scores:
        return 3.5000
    calibrated_val = max(ooc_scores) + 0.1
    return round(calibrated_val, 4)


def build_step5_pipeline(corpus_path: str = "corpus.jsonl", eval_set: list = None):
    step4_query_fn, step4_meta = build_step4_pipeline(corpus_path)
    if eval_set:
        print("  [step5]    Running automated threshold calibration on evaluation set...")
        threshold = calibrate_threshold_automatically(step4_query_fn, eval_set)
        print(f"  [step5]    Dynamic threshold calibrated at: {threshold}")
    else:
        threshold = 3.5000
        print(f"  [step5]    No eval set provided. Using default threshold: {threshold}")
    
    def query_fn(query: str, top_k: int):
        reranked_results = step4_query_fn(query, top_k=top_k)
        
        if not reranked_results:
            return []
            
        top_score = reranked_results[0][1]
        if top_score < threshold:
            return []  
            
        return reranked_results

    meta = {
        "name": "step5_abstain_automated",
        "model": f"{EMBED_MODEL} + {RERANK_MODEL_NAME}",
        "chunking": f"fixed-{CHUNK_SIZE}-chars",
        "retrieval": f"dense+BM25 RRF -> Cross-Encoder (Auto-Gated @ score >= {threshold})",
        "n_chunks": step4_meta["n_chunks"],
    }
    return query_fn, meta