import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from step4_rerank import build_step4_pipeline, RERANK_MODEL_NAME
from step3_hybrid import (
    load_docs,
    build_baseline_chunks,
    build_dense_index,
    build_bm25_index,
    hybrid_search,
    EMBED_MODEL,
    CHUNK_SIZE,
)
from sentence_transformers import SentenceTransformer, CrossEncoder

CORPUS_PATH = "corpus.jsonl"

LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" 

SYNTHESIS_SYSTEM_PROMPT = """You are a precise technical documentation assistant.
Your task is to answer the user's question STRICTLY based on the provided source excerpts.

Rules:
1. Grounding: Answer ONLY using facts directly mentioned in the excerpts. Do NOT assume, extrapolate, or use outside knowledge.
2. Citations: For every single factual claim, measurement, or rule you extract, append its source [doc_id] (e.g., [DOC-01]).
3. Contradictions/Disagreements: If different retrieved chunks present conflicting or different values for the same variable, you MUST explicitly list both values along with their respective sources and explain the distinct contexts/constraints they represent. Do not combine, average, or choose between them.
4. Out-of-Corpus/Abstention: If the provided excerpts do not contain the specific information required to answer the question, or if they are irrelevant to the topic, your response must be exactly: "No relevant document found in the corpus." Do not provide any additional explanations, descriptions, or apologies.
"""


def _build_rich_reranker(corpus_path: str):
    print(f"  [step4]    Loading Dense Embedder: {EMBED_MODEL}")
    dense_model = SentenceTransformer(EMBED_MODEL)

    print(f"  [step4]    Loading Cross-Encoder:  {RERANK_MODEL_NAME}")
    rerank_model = CrossEncoder(RERANK_MODEL_NAME)

    docs = load_docs(corpus_path)
    chunks = build_baseline_chunks(docs)
    vectors = build_dense_index(chunks, dense_model)
    bm25_idx = build_bm25_index(chunks)

    def rich_query_fn(query: str, top_k: int):
        candidates = hybrid_search(
            query, chunks, vectors, bm25_idx, dense_model,
            top_k_each=max(top_k * 3, 10),
            fused_top_k=max(top_k * 2, 8),
        )
        if not candidates:
            return []

        texts = [c["text"] for c, _ in candidates]
        scores = rerank_model.predict([(query, t) for t in texts])
        scored = list(zip([c for c, _ in candidates], map(float, scores)))
        scored.sort(key=lambda cs: -cs[1])
        return scored[:top_k]

    return rich_query_fn, len(chunks)


def build_step6_pipeline(corpus_path: str = CORPUS_PATH):
    rich_query_fn, n_chunks = _build_rich_reranker(corpus_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [step6] Loading model target onto device: {device.upper()}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME, 
        local_files_only=False
    )
    
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=model_dtype, 
        device_map=device,
        local_files_only=False
    )

    def pipeline_fn(query: str, top_k: int = 4):
        reranked = rich_query_fn(query, top_k=top_k)

        if not reranked:
            return []

        context_blocks = []
        for chunk, score in reranked:
            doc_id = chunk["doc_id"]
            text   = chunk["text"]         
            context_blocks.append(f"Source Excerpt [{doc_id}]:\n{text}")

        context_str = "\n\n".join(context_blocks)

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Source excerpts:\n{context_str}\n\nQuestion: {query}",
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = llm_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        new_tokens = [
            output[len(inputs.input_ids[0]):] for output in generated_ids
        ]
        response_text = tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        if "No relevant document found in the corpus" in response_text:
            return []

        return [(chunk["doc_id"], float(score)) for chunk, score in reranked]

    meta = {
        "name":      "step6_llm_content_gated",
        "model":     f"{EMBED_MODEL} + {RERANK_MODEL_NAME} + {LLM_MODEL_NAME}",
        "chunking":  f"fixed-{CHUNK_SIZE}-chars",
        "retrieval": "Hybrid + Reranking -> LLM Content Gating",
        "n_chunks":  n_chunks,
    }

    return pipeline_fn, meta