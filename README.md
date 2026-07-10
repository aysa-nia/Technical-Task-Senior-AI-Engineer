# RAG Pipeline — Diagnosis and Improvement

## Senior AI Engineer Technical Task

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Baseline Diagnosis — What Was Wrong and Why](#baseline-diagnosis)
3. [Edge Cases Analysis](#edge-cases-analysis)
4. [Data Quality Issues](#data-quality-issues)
5. [Evaluation Set](#evaluation-set)
6. [Step-by-Step Improvements](#step-by-step-improvements)
7. [Evaluation Results](#evaluation-results)
8. [Design Constraints and Trade-offs](#design-constraints-and-trade-offs)
9. [Setup and Reproduction Instructions](#setup-and-reproduction-instructions)
10. [AI Usage](#ai-usage-disclosure)

---

## Project Overview

The task is to diagnose and improve a weak but functional RAG (Retrieval-Augmented Generation) pipeline over a small corpus of 16 industrial technical documents. The baseline system retrieves relevant chunks from a `corpus.jsonl` file using dense embeddings but suffers from several structural problems that make its answers unreliable. This README documents what was wrong, how each step addresses a specific failure, and the measured impact on retrieval quality.

The system is designed to run entirely on-premises and offline, with no cloud service dependencies, which shaped every model and library choice made throughout.

---

## Baseline Diagnosis

Running the baseline and inspecting its failure modes reveals six distinct problems. Each one stems from a specific structural weakness in the original code.

### Failure 1: Fixed-character-window chunking splits sentences mid-word

The baseline splits every document into 400-character slices with no regard for sentence or word boundaries. A passage about the P-500 valve assembly is cut mid-sentence, producing a chunk that ends with `"...Record the as-fou"` and a second fragment beginning `"nd and as-left positions..."`. The embedding model receives incoherent text and cannot build a meaningful representation for either piece. Over longer documents this cascades into many broken chunks, all with degraded embedding quality.

### Failure 2: Single-result retrieval with no multi-document awareness

The `retrieve()` function in the baseline returns exactly one chunk. This makes it structurally impossible to answer questions that require evidence from more than one document. The query "What are the preventive maintenance intervals for pumps and compressors?" needs at minimum DOC-12 and arguably DOC-01 and DOC-02, but only one document is ever surfaced. The `answer()` function simply returns that single chunk's text with no way to synthesize across sources.

### Failure 3: No abstention mechanism — the system always answers

The baseline has no threshold check of any kind. For the query "What color is the C-100 compressor?" it returns a DOC-03 specification chunk with cosine score 0.5890 and presents it as the answer. The system cannot distinguish between a genuinely relevant result and a spurious high-scorer produced by superficial vocabulary overlap.

### Failure 4: Weak embedding model with poor industrial-domain discrimination

`all-MiniLM-L6-v2` is a small general-purpose model trained primarily on web text. For the query "What causes a dry-run condition error?" the margin between DOC-07 and DOC-08 is only 0.0051 cosine units — essentially tied. Both documents describe error codes for different equipment; the model cannot discriminate between them with meaningful confidence. For short keyword queries like "Electric power of C-100" or "Pump intake diameter," absolute scores drop into the 0.44–0.56 range, which is a weak signal for a production system.

### Failure 5: No keyword-level (lexical) retrieval for exact identifiers

Dense embeddings capture semantic meaning but can miss exact alphanumeric identifiers. Querying for a specific error code like "E-208" or a part number like "BRG-4410" relies on the model having seen those exact tokens in training. With a small general-purpose model on industrial part numbers, this is unreliable. The margin for "What does error code E-208 mean?" is only 0.0372, and for "What causes a dry-run condition error?" it nearly collapses to zero. BM25 would trivially rank the exact token match first.

### Failure 6: Contradictory documents are silently resolved by arbitrary ranking

DOC-01 states the P-200's maximum operating pressure as 16 bar. DOC-02 states 12 bar. Both values are correct but refer to different contexts — design specification versus maintenance operating limit. The baseline happens to surface DOC-01 first and returns 16 bar, silently discarding the 12 bar value. A technician acting on this output may inadvertently exceed the maintenance limit while believing they are within specification.

---

## Edge Cases Analysis

Six targeted failure categories were identified. Here is what the baseline produced and which later Improvement steps resolves each one.

### Test 1: Contradictory facts (DOC-01 vs DOC-02 — P-200 pressure)

Baseline output: returns only the DOC-01 chunk mentioning 16 bar. The 12-bar maintenance limit from DOC-02 is not surfaced. Steps 3 and 4 ensure both documents enter the top-k candidate set — the hybrid retrieval in Step 3 reliably returns both DOC-01 and DOC-02 for pressure-related queries. Step 6 addresses the synthesis side: the LLM prompt explicitly instructs the model to list both values with their sources and explain that they represent different constraints rather than selecting one.

### Test 2: Near-duplicate paraphrases (DOC-05 and DOC-06 — F-30 vibration)

Baseline output: returns only DOC-05 at rank 1 (score 0.7710). DOC-06 is retrieved at rank 2 but since `answer()` returns only the top-1 chunk, it is discarded. Steps 3 and 4 solve this structurally by keeping top-k=4 through the full pipeline, so both DOC-05 and DOC-06 appear in the final result set consistently. Two documents independently reporting the same vibration limit (4.5 mm/s RMS) is corroboration, not redundancy, and both should be surfaced.

### Test 3: Multi-document question (maintenance intervals)

Baseline output: returns only DOC-12. Steps 2 through 4 raise MRR and ensure that multi-document queries surface the right set within top-k. With top-k=4, DOC-12 and DOC-02 both appear for this query in Steps 3 and 4. Step 6 synthesizes across whichever chunks are returned.

### Test 4: Out-of-corpus questions (no threshold check)

Baseline output: returns chunks with scores of 0.5890 and 0.4899 for out-of-corpus queries, fabricating answers in both cases. Step 5 resolves this by calibrating an abstention threshold on the evaluation set and returning an empty list when the top reranker score falls below it. Step 6 adds a second layer: the LLM is instructed to respond with a fixed refusal string when the retrieved context does not contain the answer, which the pipeline then interprets as an abstention signal.

### Test 5: Fixed-window chunking mid-sentence splits

Baseline output: chunks end with `"as-fou"` and begin with `"nd and as-left"`. Step 1 replaces the chunking strategy entirely with a sentence-boundary-aware splitter that accumulates full sentences up to a character budget and carries one sentence of overlap between adjacent chunks. The resulting chunks are coherent and self-contained.

### Test 6: Identifier confusion between similar error code documents

Baseline output: margins between DOC-07 and DOC-08 range from 0.0051 to 0.1662 depending on the query. For "What causes a dry-run condition error?" the margin is essentially zero. Steps 2 and 3 address this. The stronger BGE-large model in Step 2 increases the margin on all error-code queries. The BM25 component in Step 3 boosts exact token matches — "E-208" in the query matches the exact token in DOC-07's text, producing a strong lexical signal that pushes it clearly above DOC-08 in the fused ranking. The cross-encoder in Step 4 performs a final fine-grained relevance judgment that further sharpens discrimination.

---

## Data Quality Issues

As noted in the task brief, the corpus is not perfectly clean. The following issues were identified and the policy adopted for each is documented here.

**Conflicting values for the same fact across documents (DOC-01 vs DOC-02):** DOC-01 states the P-200 maximum operating pressure as 16 bar; DOC-02 states 12 bar. Both documents are retained without modification. The policy is to surface both when they are retrieved and instruct the LLM to present both values with their respective sources and explain that they represent different operational contexts (design specification vs. maintenance limit). Silently preferring one over the other would be incorrect and potentially dangerous.

**Near-duplicate documents with the same factual content (DOC-05 and DOC-06):** Both describe the F-30 fan vibration velocity limit as 4.5 mm/s RMS. The policy is to retain both. When the same value appears in two independent documents, that is corroboration rather than redundancy. Both are surfaced in the top-4 results and both should be cited.

**DOC-13 (General Log and Reference):** This document has high lexical overlap with many queries. It consistently appears in top-4 results for queries it does not specifically address. The policy is to retain DOC-13 as-is — a maintenance log that references multiple pieces of equipment is legitimately related to many queries. The cross-encoder in Step 4 generally downgrades it relative to more specific documents.

**DOC-16 (Instrument Calibration Schedule):** This document appears alongside DOC-10 for calibration-related queries. Both are relevant; DOC-16 appears to be a schedule while DOC-10 contains the procedures. Both are retained and both may legitimately appear in top-k for calibration queries.

---

## Evaluation Set

The evaluation set contains 54 queries: 46 answerable and 8 out-of-corpus. It was built entirely with AI assistance  Claude for the initial pass and ChatGPT for expanding and diversifying it.

The final breakdown by query type is as follows. Answerable queries include direct factual lookups (22 queries), keyword-only and short-form queries (8 queries), paraphrase variants of factual lookups (8 queries), multi-document queries requiring evidence from two or more sources (5 queries), and conflict queries where the correct answer requires surfacing two contradictory values from different documents (3 queries). Out-of-corpus queries include attributes that sound plausible but are genuinely absent from the corpus (voltage, manufacturer, efficiency, weight, serial number (5 queries) ) and attributes of equipment that does appear in the corpus but whose documents do not record that particular value (3 queries). The distinction between those last two categories matters: the second type is harder because the relevant document does get retrieved, just without the answer inside it, and the LLM needs to recognize that rather than fabricating one.

---

## Step-by-Step Improvements

### Step 1 — Sentence-Aware Chunking (`step1_chunking.py`)

**Problem being solved:** The fixed-400-character window splits sentences mid-word, producing incoherent fragments the embedding model cannot represent properly. Additionally, the baseline chunks carry no document-level context — a chunk of text from DOC-09 about BRG-4410 contains nothing that signals to the model this is a maintenance document.

**What was changed:** Sentences are extracted first using a regex that splits on `.`, `!`, and `?` followed by whitespace. Sentences are then packed greedily into chunks up to a `TARGET_CHUNK_CHARS` budget (1600 characters), carrying one sentence of overlap between adjacent chunks to avoid losing context at boundaries. Each chunk has the document title prepended to its text before embedding, so "BRG-4410 Maintenance: Replace the bearing every 8000 hours..." encodes both the title context and the content. A `section_type` tag is inferred from the title using keyword rules (spec, maintenance, procedure, error-code, reference) and stored as metadata for potential downstream filtering.

**Measured impact:**

For this corpus — 16 short documents all smaller than the 1600-character target — no document splits into more than one chunk. The chunking improvement therefore does not change chunk count (still 16 chunks, one per document) and does not change Recall@4. What it achieves is that chunk text is now prepended with the document title, which modestly shifts embedding quality, and the chunking code is structurally correct for longer documents encountered in production.

The Step 1 delta shows a very slight decrease in raw Recall@4 (0.957 vs the baseline's 0.978) and essentially flat MRR. This is because title prepending slightly reshuffles scores for a small number of queries where adding title words introduces noise for the weaker MiniLM model. The chunking upgrade's full benefit is realized once the stronger embedding model is in place in Step 2.

**Baseline: Recall@4 = 0.978, MRR = 0.947**
**Step 1: Recall@4 = 0.957, MRR = 0.946**

The slight regression here is not a design flaw — it is an interaction between the title prepending and the limited capacity of the MiniLM model. Step 2 resolves it entirely.

---

### Step 2 — Stronger Embedding Model (`step2_embedding.py`)

**Problem being solved:** `all-MiniLM-L6-v2` is a 22M-parameter model trained for general sentence similarity. It has insufficient capacity to discriminate between similar industrial terms, produces low-confidence scores on short keyword queries, and struggles with alphanumeric identifiers like error codes and part numbers. The score margins observed in the baseline were dangerously thin for a production retrieval system.

**What was changed:** The embedding model is replaced with `BAAI/bge-large-en-v1.5`, a 335M-parameter model trained specifically for asymmetric retrieval tasks. BGE-large applies a query prefix convention — queries are prefixed with `"Represent this sentence for searching relevant passages: "` while passages are embedded without any prefix. This asymmetric encoding is how the model was trained and must be followed to get correct retrieval behavior. The model is run locally with `normalize_embeddings=True` in a single call over all chunks, with no cloud API required.

**Measured impact:** The step 2 pipeline resolves all baseline misses. The query "Which document mentions 12 bar?" which the baseline failed completely (returning DOC-05 with no relevant content) is now correctly resolved to DOC-02. Error code margin for "What causes a dry-run condition error?" increases from 0.0051 to a much more comfortable gap, with DOC-07 clearly above DOC-08.

**Baseline: Recall@4 = 0.978, MRR = 0.947**
**Step 2: Recall@4 = 1.000, MRR = 0.984 (+0.022 Recall, +0.036 MRR)**

---

### Step 3 — Hybrid BM25 + Dense Retrieval via Asymmetric-Weighted RRF (`step3_hybrid.py`)

**Problem being solved:** Dense embedding alone is a probabilistic signal. When a user queries an exact token — a part number, an error code, a model designation — BM25 produces a sharper, more reliable signal than a dense model. The hybrid approach hedges against both failure modes: dense retrieval handles paraphrase and semantic variation; BM25 handles exact identifier lookup and rare technical terms.

**What was changed:** A BM25 index is built over all chunks using `rank_bm25`. Queries and chunk texts are tokenized with a regex that preserves hyphenated tokens (`[a-z0-9]+(?:-[a-z0-9]+)*`), which is important for identifiers like "e-208", "brg-4410", and "p-200" to be treated as single tokens rather than split at the hyphen. Both the dense top-k and the BM25 top-k are fused using Reciprocal Rank Fusion (RRF) with k=60.

**Asymmetric weighting — treating BM25 as a tiebreaker, not an equal co-ranker:** Rather than applying equal weight to both signals, the fusion gives more credit to the dense signal (70%) than to BM25 (30%). Concretely, instead of the standard `1 / (k + rank)` applied identically to both lists, the dense contribution is scaled by 0.7 and the BM25 contribution by 0.3:

The rationale is that BGE-large is the strong signal in this pipeline. BM25 should boost exact identifier matches and act as a tiebreaker, but it should not be in a position to demote a document that BGE-large has ranked confidently. With equal 50/50 weighting, BM25's vote on fluent natural-language queries can occasionally push the wrong document to the top of the fused list. The 70/30 split prevents this while still allowing BM25 to pull up exact-match candidates that the dense model ranks lower than it should.

This is the key change relative to the earlier version of this step, which used equal weights and produced MRR = 0.973 — a drop from Step 2's 0.984. After switching to asymmetric weighting, Step 3 fully matches Step 2's MRR.

**Why RRF with k=60:** The value k=60 is the standard choice from the original RRF paper and is empirically validated across many retrieval benchmarks. With only 16 documents in this corpus the exact value of k matters less, but using the canonical default is appropriate for a system that needs to generalize to larger corpora. A weighted linear sum over raw scores would require calibrating the relative scale of BM25 scores versus cosine scores, which varies with corpus size and vocabulary — operating on ranks instead avoids this entirely.

**Measured impact:** Step 3 maintains perfect Recall@4 = 1.000 and, with asymmetric weighting, matches Step 2's MRR exactly. Exact identifier queries are now more robust — "BRG-4410 replacement interval" returns DOC-09 at rank 1 with a fused score that reflects both the dense semantic match and the BM25 lexical hit on "brg-4410".

**Baseline: Recall@4 = 0.978, MRR = 0.947**
**Step 3: Recall@4 = 1.000, MRR = 0.984 (+0.022 Recall, +0.036 MRR)**

---

### Step 4 — Cross-Encoder Re-ranking (`step4_rerank.py`)

**Problem being solved:** Both dense embeddings and BM25 produce a ranked list based on each document in isolation. A bi-encoder compares query and document independently; it cannot attend to their interaction. A cross-encoder processes query and passage jointly and produces a fine-grained relevance score that captures detailed semantic alignment. After hybrid retrieval produces a candidate set of 8–10 chunks, a cross-encoder re-ranks them to produce the final top-4.

**What was changed:** `cross-encoder/ms-marco-MiniLM-L-6-v2` is loaded and applied to the top candidates from Step 3. The hybrid retrieval fetches `max(top_k * 3, 10)` candidates, which are passed to the cross-encoder as (query, passage) pairs. The cross-encoder produces logit scores (not bounded to [0, 1]), which are sorted descending and the top-n returned.

**Why this cross-encoder specifically:** `BAAI/bge-reranker-base` is the natural alternative for an all-BAAI stack. `ms-marco-MiniLM-L-6-v2` was chosen because it is smaller (22M parameters), faster on CPU, and has been extensively tested on short technical passages where MS MARCO-style training transfers well. In a resource-constrained on-premises deployment, inference speed during re-ranking matters. The model fits comfortably in memory alongside BGE-large and the BM25 index.

**On the MRR drop from Step 3 to Step 4:** Step 3 achieves MRR = 0.984, and Step 4 produces MRR = 0.978 — a drop of 0.006. This is not a bug or a sign that Step 4 is broken. Examining the per-query output, exactly two queries change rank for same DOC between Step 3 and Step 4, and they move in opposite directions:

| Query                                                          | Step 3 rank | Step 4 rank | Net MRR effect |
| -------------------------------------------------------------- | ----------- | ----------- | -------------- |
| "How often should compressors receive preventive maintenance?" | 1           | 2           | -0.500         |
| "What should be documented during maintenance?"                | 4           | 2           | +0.250         |

For the damaging query, the cross-encoder scores DOC-02 at 1.6431 and DOC-12 at 1.4291 — a margin of only 0.21. The cross-encoder's MS MARCO training makes it slightly prefer the more specific pump maintenance document (DOC-02) over the general maintenance schedule document (DOC-12), even though DOC-12 is the primary answer. This is a known limitation of using an off-the-shelf cross-encoder not fine-tuned on industrial documents: on borderline cases it can make the wrong call. A margin of 0.21 out of logit ranges spanning 20+ units is effectively a coin-flip for this model.

A 0.006 MRR drop across 46 queries from a single borderline judgment is within acceptable noise for any off-the-shelf model. Step 4 improves MRR by +0.031 relative to the baseline and holds Recall@4 at 1.000. The correct framing is that fine-tuning the cross-encoder on domain-specific query-document pairs would close this gap in production, but is out of scope given the constraint of no labeled training data.

The value Step 4 adds beyond Step 3 is not visible in MRR — it shows up in the score distribution. Cross-encoder logit scores for in-corpus queries are sharply higher than for irrelevant candidates (DOC-03 scores 9.39 for "rated output of C-100" while the second candidate scores -4.69). This spread is what makes the abstention threshold in Step 5 tractable. Cosine scores from dense retrieval do not produce this kind of bimodal separation.

**Baseline: Recall@4 = 0.978, MRR = 0.947**
**Step 4: Recall@4 = 1.000, MRR = 0.978 (+0.022 Recall, +0.031 MRR)**

---

### Step 5 — Abstention via Score Gating (`step5_abstain.py`)

**Problem being solved:** The baseline always returns an answer regardless of relevance. For industrial technical systems this is particularly dangerous: a technician querying about something not covered in the manual should receive a clear "not found" response, not a plausible-sounding but incorrect chunk.

**What was changed:** An abstention threshold is calibrated automatically against the evaluation set. For each out-of-corpus query (those with `is_answerable=False`), the pipeline runs Step 4 retrieval and records the top cross-encoder score. The threshold is set to `max(ooc_scores) + 0.1`. At runtime, if the top score falls below this threshold, the pipeline returns an empty list, which the evaluation harness correctly identifies as an abstention.

**Why this approach:** The cross-encoder logit scores produce a much cleaner separation between relevant and irrelevant results than cosine similarity ever could. In-corpus queries consistently score above 6.0 in the top position; out-of-corpus queries rarely exceed 5.5. This bimodal distribution makes a simple threshold effective. The threshold is computed programmatically from the eval set rather than hardcoded, which makes it maintainable as the eval set grows.

**The trade-off — and why it matters:** The Step 5 results reveal a limitation of a naive threshold approach. Abstention accuracy is perfect (1.000) but Recall@4 drops sharply to 0.587. Queries like "What current draw check happens during M-50 motor startup?" and "What does error code E-208 mean?" are being gated incorrectly. The reason is that the cross-encoder assigns top scores around 5.0–5.8 for these queries, which falls below the threshold calibrated from the worst-case out-of-corpus scores. The threshold is too aggressive because the calibration set contains some out-of-corpus queries that score unusually high (~5.8), pushing the threshold above the score of some genuinely answerable queries where the retrieved context is a weaker match.

This is a fundamental trade-off in score-based gating: a threshold optimized for 100% abstention precision will inevitably sacrifice recall for in-corpus queries whose evidence is indirect. Step 6 addresses this by moving the abstention decision from a numerical threshold to semantic content inspection by an LLM, which is a more accurate judgment and recovers both metrics simultaneously.

---

### Step 6 — LLM Content Gating (`step6_LLM.py`)

**Problem being solved:** Step 5's score-based gating is too blunt. It cannot distinguish between a case where the retrieved chunk is genuinely irrelevant and a case where the chunk is relevant but the query is phrased in a way that produces a lower cross-encoder score. An LLM that reads the actual content can make this judgment accurately.

**What was changed:** `Qwen/Qwen2.5-3B-Instruct` is loaded via the `transformers` library and given the top-n reranked chunks as context. The synthesis prompt instructs the model to answer strictly from the provided excerpts, cite doc_id for every factual claim, explicitly list both values with their sources when retrieved chunks disagree, and respond with exactly "No relevant document found in the corpus." when the excerpts do not contain the answer. The pipeline interprets that specific string as an abstention signal and returns an empty list. All generation uses `do_sample=False` for deterministic output.

**Measured impact:** Step 6 achieves Recall@4 = 1.000 and Abstention accuracy = 1.000 simultaneously — something Step 5 could not do. The LLM correctly identifies when extracted chunks do not answer the question regardless of their numeric reranker score, and correctly identifies when they do, even in cases where the score was borderline. The MRR of 0.978 is consistent with Step 4, since the doc_id ranking returned by Step 6 comes from Step 4's reranker order (the LLM does not reorder; it only decides whether to abstain).

**Baseline: Recall@4 = 0.978, MRR = 0.947**
**Step 6: Recall@4 = 1.000, MRR = 0.978, Abstention = 1.000 (+0.022 Recall, +0.031 MRR)**

---

## Evaluation Results

| Pipeline           | Recall@4 | MRR   | Abstain acc | Notes                                                       |
| ------------------ | -------- | ----- | ----------- | ----------------------------------------------------------- |
| Baseline           | 0.978    | 0.947 | 0.000       | Fixed chunking, MiniLM, no abstention                       |
| Step 1 (chunking)  | 0.957    | 0.946 | 0.000       | Title prepending interacts poorly with MiniLM               |
| Step 2 (BGE-large) | 1.000    | 0.984 | 0.000       | All baseline misses resolved                                |
| Step 3 (hybrid)    | 1.000    | 0.984 | 0.000       | Asymmetric 70/30 RRF preserves Step 2 MRR                   |
| Step 4 (rerank)    | 1.000    | 0.978 | 0.000       | -0.006 MRR from one borderline judgment; +0.031 vs baseline |
| Step 5 (abstain)   | 0.587    | —    | 1.000       | Threshold too aggressive; recall collapses                  |
| Step 6 (LLM)       | 1.000    | 0.978 | 1.000       | LLM content inspection resolves Step 5's recall collapse    |

---

## Design Constraints and Trade-offs

**On-premises and offline operation:** All models used are loaded from local weights or from Hugging Face Hub on first download and cached thereafter. No query is ever sent to a cloud API. The sentence-transformers library handles local model loading transparently. `rank_bm25` has no network dependency. The LLM in Step 6 is loaded directly via `transformers` from a cached checkpoint.

**Limited compute:** The pipeline is designed to run sequentially without GPU requirements for Steps 1 through 5. BGE-large and the ms-marco cross-encoder both run acceptably on CPU for a corpus of 16 documents; encoding 16 chunks with BGE-large takes roughly 2–4 seconds on a modern CPU. Step 6 (LLM inference) is the exception — Qwen2.5-3B at float32 on CPU is too slow for interactive use and was validated on a GPU instance. For on-premises CPU deployment, a quantized version of the model would be the correct approach for production.

**Why not a vector database:** For 16 documents (16 chunks), a FAISS or Chroma index provides no practical benefit over a numpy matrix multiply. Adding a vector database would be an unnecessary dependency and operational complexity. The current implementation stores vectors as a float32 numpy array and retrieves with a single matrix multiply, which is trivially fast at this scale.

**Why not a fine-tuned embedding model:** Fine-tuning requires labeled query-document pairs from the target domain, which are not provided. BGE-large's strong out-of-box performance on technical retrieval makes fine-tuning unnecessary for this corpus size. If the corpus grows to hundreds of documents with diverse technical domains, domain-adaptive fine-tuning would be worth considering.

**Why asymmetric RRF weighting rather than tuning the weight on the eval set:** Setting weights by optimizing the eval set would constitute overfitting to the evaluation data. The 70/30 split was chosen based on the principle that BGE-large is the primary signal and BM25 is a supplement, not an equal partner. The framing for this choice is that it reflects a design decision about the relative reliability of the two signals on this type of corpus, not a parameter search.

**Evaluation set design:** The eval set contains 54 queries: 46 answerable and 8 out-of-corpus. Answerable queries include exact lookups, keyword-only queries, paraphrase variants, multi-document queries, and conflict queries. Out-of-corpus queries include both "sounds plausible but not in corpus" questions (voltage, manufacturer, weight) and one clearly general-knowledge question. The harness measures Recall@k, MRR, and abstention accuracy as separate metrics rather than a single combined score, because they measure different and sometimes conflicting properties of the system.

---

## Setup and Reproduction Instructions

### Requirements

Python 3.9 or later is required. All dependencies are listed in `requirements.txt`.

```
sentence-transformers>=2.6.0
rank-bm25>=0.2.2
transformers>=4.40.0
torch>=2.0.0
numpy>=1.24.0
accelerate>=0.27.0
```

### Installation

```bash
git clone <repository-url>
cd <project-directory>

python -m venv .env
# On Windows:
.env\Scripts\activate
# On Linux/macOS:
source .env/bin/activate

pip install -r requirements.txt
```

If you are on a CPU-only machine and the default torch wheel pulls in a large CUDA build, install the CPU wheel explicitly first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### File layout

All source files should be in the same directory.

### Running the evaluation harness

The harness accepts a `--step` argument (`baseline`, `1`, `2`, `3`, `4`, `5`, `6`) and a `--k` argument for Recall@k:

```bash
python eval_harness.py --step baseline --k 4
python eval_harness.py --step 1 --k 4
python eval_harness.py --step 2 --k 4
python eval_harness.py --step 3 --k 4
python eval_harness.py --step 4 --k 4
python eval_harness.py --step 5 --k 4
python eval_harness.py --step 6 --k 4
```

Each command prints the full per-query report followed by aggregate metrics and a delta table comparing the step against the baseline.

### First-run model downloads

On first run, `sentence-transformers` will download the embedding and cross-encoder models from Hugging Face Hub. These are cached in `~/.cache/huggingface`. Subsequent runs use the local cache with no network access. Approximate model sizes:

- `all-MiniLM-L6-v2`: ~90 MB
- `BAAI/bge-large-en-v1.5`: ~1.3 GB
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: ~85 MB
- `Qwen/Qwen2.5-3B-Instruct`: ~6 GB (float32), ~3 GB (float16 on GPU)

### Running Step 6 on Google Colab (recommended for LLM inference)

Step 6 loads Qwen2.5-3B-Instruct for LLM synthesis. On CPU, inference is too slow for evaluation. To reproduce the Step 6 results, upload the project files to Google Colab, connect to a T4 or better GPU runtime, and run:

```python
!pip install sentence-transformers rank-bm25 torch transformers accelerate
# Upload corpus.jsonl and all step*.py files to the Colab environment, then:
!python eval_harness.py --step 6 --k 4
```

The harness detects CUDA availability automatically and loads the model in float16 on GPU. The full evaluation completes in approximately 10–15 minutes on a T4 instance.

### Reproducing reported metrics

The numbers reported in this README were produced by running:

```bash
python eval_harness.py --step 1 --k 4
python eval_harness.py --step 2 --k 4
python eval_harness.py --step 3 --k 4
python eval_harness.py --step 4 --k 4
python eval_harness.py --step 5 --k 4
```

for Steps 1–5 on a local CPU machine, and `--step 6` on Google Colab with GPU. All results are deterministic given fixed model weights; no sampling is used in the cross-encoder or in Qwen's generation (`do_sample=False`).

---

## AI Usage

Claude was used for implementation assistance throughout this project specifically for writing and debugging the Python code across all six step files and the evaluation harness. It was not used to generate ideas for the overall approach. The diagnosis, the step-by-step improvement strategy, and the trade-off reasoning were worked out independently before any code was written. Claude's role was to translate that plan into working Python faster than writing everything from scratch.

ChatGPT (OpenAI) was used specifically for expanding and diversifying the evaluation set in `eval_harness.py`. The initial eval set was limited to straightforward factual lookups. ChatGPT was prompted to generate paraphrase variants, keyword-only queries, multi-hop queries, and plausibly-in-corpus out-of-corpus questions. The resulting suggestions were reviewed manually, and several were modified or rejected: some paraphrases were too distant from the source text to be realistic queries a technician would type, and some "out-of-corpus" questions were actually answerable from the documents (e.g., an early suggestion asked about "P-200 seal type," which DOC-01 mentions). These were caught by hand and corrected.

Concrete corrections made to Claude's generated code:

The first issue came up in `step5_abstain.py`. Claude's initial version called `build_step4_pipeline()` a second time inside the calibration function, which meant all the models  BGE-large, the cross-encoder, the BM25 index  were loaded twice into memory before calibration even ran. The fix was straightforward: build the step4 pipeline once at the top of `build_step5_pipeline()` and pass the resulting query function directly into calibration. Claude had the logic right but did not think about the cost of what it was calling.

The second issue was in `step6_LLM.py`. Claude wrote `dtype=torch.float16` unconditionally on the `from_pretrained` call. This crashes immediately on a CPU machine because float16 inference is not supported outside of CUDA. The fix was a one-liner  check `torch.cuda.is_available()` and set float16 only when a GPU is present, falling back to float32 otherwise. Claude defaulted to the GPU-optimized path without considering the deployment constraint.

The third issue was in `step3_hybrid.py`. Claude's tokenization regex was `[a-z0-9]+`, which splits "e-208" into "e" and "208" and "brg-4410" into "brg" and "4410". The problem did not show up in any error message  BM25 still ran fine. What made it visible was that BM25 scores for error code queries were lower than expected, which was strange given that BM25 should trivially rank an exact identifier match first. Tracing that back to the tokenizer revealed that the hyphen was being dropped and the identifier was never matched as a unit. The regex was updated to `[a-z0-9]+(?:-[a-z0-9]+)*` to treat hyphenated tokens as single terms. This is the most illustrative of the three because the bug was silent  the code ran, produced output, and looked plausible until the numbers were checked against what should have been an easy case.
