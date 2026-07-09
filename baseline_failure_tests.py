import numpy as np
from sentence_transformers import SentenceTransformer
from baseline_rag import load_docs, chunk_text, build_index, retrieve, answer, CORPUS_PATH

IDENTIFIER_QUERIES = [
    "What does error code E-208 mean?",
    "What does error code E-207 mean?",
    "What does error code E-115 mean?",
    "What does error code E-120 mean?",
    "The display is showing E-208, what should I do?",
    "Unit tripped with E115, what does that indicate?",
    "My equipment shows an overpressure trip code",
    "What causes a dry-run condition error?",
]


def test_contradictory_facts(chunks, vectors, model):
    query = "What is the maximum operating pressure of the P-200 pump?"
    print("Q:", query)
    print("A:", answer(query, chunks, vectors, model))
    print("DOC-01 says 16 bar, DOC-02 says 12 bar -- only one can appear above.")


def test_near_duplicates(chunks, vectors, model):
    query = "What is the vibration limit for the F-30 fan?"
    print("Q:", query)
    print("A:", answer(query, chunks, vectors, model))
    hit, score = retrieve(query, chunks, vectors, model)
    print(f"Top-1 doc_id: {hit['doc_id']}  (score={score:.4f})")
    print("DOC-05 and DOC-06 both describe this limit -- only one is returned.")


def test_multi_document(chunks, vectors, model):
    query = "What are the preventive maintenance intervals for pumps and compressors?"
    print("Q:", query)
    print("A:", answer(query, chunks, vectors, model))
    print("Needs DOC-12 at minimum, arguably DOC-01/DOC-02 -- answer() can only return one document.")


def test_out_of_corpus(chunks, vectors, model):
    for query in ["What color is the C-100 compressor?",
                  "What is the tallest mountain in South America?"]:
        hit, score = retrieve(query, chunks, vectors, model)
        print("Q:", query)
        print("A:", answer(query, chunks, vectors, model))
        print(f"   score={score:.4f}, no threshold check exists in the code")


def test_chunk_boundaries():
    text = (
        "The P-500 valve assembly requires quarterly inspection of the seal "
        "and actuator. Confirm that the position indicator matches the "
        "commanded state before returning the valve to automatic control. "
        "Replace the diaphragm if any sign of cracking or permanent set is "
        "observed during the inspection. Torque the bonnet bolts to the "
        "value specified on the nameplate, following a star pattern. Record "
        "the as-found and as-left positions for every inspection cycle."
    )
    pieces = chunk_text(text)
    for i, p in enumerate(pieces):
        print(f"chunk {i}: {p!r}")


def ranked(query, chunks, vectors, model, top_k=5):
    q = model.encode([query])[0].astype("float32")
    q = q / np.linalg.norm(q)
    sims = vectors @ q
    order = np.argsort(-sims)[:top_k]
    return [(chunks[i]["doc_id"], float(sims[i])) for i in order]


def test_identifier_confusion(chunks, vectors, model):
    print(f"{'Query':50s} {'#1':12s} {'#2':12s} {'margin':8s}")
    for query in IDENTIFIER_QUERIES:
        top5 = ranked(query, chunks, vectors, model)
        doc1, score1 = top5[0]
        doc2, score2 = top5[1]
        margin = score1 - score2
        print(f"{query:50s} {doc1}:{score1:.3f}  {doc2}:{score2:.3f}  {margin:.4f}")


if __name__ == "__main__":
    docs = load_docs(CORPUS_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks, vectors = build_index(docs, model)

    print("Test 1: contradictory facts across documents")
    test_contradictory_facts(chunks, vectors, model)
    print("-" * 60)

    print("Test 2: near-duplicate paraphrases")
    test_near_duplicates(chunks, vectors, model)
    print("-" * 60)

    print("Test 3: multi-document question")
    test_multi_document(chunks, vectors, model)
    print("-" * 60)

    print("Test 4: out-of-corpus questions")
    test_out_of_corpus(chunks, vectors, model)
    print("-" * 60)

    print("Test 5: fixed-window chunking")
    test_chunk_boundaries()
    print("-" * 60)

    print("Test 6: identifier confusion between similar error codes")
    test_identifier_confusion(chunks, vectors, model)
