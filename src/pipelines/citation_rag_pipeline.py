"""
Citation-grounded RAG pipeline for AdvisingBench.

This pipeline:
1. Loads advising questions
2. Retrieves top-k chunks from the FAISS index
3. Creates citation-grounded placeholder answers
4. Adds high-risk safety warnings
5. Saves outputs to data/outputs/citation_rag_outputs.csv

Later, this placeholder answer will be replaced with an LLM call that must answer
only from retrieved UIC sources and cite source IDs.
"""

from pathlib import Path
import pickle

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
INDEX_FILE = Path("data/processed/vector_index/advisingbench.faiss")
METADATA_FILE = Path("data/processed/vector_index/chunk_metadata.pkl")
OUTPUT_FILE = Path("data/outputs/citation_rag_outputs.csv")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_index_and_metadata():
    """Load FAISS index and chunk metadata."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve_chunks(question: str, model, index, metadata, top_k: int = TOP_K):
    """Retrieve top-k chunks for a question."""
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def make_citation_string(retrieved_chunks: list[dict]) -> str:
    """Create compact citation string from retrieved chunks."""
    return "; ".join(
        [
            f"{item['source_id']}|{item['chunk_id']}|{item['url']}|score={item['score']:.4f}"
            for item in retrieved_chunks
        ]
    )


def make_grounded_placeholder_answer(
    question: str,
    risk_level: str,
    retrieved_chunks: list[dict],
) -> str:
    """Create placeholder answer that references retrieved sources."""
    if not retrieved_chunks:
        return (
            "PLACEHOLDER_CITATION_RAG_ANSWER: I could not retrieve enough official "
            "UIC source context. The final answer should recommend checking with the "
            "appropriate official UIC office."
        )

    top_sources = []
    seen = set()

    for chunk in retrieved_chunks:
        source_id = chunk["source_id"]
        if source_id not in seen:
            seen.add(source_id)
            top_sources.append(source_id)

    source_list = ", ".join(top_sources)

    warning = ""
    if risk_level == "High":
        warning = (
            " Because this is a high-risk advising question, the final answer should "
            "tell the student to verify with the appropriate official UIC office."
        )

    return (
        "PLACEHOLDER_CITATION_RAG_ANSWER: The final answer should be based only on "
        f"retrieved official UIC sources: {source_list}.{warning}"
    )


def main() -> None:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_FILE}")

    print("Loading questions...")
    questions = pd.read_csv(QUESTIONS_FILE)

    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()

    print(f"Loaded index with {index.ntotal} vectors.")
    print(f"Loaded {len(metadata)} metadata records.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    rows = []

    for _, row in questions.iterrows():
        question = row["question"]
        risk_level = row["risk_level"]

        retrieved = retrieve_chunks(question, model, index, metadata, TOP_K)
        citations = make_citation_string(retrieved)

        retrieved_source_ids = sorted({item["source_id"] for item in retrieved})
        retrieved_chunk_ids = [item["chunk_id"] for item in retrieved]

        answer = make_grounded_placeholder_answer(question, risk_level, retrieved)

        rows.append({
            "question_id": row["question_id"],
            "question": question,
            "category": row["category"],
            "risk_level": risk_level,
            "answer_type": row["answer_type"],
            "pipeline": "citation_rag",
            "model_name": "placeholder",
            "answer": answer,
            "used_retrieval": "yes",
            "requires_citation": "yes",
            "top_k": TOP_K,
            "retrieved_source_ids": ",".join(retrieved_source_ids),
            "retrieved_chunk_ids": ",".join(retrieved_chunk_ids),
            "citations": citations,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved citation-grounded RAG outputs to: {OUTPUT_FILE}")
    print(f"Total outputs: {len(rows)}")


if __name__ == "__main__":
    main()
