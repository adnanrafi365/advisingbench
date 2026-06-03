"""
Basic RAG pipeline for AdvisingBench.

This pipeline:
1. Loads advising questions
2. Loads FAISS index and chunk metadata
3. Retrieves top-k source chunks for each question
4. Creates placeholder RAG answers
5. Saves outputs to data/outputs/basic_rag_outputs.csv

The placeholder answer will later be replaced with an actual LLM response.
"""

from pathlib import Path
import pickle

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
INDEX_FILE = Path("data/processed/vector_index/advisingbench.faiss")
METADATA_FILE = Path("data/processed/vector_index/chunk_metadata.pkl")
OUTPUT_FILE = Path("data/outputs/basic_rag_outputs.csv")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_index_and_metadata():
    """Load FAISS index and metadata."""
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


def make_placeholder_rag_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Create a placeholder RAG answer."""
    if not retrieved_chunks:
        return (
            "PLACEHOLDER_BASIC_RAG_ANSWER: No source chunks were retrieved. "
            "The final pipeline should answer cautiously."
        )

    top_source = retrieved_chunks[0]

    return (
        "PLACEHOLDER_BASIC_RAG_ANSWER: This answer will be generated using "
        f"retrieved UIC source context. Top retrieved source: {top_source['source_id']} "
        f"({top_source['title']})."
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
        retrieved = retrieve_chunks(question, model, index, metadata, TOP_K)

        citations = "; ".join(
            [
                f"{item['source_id']}|{item['chunk_id']}|{item['url']}|score={item['score']:.4f}"
                for item in retrieved
            ]
        )

        retrieved_text_preview = "\n\n".join(
            [
                f"[{item['chunk_id']}] {item['text'][:500]}"
                for item in retrieved
            ]
        )

        answer = make_placeholder_rag_answer(question, retrieved)

        rows.append({
            "question_id": row["question_id"],
            "question": question,
            "category": row["category"],
            "risk_level": row["risk_level"],
            "answer_type": row["answer_type"],
            "pipeline": "basic_rag",
            "model_name": "placeholder",
            "answer": answer,
            "used_retrieval": "yes",
            "top_k": TOP_K,
            "citations": citations,
            "retrieved_text_preview": retrieved_text_preview,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved Basic RAG outputs to: {OUTPUT_FILE}")
    print(f"Total outputs: {len(rows)}")


if __name__ == "__main__":
    main()
