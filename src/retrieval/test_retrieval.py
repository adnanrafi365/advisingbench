"""
Test FAISS retrieval for AdvisingBench.

This script:
1. Loads the FAISS index
2. Loads chunk metadata
3. Embeds a test question
4. Retrieves the top-k most relevant chunks
5. Prints source IDs, titles, URLs, scores, and text previews
"""

from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/processed/vector_index/advisingbench.faiss")
METADATA_FILE = Path("data/processed/vector_index/chunk_metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_index_and_metadata():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve(question: str, model, index, metadata, top_k: int = TOP_K):
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


def print_results(question: str, results):
    print("=" * 100)
    print(f"Question: {question}")
    print("=" * 100)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source ID: {result['source_id']}")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(f"URL: {result['url']}")
        print("\nText preview:")
        print(result["text"][:900])
        print("-" * 100)


def main():
    test_questions = [
        "Can an F-1 international student take only online classes?",
        "Where can I find BS in Computer Science degree requirements?",
        "How can I check if my transfer course counts at UIC?",
        "Where can I find UIC graduation requirements?",
        "What should I do before withdrawing from the term?",
    ]

    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()

    print(f"Loaded index with {index.ntotal} vectors.")
    print(f"Loaded {len(metadata)} metadata records.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for question in test_questions:
        results = retrieve(question, model, index, metadata, TOP_K)
        print_results(question, results)


if __name__ == "__main__":
    main()
