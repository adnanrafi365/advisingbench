"""
Build FAISS vector index for AdvisingBench source chunks.

This script:
1. Loads processed source chunks
2. Creates sentence-transformer embeddings
3. Builds a FAISS vector index
4. Saves the index locally
5. Saves chunk metadata locally

Note:
The FAISS index is built from UIC source text chunks and should stay local.
"""

from pathlib import Path
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/source_chunks.csv")
INDEX_DIR = Path("data/processed/vector_index")
FAISS_INDEX_FILE = INDEX_DIR / "advisingbench.faiss"
METADATA_FILE = INDEX_DIR / "chunk_metadata.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {CHUNKS_FILE}. Run chunk_sources.py first."
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_csv(CHUNKS_FILE)

    if chunks.empty:
        raise ValueError("No chunks found. Check source_chunks.csv.")

    texts = chunks["text"].fillna("").tolist()

    print(f"Loaded {len(texts)} chunks.")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_FILE))

    metadata = chunks.to_dict(orient="records")

    with METADATA_FILE.open("wb") as f:
        pickle.dump(metadata, f)

    print("\nFAISS index built successfully.")
    print(f"Index saved to: {FAISS_INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print(f"Total vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
