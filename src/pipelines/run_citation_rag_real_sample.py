"""
Run real Gemini Citation-Grounded RAG answers for a small sample of AdvisingBench questions.

This script:
1. Loads the first 3 advising questions
2. Retrieves top-k UIC source chunks from FAISS
3. Sends question + retrieved context to Gemini
4. Requires source-ID citations in the answer
5. Saves real citation-grounded RAG outputs locally
6. Handles Gemini errors without crashing

Outputs are kept local and ignored by Git.
"""

from pathlib import Path
import pickle
import sys
import time

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.llm_client import generate_text, DEFAULT_MODEL


QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
INDEX_FILE = Path("data/processed/vector_index/advisingbench.faiss")
METADATA_FILE = Path("data/processed/vector_index/chunk_metadata.pkl")
OUTPUT_FILE = Path("data/outputs/citation_rag_real_sample_outputs.csv")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_QUESTIONS = 3
TOP_K = 5
REQUEST_DELAY_SECONDS = 5


def load_index_and_metadata():
    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve_chunks(question: str, model, index, metadata, top_k: int = TOP_K):
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


def build_context(retrieved_chunks: list[dict]) -> str:
    context_blocks = []

    for item in retrieved_chunks:
        context_blocks.append(
            f"Source ID: {item['source_id']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Title: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"Text: {item['text'][:1200]}"
        )

    return "\n\n---\n\n".join(context_blocks)


def build_prompt(question: str, risk_level: str, context: str) -> str:
    return f"""
You are answering a university academic advising question using retrieved official UIC source text.

Strict rules:
- Use ONLY the retrieved UIC source context below.
- Do NOT invent policy details.
- Every factual advising claim must include a source ID citation like [S001].
- Do not cite chunk IDs in the final answer; cite source IDs only.
- If the retrieved context is not enough, say the information is not clear from the retrieved sources.
- If the question depends on a student's specific situation, tell the student to verify with the official UIC office or advisor.
- If the risk level is High, include a clear caution that the student should confirm with the appropriate official office before acting.
- Keep the answer concise and student-friendly.

Question risk level:
{risk_level}

Retrieved UIC source context:
{context}

Question:
{question}

Citation-grounded answer:
""".strip()


def make_citation_string(retrieved_chunks: list[dict]) -> str:
    return "; ".join(
        [
            f"{item['source_id']}|{item['chunk_id']}|{item['url']}|score={item['score']:.4f}"
            for item in retrieved_chunks
        ]
    )


def main() -> None:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_FILE}")

    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("Missing FAISS index or metadata. Build the index first.")

    questions = pd.read_csv(QUESTIONS_FILE).head(NUM_QUESTIONS)
    rows = []

    print(f"Running Gemini Citation-Grounded RAG sample on {len(questions)} questions.")
    print(f"Model: {DEFAULT_MODEL}")

    print("Loading retrieval system...")
    index, metadata = load_index_and_metadata()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for _, row in questions.iterrows():
        question_id = row["question_id"]
        question = row["question"]
        risk_level = row["risk_level"]

        print(f"\nProcessing {question_id}: {question}")

        retrieved = retrieve_chunks(question, embedding_model, index, metadata, TOP_K)
        context = build_context(retrieved)
        prompt = build_prompt(question, risk_level, context)

        try:
            answer = generate_text(prompt)
            status = "success"
        except Exception as error:
            answer = f"ERROR: {type(error).__name__}: {error}"
            status = "failed"
            print("Gemini call failed, saving error and continuing.")

        citations = make_citation_string(retrieved)
        retrieved_source_ids = ",".join(sorted({item["source_id"] for item in retrieved}))
        retrieved_chunk_ids = ",".join([item["chunk_id"] for item in retrieved])

        rows.append({
            "question_id": question_id,
            "question": question,
            "category": row["category"],
            "risk_level": risk_level,
            "answer_type": row["answer_type"],
            "pipeline": "citation_rag",
            "model_name": DEFAULT_MODEL,
            "answer": answer,
            "used_retrieval": "yes",
            "requires_citation": "yes",
            "top_k": TOP_K,
            "retrieved_source_ids": retrieved_source_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "citations": citations,
            "status": status,
        })

        print("Retrieved sources:", retrieved_source_ids)
        print("Answer preview:")
        print(answer[:500])

        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved real Citation RAG sample outputs to: {OUTPUT_FILE}")
    print(f"Total outputs: {len(rows)}")


if __name__ == "__main__":
    main()
