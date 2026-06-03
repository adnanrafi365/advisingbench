"""
Run real Gemini LLM-only answers for a small sample of AdvisingBench questions.

This script:
1. Loads the first 3 advising questions
2. Sends each question to Gemini without retrieval
3. Saves real LLM-only outputs locally

Outputs are kept local and ignored by Git.
"""

from pathlib import Path
import time
import pandas as pd

from src.pipelines.llm_client import generate_text, DEFAULT_MODEL


QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
OUTPUT_FILE = Path("data/outputs/llm_only_real_sample_outputs.csv")
NUM_QUESTIONS = 3
REQUEST_DELAY_SECONDS = 2


def build_prompt(question: str) -> str:
    return f"""
You are answering a university academic advising question.

Important rules:
- Answer clearly and concisely.
- If the question depends on student-specific details, say the student should verify with the official university office or advisor.
- Do not invent policy details.
- Do not cite sources because this is the LLM-only baseline and no source documents are provided.

Question:
{question}

Answer:
""".strip()


def main() -> None:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE).head(NUM_QUESTIONS)
    rows = []

    print(f"Running Gemini LLM-only sample on {len(questions)} questions.")
    print(f"Model: {DEFAULT_MODEL}")

    for _, row in questions.iterrows():
        question_id = row["question_id"]
        question = row["question"]

        print(f"\nProcessing {question_id}: {question}")

        prompt = build_prompt(question)
        answer = generate_text(prompt)

        rows.append({
            "question_id": question_id,
            "question": question,
            "category": row["category"],
            "risk_level": row["risk_level"],
            "answer_type": row["answer_type"],
            "pipeline": "llm_only",
            "model_name": DEFAULT_MODEL,
            "answer": answer,
            "used_retrieval": "no",
            "citations": "",
        })

        print("Answer preview:")
        print(answer[:500])

        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved real LLM-only sample outputs to: {OUTPUT_FILE}")
    print(f"Total outputs: {len(rows)}")


if __name__ == "__main__":
    main()
