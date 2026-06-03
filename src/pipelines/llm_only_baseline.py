"""
LLM-only baseline for AdvisingBench.

This baseline answers advising questions without using retrieved UIC source documents.

For safety and reproducibility, this first version creates placeholder outputs.
Later, this script can be connected to OpenAI, Gemini, or another LLM API.
"""

from pathlib import Path
import pandas as pd


QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
OUTPUT_FILE = Path("data/outputs/llm_only_outputs.csv")


def generate_placeholder_answer(question: str) -> str:
    """
    Placeholder baseline answer.

    This will be replaced with an actual LLM API call later.
    """
    return (
        "PLACEHOLDER_LLM_ONLY_ANSWER: This answer was generated without retrieval. "
        "In the final pipeline, this will be replaced by an LLM response."
    )


def main() -> None:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    rows = []

    for _, row in questions.iterrows():
        answer = generate_placeholder_answer(row["question"])

        rows.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "category": row["category"],
            "risk_level": row["risk_level"],
            "answer_type": row["answer_type"],
            "pipeline": "llm_only",
            "model_name": "placeholder",
            "answer": answer,
            "used_retrieval": "no",
            "citations": "",
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print(f"Saved LLM-only baseline outputs to: {OUTPUT_FILE}")
    print(f"Total outputs: {len(rows)}")


if __name__ == "__main__":
    main()
