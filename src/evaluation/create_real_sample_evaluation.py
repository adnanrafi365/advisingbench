"""
Create evaluation template for real Gemini sample outputs.

This combines:
- real LLM-only sample outputs
- real Basic RAG sample outputs
- real Citation RAG sample outputs

The resulting CSV can be manually scored using the AdvisingBench rubric.
"""

from pathlib import Path
import pandas as pd


OUTPUT_FILES = [
    Path("data/outputs/llm_only_real_sample_outputs.csv"),
    Path("data/outputs/basic_rag_real_sample_outputs.csv"),
    Path("data/outputs/citation_rag_real_sample_outputs.csv"),
]

QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
EVALUATION_FILE = Path("data/scores/real_sample_evaluation_template.csv")

SCORE_COLUMNS = [
    "factual_accuracy",
    "hallucination",
    "groundedness",
    "citation_precision",
    "completeness",
    "uncertainty_handling",
    "error_labels",
    "evaluator_notes",
]


def main() -> None:
    questions = pd.read_csv(QUESTIONS_FILE)

    outputs = []
    for file in OUTPUT_FILES:
        if not file.exists():
            print(f"Skipping missing file: {file}")
            continue

        outputs.append(pd.read_csv(file))

    if not outputs:
        raise FileNotFoundError("No real sample output files found.")

    combined = pd.concat(outputs, ignore_index=True)

    merged = combined.merge(
        questions[["question_id", "gold_answer", "source_id", "source_url", "notes"]],
        on="question_id",
        how="left",
    )

    for col in SCORE_COLUMNS:
        merged[col] = ""

    EVALUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(EVALUATION_FILE, index=False)

    print(f"Saved real sample evaluation template to: {EVALUATION_FILE}")
    print(f"Total rows: {len(merged)}")
    print("\nRows by pipeline:")
    print(merged["pipeline"].value_counts())


if __name__ == "__main__":
    main()
