"""
Create evaluation template for AdvisingBench model outputs.

This script combines model outputs into one evaluator-friendly CSV.

The evaluator will later fill scores for:
- factual_accuracy
- hallucination
- groundedness
- citation_precision
- completeness
- uncertainty_handling
- error_labels
- evaluator_notes
"""

from pathlib import Path
import pandas as pd


OUTPUT_FILES = [
    Path("data/outputs/llm_only_outputs.csv"),
    Path("data/outputs/basic_rag_outputs.csv"),
    Path("data/outputs/citation_rag_outputs.csv"),
]

QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
EVALUATION_TEMPLATE_FILE = Path("data/scores/evaluation_template.csv")


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
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)

    all_outputs = []

    for file in OUTPUT_FILES:
        if not file.exists():
            print(f"Skipping missing output file: {file}")
            continue

        df = pd.read_csv(file)
        all_outputs.append(df)

    if not all_outputs:
        raise FileNotFoundError("No model output files found. Run pipelines first.")

    outputs = pd.concat(all_outputs, ignore_index=True)

    merged = outputs.merge(
        questions[
            [
                "question_id",
                "gold_answer",
                "source_id",
                "source_url",
                "notes",
            ]
        ],
        on="question_id",
        how="left",
    )

    for col in SCORE_COLUMNS:
        merged[col] = ""

    column_order = [
        "question_id",
        "question",
        "category",
        "risk_level",
        "answer_type",
        "pipeline",
        "model_name",
        "used_retrieval",
        "answer",
        "gold_answer",
        "source_id",
        "source_url",
        "citations",
        "factual_accuracy",
        "hallucination",
        "groundedness",
        "citation_precision",
        "completeness",
        "uncertainty_handling",
        "error_labels",
        "evaluator_notes",
        "notes",
    ]

    existing_columns = [col for col in column_order if col in merged.columns]
    merged = merged[existing_columns]

    EVALUATION_TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(EVALUATION_TEMPLATE_FILE, index=False)

    print(f"Saved evaluation template to: {EVALUATION_TEMPLATE_FILE}")
    print(f"Total rows: {len(merged)}")
    print("\nRows by pipeline:")
    print(merged["pipeline"].value_counts())


if __name__ == "__main__":
    main()
