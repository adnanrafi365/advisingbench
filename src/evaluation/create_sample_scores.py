"""
Create a small sample scored evaluation file for testing metric calculation.

This does not represent final evaluation.
It only fills example scores for a few rows so we can test the metric script.
"""

from pathlib import Path
import pandas as pd


TEMPLATE_FILE = Path("data/scores/evaluation_template.csv")
SAMPLE_FILE = Path("data/scores/evaluation_sample_scored.csv")


def main() -> None:
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing evaluation template: {TEMPLATE_FILE}. "
            "Run create_evaluation_template.py first."
        )

    df = pd.read_csv(TEMPLATE_FILE)

    # Select 3 rows from each pipeline for a balanced sample.
    sample = (
        df.groupby("pipeline", group_keys=False)
        .head(3)
        .copy()
    )

    score_columns = [
        "factual_accuracy",
        "hallucination",
        "groundedness",
        "citation_precision",
        "completeness",
        "uncertainty_handling",
    ]

    for col in score_columns:
        sample[col] = 0.0

    sample["error_labels"] = ""
    sample["evaluator_notes"] = ""

    for index, row in sample.iterrows():
        pipeline = row["pipeline"]

        if pipeline == "llm_only":
            sample.loc[index, "factual_accuracy"] = 0.5
            sample.loc[index, "hallucination"] = 0.5
            sample.loc[index, "groundedness"] = 0
            sample.loc[index, "citation_precision"] = 0
            sample.loc[index, "completeness"] = 0.5
            sample.loc[index, "uncertainty_handling"] = 0.5
            sample.loc[index, "error_labels"] = "unsupported claim, missing citation"

        elif pipeline == "basic_rag":
            sample.loc[index, "factual_accuracy"] = 0.75
            sample.loc[index, "hallucination"] = 0.75
            sample.loc[index, "groundedness"] = 0.75
            sample.loc[index, "citation_precision"] = 0.5
            sample.loc[index, "completeness"] = 0.75
            sample.loc[index, "uncertainty_handling"] = 0.75
            sample.loc[index, "error_labels"] = "partial citation"

        elif pipeline == "citation_rag":
            sample.loc[index, "factual_accuracy"] = 1
            sample.loc[index, "hallucination"] = 1
            sample.loc[index, "groundedness"] = 1
            sample.loc[index, "citation_precision"] = 1
            sample.loc[index, "completeness"] = 0.75
            sample.loc[index, "uncertainty_handling"] = 1
            sample.loc[index, "error_labels"] = ""

        sample.loc[index, "evaluator_notes"] = "Sample score for testing only."

    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_FILE, index=False)

    print(f"Saved sample scored file to: {SAMPLE_FILE}")
    print(f"Rows: {len(sample)}")
    print(sample[["question_id", "pipeline", "factual_accuracy", "groundedness"]])
    print("\nRows by pipeline:")
    print(sample["pipeline"].value_counts())


if __name__ == "__main__":
    main()
