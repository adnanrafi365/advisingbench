"""
Calculate AdvisingBench evaluation metrics.

This script reads an evaluation CSV and calculates:
1. Average metric scores by pipeline
2. Average metric scores by category
3. Average metric scores by risk level
4. Overall score per answer
5. Error label counts

Example:
python src/evaluation/calculate_metrics.py
python src/evaluation/calculate_metrics.py --input data/scores/evaluation_sample_scored.csv
"""

from pathlib import Path
import argparse
import pandas as pd


DEFAULT_EVALUATION_FILE = Path("data/scores/evaluation_template.csv")
RESULTS_DIR = Path("results")
SUMMARY_BY_PIPELINE_FILE = RESULTS_DIR / "summary_by_pipeline.csv"
SUMMARY_BY_CATEGORY_FILE = RESULTS_DIR / "summary_by_category.csv"
SUMMARY_BY_RISK_FILE = RESULTS_DIR / "summary_by_risk_level.csv"
ERROR_COUNTS_FILE = RESULTS_DIR / "error_label_counts.csv"

SCORE_COLUMNS = [
    "factual_accuracy",
    "hallucination",
    "groundedness",
    "citation_precision",
    "completeness",
    "uncertainty_handling",
]


def split_error_labels(series: pd.Series) -> pd.Series:
    """Split comma-separated error labels and count them."""
    labels = []

    for value in series.dropna():
        value = str(value).strip()

        if not value:
            continue

        for label in value.split(","):
            label = label.strip()
            if label:
                labels.append(label)

    if not labels:
        return pd.Series(dtype="int64")

    return pd.Series(labels).value_counts()


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate AdvisingBench evaluation metrics.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_EVALUATION_FILE),
        help="Path to scored evaluation CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_file = Path(args.input)

    if not evaluation_file.exists():
        raise FileNotFoundError(
            f"Missing evaluation file: {evaluation_file}. "
            "Run create_evaluation_template.py first or provide --input."
        )

    df = pd.read_csv(evaluation_file)

    missing_cols = [col for col in SCORE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing score columns: {missing_cols}")

    for col in SCORE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    scored_df = df.dropna(subset=SCORE_COLUMNS, how="any").copy()

    if scored_df.empty:
        print("No fully scored rows yet.")
        print("Fill in the evaluation score columns first.")
        print("Expected score columns:")
        for col in SCORE_COLUMNS:
            print(f"- {col}")
        return

    scored_df["overall_score"] = scored_df[SCORE_COLUMNS].sum(axis=1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_by_pipeline = scored_df.groupby("pipeline")[SCORE_COLUMNS + ["overall_score"]].mean()
    summary_by_category = scored_df.groupby(["pipeline", "category"])[SCORE_COLUMNS + ["overall_score"]].mean()
    summary_by_risk = scored_df.groupby(["pipeline", "risk_level"])[SCORE_COLUMNS + ["overall_score"]].mean()

    error_counts = split_error_labels(scored_df.get("error_labels", pd.Series(dtype="object")))

    summary_by_pipeline.to_csv(SUMMARY_BY_PIPELINE_FILE)
    summary_by_category.to_csv(SUMMARY_BY_CATEGORY_FILE)
    summary_by_risk.to_csv(SUMMARY_BY_RISK_FILE)
    error_counts.to_csv(ERROR_COUNTS_FILE, header=["count"])

    print(f"Metric calculation complete for: {evaluation_file}")
    print(f"Scored rows: {len(scored_df)}")
    print("\nSummary by pipeline:")
    print(summary_by_pipeline)

    print(f"\nSaved: {SUMMARY_BY_PIPELINE_FILE}")
    print(f"Saved: {SUMMARY_BY_CATEGORY_FILE}")
    print(f"Saved: {SUMMARY_BY_RISK_FILE}")
    print(f"Saved: {ERROR_COUNTS_FILE}")


if __name__ == "__main__":
    main()
