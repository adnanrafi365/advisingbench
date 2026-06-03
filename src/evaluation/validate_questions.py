"""
Validate AdvisingBench question dataset.

This script checks:
1. Required columns exist
2. No missing values in important columns
3. Question IDs are unique
4. Source IDs exist in the source manifest
5. Risk levels are valid
6. Answer types are valid
"""

from pathlib import Path
import pandas as pd


QUESTIONS_FILE = Path("data/questions/advising_questions_v1.csv")
SOURCE_MANIFEST_FILE = Path("data/raw/source_manifest.csv")

REQUIRED_COLUMNS = [
    "question_id",
    "question",
    "category",
    "gold_answer",
    "source_id",
    "source_url",
    "risk_level",
    "answer_type",
    "notes",
]

VALID_RISK_LEVELS = {"Low", "Medium", "High"}
VALID_ANSWER_TYPES = {"Direct", "Conditional", "Uncertain"}


def main() -> None:
    errors = []

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_FILE}")

    if not SOURCE_MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Missing source manifest file: {SOURCE_MANIFEST_FILE}")

    questions = pd.read_csv(QUESTIONS_FILE)
    sources = pd.read_csv(SOURCE_MANIFEST_FILE)

    print(f"Loaded {len(questions)} questions.")
    print(f"Loaded {len(sources)} sources.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in questions.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    for col in REQUIRED_COLUMNS:
        if col in questions.columns:
            missing_count = questions[col].isna().sum()
            if missing_count > 0:
                errors.append(f"Column '{col}' has {missing_count} missing values.")

    if "question_id" in questions.columns:
        duplicate_ids = questions[questions["question_id"].duplicated()]["question_id"].tolist()
        if duplicate_ids:
            errors.append(f"Duplicate question IDs: {duplicate_ids}")

    if "source_id" in questions.columns:
        valid_source_ids = set(sources["source_id"].dropna())
        question_source_ids = set(questions["source_id"].dropna())
        invalid_source_ids = sorted(question_source_ids - valid_source_ids)

        if invalid_source_ids:
            errors.append(f"Invalid source IDs not found in manifest: {invalid_source_ids}")

    if "risk_level" in questions.columns:
        invalid_risk_levels = sorted(set(questions["risk_level"].dropna()) - VALID_RISK_LEVELS)
        if invalid_risk_levels:
            errors.append(f"Invalid risk levels: {invalid_risk_levels}")

    if "answer_type" in questions.columns:
        invalid_answer_types = sorted(set(questions["answer_type"].dropna()) - VALID_ANSWER_TYPES)
        if invalid_answer_types:
            errors.append(f"Invalid answer types: {invalid_answer_types}")

    print("\nDataset summary:")
    print("\nCategories:")
    print(questions["category"].value_counts())

    print("\nRisk levels:")
    print(questions["risk_level"].value_counts())

    print("\nAnswer types:")
    print(questions["answer_type"].value_counts())

    if errors:
        print("\nValidation failed.")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nValidation passed. Dataset looks good.")


if __name__ == "__main__":
    main()
