"""
Create initial manual-style scores for the real Gemini sample outputs.

These scores are starter scores for testing the evaluation flow.
They should be reviewed and adjusted carefully before being treated as final.
"""

from pathlib import Path
import pandas as pd


TEMPLATE_FILE = Path("data/scores/real_sample_evaluation_template.csv")
SCORED_FILE = Path("data/scores/real_sample_scored.csv")


def score_row(row):
    pipeline = row["pipeline"]
    answer = str(row["answer"])

    # Starter scoring assumptions:
    # LLM-only can be useful but is not grounded or cited.
    # Basic RAG is grounded but citation behavior may be less strict.
    # Citation RAG should be strongest if it cites source IDs properly.

    if pipeline == "llm_only":
        return {
            "factual_accuracy": 0.5,
            "hallucination": 0.5,
            "groundedness": 0.0,
            "citation_precision": 0.0,
            "completeness": 0.5,
            "uncertainty_handling": 0.5,
            "error_labels": "missing citation, unsupported claim",
            "evaluator_notes": "Starter score: LLM-only answer has no retrieved source grounding.",
        }

    if pipeline == "basic_rag":
        return {
            "factual_accuracy": 0.75,
            "hallucination": 0.75,
            "groundedness": 0.75,
            "citation_precision": 0.5,
            "completeness": 0.75,
            "uncertainty_handling": 0.75,
            "error_labels": "partial citation",
            "evaluator_notes": "Starter score: Basic RAG used retrieved context but citation strictness is weaker.",
        }

    if pipeline == "citation_rag":
        has_source_citation = "[" in answer and "]" in answer

        return {
            "factual_accuracy": 1.0,
            "hallucination": 1.0,
            "groundedness": 1.0,
            "citation_precision": 1.0 if has_source_citation else 0.5,
            "completeness": 0.75,
            "uncertainty_handling": 1.0,
            "error_labels": "" if has_source_citation else "missing citation",
            "evaluator_notes": "Starter score: Citation RAG used retrieved context and cited source IDs.",
        }

    return {
        "factual_accuracy": "",
        "hallucination": "",
        "groundedness": "",
        "citation_precision": "",
        "completeness": "",
        "uncertainty_handling": "",
        "error_labels": "unknown pipeline",
        "evaluator_notes": "Pipeline was not recognized.",
    }


def main() -> None:
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing template file: {TEMPLATE_FILE}. "
            "Run create_real_sample_evaluation.py first."
        )

    df = pd.read_csv(TEMPLATE_FILE)

    for index, row in df.iterrows():
        scores = score_row(row)
        for key, value in scores.items():
            df.loc[index, key] = value

    SCORED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORED_FILE, index=False)

    print(f"Saved scored real sample file to: {SCORED_FILE}")
    print(f"Total rows: {len(df)}")
    print("\nRows by pipeline:")
    print(df["pipeline"].value_counts())


if __name__ == "__main__":
    main()
