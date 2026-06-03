# AdvisingBench

**AdvisingBench** is a research-style benchmark for evaluating whether large language models can accurately answer undergraduate academic advising questions using official university documents.

## Full Title

**AdvisingBench: A Groundedness Benchmark for Evaluating LLM Hallucinations in Academic Advising**

## Project Goal

This project evaluates hallucination, groundedness, citation precision, completeness, and uncertainty handling in LLM-generated academic advising answers.

The main question is:

> How accurately can large language models answer undergraduate academic advising questions when grounded in official university documents, and where do they still hallucinate or misuse citations?

## Why This Matters

Academic advising questions can affect:

- Course registration
- Transfer credit planning
- Graduation progress
- Academic standing
- Petitions
- Withdrawal decisions
- International student compliance

If an AI system gives incorrect or overconfident advice, it can negatively affect a student’s academic progress or immigration status.

## Current Version

This repository currently includes:

- 20 public official UIC source pages
- 30 starter academic advising questions
- Gold/reference answers
- Source IDs and official source URLs
- Risk level labels
- Answer type labels
- Source quality analysis
- FAISS-based retrieval system
- LLM-only baseline foundation
- Basic RAG pipeline foundation
- Citation-grounded RAG pipeline foundation
- Evaluation template
- Metric calculation script
- Results analysis script
- Streamlit demo app
- Technical report foundation

## Version 1 Target

The final Version 1 goal is:

- 150–300 academic advising questions
- 20–50 official university sources
- LLM-only baseline
- Basic RAG pipeline
- Citation-grounded RAG pipeline
- 5–6 evaluation metrics
- Error analysis
- Streamlit demo
- 6–10 page technical report

## Pipelines

### 1. LLM-only Baseline

Answers questions without using official source documents.

### 2. Basic RAG

Retrieves relevant UIC source chunks and prepares an answer using retrieved context.

### 3. Citation-grounded RAG

Retrieves official UIC source chunks, requires citations, and includes safety behavior for high-risk advising questions.

## Evaluation Metrics

AdvisingBench evaluates model outputs using:

- Factual accuracy
- Hallucination
- Groundedness
- Citation precision
- Completeness
- Uncertainty handling

## Error Labels

The benchmark tracks error types such as:

- Unsupported claim
- Wrong policy interpretation
- Missing condition
- Outdated information
- Wrong citation
- Overconfident answer
- Irrelevant retrieval
- Incomplete answer
- Unsafe advising

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- sentence-transformers
- FAISS
- Streamlit
- matplotlib
- GitHub
- Markdown

## Repository Structure

```text
advisingbench/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── questions/
│   ├── outputs/
│   └── scores/
├── docs/
├── reports/
├── results/
├── src/
│   ├── collection/
│   ├── retrieval/
│   ├── pipelines/
│   ├── evaluation/
│   └── analysis/
└── README.md
