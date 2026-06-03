# AdvisingBench GitHub Summary

AdvisingBench is a research-style benchmark for evaluating hallucination, groundedness, citation precision, completeness, and uncertainty handling in LLM-generated academic advising answers.

## What It Does

The project tests whether LLMs can answer undergraduate academic advising questions using official university documents without hallucinating or misusing citations.

## Current Version

The current version includes:

- 20 public official UIC sources
- 30 starter academic advising questions
- Gold/reference answers
- Source URLs and source IDs
- Risk level labels
- Answer type labels
- Source fetching script
- Source quality analysis
- Chunking pipeline
- FAISS vector index builder
- Retrieval test script
- LLM-only baseline foundation
- Basic RAG pipeline foundation
- Citation-grounded RAG pipeline foundation
- Evaluation template
- Metric calculation script
- Results analysis script
- Streamlit demo app
- Technical report foundation

## Why It Matters

Academic advising questions can affect registration, graduation planning, transfer credit, petitions, withdrawal decisions, and international student compliance. This project evaluates whether LLM systems can safely answer these questions when grounded in official sources.

## Tech Stack

- Python
- pandas
- sentence-transformers
- FAISS
- Streamlit
- matplotlib
- GitHub
- Markdown

## Disclaimer

This project is an independent research-style benchmark and is not an official UIC advising tool. Students should verify important academic decisions with official UIC advisors, departments, the Registrar, or the Office of International Services.
