# AdvisingBench

**AdvisingBench** is a research-style benchmark for evaluating whether large language models can accurately answer undergraduate academic advising questions using official university documents.

## Research Question

How accurately can large language models answer undergraduate academic advising questions when grounded in official university documents, and where do they still hallucinate or misuse citations?

## Why This Matters

Academic advising questions can affect course registration, transfer planning, graduation progress, petitions, academic standing, and international student compliance. Incorrect AI-generated advice can negatively impact students.

## Version 1 Scope

- UIC-only scope
- 150–300 academic advising questions
- 20–50 official UIC sources
- LLM-only baseline
- Basic RAG pipeline
- RAG + citation-checking pipeline
- Evaluation for hallucination, groundedness, citation precision, completeness, and uncertainty handling
- Streamlit or Gradio demo
- 6–10 page technical report

## Planned Pipelines

1. LLM-only baseline
2. Basic RAG pipeline
3. Citation-grounded RAG pipeline
4. Optional: RAG + self-verification

## Evaluation Metrics

- Factual accuracy
- Hallucination rate
- Groundedness
- Citation precision
- Completeness
- Uncertainty handling

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- sentence-transformers
- FAISS or ChromaDB
- LangChain
- OpenAI or Gemini API
- Streamlit or Gradio

## Repository Structure

```text
advisingbench/
├── app/
├── configs/
├── data/
├── docs/
├── notebooks/
├── reports/
├── results/
└── src/

