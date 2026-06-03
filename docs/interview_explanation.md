# AdvisingBench Interview Explanation

## 30-Second Explanation

I built AdvisingBench, a research-style benchmark for evaluating whether large language models can safely answer undergraduate academic advising questions using official university documents. The project uses public UIC sources, a dataset of advising questions, FAISS-based retrieval, RAG pipelines, and evaluation metrics such as factual accuracy, hallucination, groundedness, citation precision, completeness, and uncertainty handling.

## 1-Minute Explanation

AdvisingBench is a benchmark I built to test whether LLMs can answer academic advising questions without hallucinating. I focused on UIC advising questions because students often ask important questions about degree requirements, transfer credit, registration, graduation, international student enrollment, and withdrawal policies. If an AI gives wrong advice in these areas, it can seriously affect a student.

I collected public official UIC sources, created a starter dataset of advising questions with gold answers and source URLs, chunked the source documents, built a FAISS vector database using sentence-transformers, and created three pipeline foundations: LLM-only, basic RAG, and citation-grounded RAG. I also built an evaluation framework to score outputs on factual accuracy, hallucination, groundedness, citation precision, completeness, and uncertainty handling.

## Technical Explanation

The project has multiple stages. First, I collected public official UIC sources and created a source manifest with metadata and quality labels. Then I created a starter benchmark dataset of advising questions with categories, risk levels, answer types, gold answers, and source URLs.

For retrieval, I cleaned and chunked the source documents into 300 chunks, embedded them using sentence-transformers/all-MiniLM-L6-v2, and built a FAISS vector index. I then created retrieval tests to verify that questions retrieve relevant UIC sources.

For pipelines, I implemented LLM-only, basic RAG, and citation-grounded RAG foundations. I also created an evaluation template and metric calculation scripts to compare pipeline performance. Finally, I built a Streamlit demo that lets users ask a question and view retrieved official UIC sources.

## Why This Project Matters

This project matters because academic advising is a high-impact setting. Students may make decisions about registration, graduation, withdrawal, transfer credit, or immigration compliance based on advice they receive. LLMs can sound confident even when they are wrong, so it is important to evaluate hallucination, citation accuracy, and uncertainty handling before using AI in student support settings.

## What I Learned

Through this project, I learned how to structure a research-style ML/NLP project from end to end, including source collection, dataset creation, retrieval system design, RAG pipeline development, evaluation metrics, error analysis, and demo development. I also learned how important it is to design evaluation criteria before trusting model outputs.
