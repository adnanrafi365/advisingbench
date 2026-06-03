# AdvisingBench: A Groundedness Benchmark for Evaluating LLM Hallucinations in Academic Advising

## Abstract

AdvisingBench is a research-style benchmark for evaluating whether large language models can accurately answer undergraduate academic advising questions using official university documents. The project focuses on hallucination, groundedness, citation precision, completeness, and uncertainty handling in academic advising answers. Version 1 uses public official University of Illinois Chicago sources, a starter dataset of advising questions, retrieval-based pipelines, and an evaluation framework to compare LLM-only, basic RAG, and citation-grounded RAG approaches.

## 1. Introduction

Large language models are increasingly used for information-seeking tasks, including student support and academic planning. However, academic advising is a high-impact domain where incorrect or overconfident answers can negatively affect course registration, graduation planning, transfer credit decisions, academic standing, petitions, and international student compliance.

This project investigates whether LLMs can answer academic advising questions more safely when grounded in official university sources.

## 2. Motivation

Students often ask advising questions such as:

- Can I repeat a course?
- Will my transfer credit count?
- How many credits do I need to graduate?
- Can an international student take only online classes?
- What happens if I withdraw from a term?

If an AI system answers these questions incorrectly, the consequences may be serious. AdvisingBench evaluates where LLMs succeed, where retrieval helps, and where models still hallucinate or misuse citations.

## 3. Research Question

How accurately can large language models answer undergraduate academic advising questions when grounded in official university documents, and where do they still hallucinate or misuse citations?

## 4. Dataset

Version 1 uses a UIC-only scope.

The current starter dataset includes:

- 30 academic advising questions
- Gold/reference answers
- Category labels
- Source IDs
- Official source URLs
- Risk levels
- Answer types

The final Version 1 goal is to expand the dataset to 150–300 advising questions.

## 5. Source Collection

The project uses public official UIC sources only. These include UIC Catalog pages, Registrar pages, Admissions pages, College of Engineering pages, Computer Science degree pages, and Office of International Services pages.

Private student data, login-protected pages, transcripts, degree audits, advisor emails, Canvas content, and my.uic.edu data are excluded.

Raw copied page text is kept local and is not published publicly. Public GitHub files include source metadata, source URLs, code, and project documentation.

## 6. Retrieval System

The retrieval system uses official UIC source text files that are cleaned and split into chunks. The current retrieval system includes:

- Source chunking script
- 300 local text chunks
- Sentence-transformer embeddings
- FAISS vector index
- Retrieval testing script

For a student question, the system retrieves the top relevant source chunks and returns source IDs, URLs, scores, and text previews.

## 7. Answer Pipelines

The project currently includes three pipeline foundations:

### 7.1 LLM-only Baseline

This pipeline answers questions without retrieval. It serves as the baseline for measuring hallucination and lack of grounding.

### 7.2 Basic RAG Pipeline

This pipeline retrieves relevant UIC source chunks and prepares an answer using retrieved context.

### 7.3 Citation-grounded RAG Pipeline

This pipeline retrieves official source chunks and requires citations. It also includes safety behavior for high-risk advising questions.

The current version uses placeholder answers. Future versions will connect these pipelines to an LLM API.

## 8. Evaluation Metrics

AdvisingBench evaluates model outputs using six metrics:

- Factual accuracy
- Hallucination
- Groundedness
- Citation precision
- Completeness
- Uncertainty handling

Each metric is scored using a 0, 0.5, or 1 scale.

The total score is calculated as:

```text
overall_score = factual_accuracy + hallucination + groundedness + citation_precision + completeness + uncertainty_handling
