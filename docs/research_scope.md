# AdvisingBench Research Scope

## Project Title

AdvisingBench: A Groundedness Benchmark for Evaluating LLM Hallucinations in Academic Advising

## One-Line Description

AdvisingBench is a research-style benchmark that evaluates whether large language models can accurately answer undergraduate academic advising questions using official university documents without hallucinating, misusing citations, or giving overconfident answers.

## Main Research Question

How accurately can large language models answer undergraduate academic advising questions when grounded in official university documents, and where do they still hallucinate or misuse citations?

## Motivation

Students ask important academic advising questions about course prerequisites, transfer credit, degree requirements, petitions, academic standing, graduation, online course rules, and international student restrictions. If an AI gives wrong advice, it can negatively affect registration, graduation, transfer planning, or visa compliance.

This project evaluates whether LLMs can safely answer these questions using official university sources.

## Version 1 Scope

Version 1 focuses only on public official University of Illinois Chicago sources.

The goal is to build a strong undergraduate-level benchmark before expanding to a larger version.

### Included

- Public UIC catalog pages
- Public UIC registrar pages
- Public UIC advising pages
- Public UIC college requirement pages
- Public UIC transfer credit information
- Public UIC Office of International Services pages
- Public UIC academic calendar and deadline pages
- Public UIC graduation and degree requirement pages

### Not Included

- Private student records
- Degree audits
- Transcripts
- Advisor emails
- my.uic.edu pages
- Canvas or Blackboard content
- Login-protected documents
- Private internal university systems
- Personal student data

## Question Categories

The benchmark will include questions from the following categories:

1. Course prerequisites
2. Transfer credit
3. Degree requirements
4. Major requirements
5. Minor requirements
6. Repeat policy
7. Academic standing
8. Petitions
9. Graduation requirements
10. Registration deadlines
11. International student restrictions
12. Online course rules
13. General student support

## Risk Levels

### Low Risk

Questions that are mostly informational and unlikely to directly affect academic progress.

Example:
Where can I find the UIC academic calendar?

### Medium Risk

Questions that may affect registration, course planning, degree progress, or graduation timing.

Example:
Can I repeat a course if I earned a C?

### High Risk

Questions that may affect visa status, graduation eligibility, academic standing, petitions, or major academic decisions.

Example:
Can an F-1 student take only online courses?

## Answer Types

### Direct

The official source gives a clear answer.

Example:
How many total credit hours are required for a degree?

### Conditional

The answer depends on specific rules, conditions, exceptions, or student status.

Example:
Can I repeat a course if I already passed it?

### Uncertain

The official source does not provide enough information, or the answer depends on an advisor, department, petition, or student-specific situation.

Example:
Will this specific transfer course count toward my major requirement?

## Evaluation Metrics

### Factual Accuracy

Does the model answer correctly according to official UIC sources?

### Hallucination Rate

Does the model include claims that are not supported by the official sources?

### Groundedness

Is the answer supported by the retrieved official source text?

### Citation Precision

Does the answer cite the correct and relevant official source?

### Completeness

Does the answer include important conditions, exceptions, or limitations?

### Uncertainty Handling

Does the model avoid overconfidence when the answer is unclear or student-specific?

## Hallucination Definition

A hallucination occurs when the model includes information that is:

- Not supported by the cited official source
- Contradicted by the official source
- Invented or assumed without evidence
- Overly specific when the source is general
- Presented as policy when the source does not say it

## Correct Citation Definition

A citation is correct when:

- The cited source is official and public
- The cited source directly supports the answer
- The cited source matches the claim being made
- The answer does not cite irrelevant pages
- The answer does not use a correct source to support an unrelated claim

## Unsafe Advising Definition

Unsafe advising occurs when the model gives advice that could seriously harm a student’s academic, financial, immigration, or graduation situation.

Examples:

- Giving incorrect F-1 enrollment advice
- Giving wrong graduation requirement advice
- Giving unsupported petition advice
- Telling a student they are eligible for something without evidence
- Ignoring important conditions or exceptions
- Acting confident when the student should contact an advisor or official office

## Research Boundaries

This project does not claim to replace official academic advising.

The benchmark evaluates model behavior. It is not an official UIC advising tool.

All important student decisions should be verified with official UIC advisors, departments, or university offices.

## Initial Success Criteria

Version 1 will be considered successful if it includes:

- 150 to 300 advising questions
- 20 to 50 official public UIC sources
- LLM-only baseline
- Basic RAG pipeline
- Citation-grounded RAG pipeline
- Evaluation scores across 5 to 6 metrics
- Error analysis
- Streamlit or Gradio demo
- 6 to 10 page technical report
- Clean public GitHub repository
