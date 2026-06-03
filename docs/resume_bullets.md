# AdvisingBench Resume Bullets

## Main Resume Bullet

Built AdvisingBench, a research-style benchmark evaluating hallucination, groundedness, citation precision, completeness, and uncertainty handling in LLM-generated academic advising answers using official university documents.

## Detailed Resume Bullets

- Built a research-style benchmark of undergraduate academic advising questions using official UIC policy documents to evaluate hallucination, groundedness, citation precision, completeness, and uncertainty handling in LLM-generated answers.

- Collected and organized 20 public official UIC sources across catalog, registrar, transfer credit, international student enrollment, engineering advising, and degree requirement pages, while excluding private or login-protected student data.

- Created a starter dataset of 30 academic advising questions with gold answers, source URLs, category labels, risk levels, and answer types, with plans to expand to 150–300 questions.

- Implemented a retrieval system using sentence-transformers and FAISS to embed 300 source chunks and retrieve relevant official UIC sources for academic advising questions.

- Built and compared LLM-only, basic RAG, and citation-grounded RAG pipeline foundations for evaluating how retrieval and citation grounding affect answer quality.

- Designed an evaluation framework with scoring metrics for factual accuracy, hallucination, groundedness, citation precision, completeness, and uncertainty handling.

- Developed scripts for dataset validation, source quality analysis, retrieval testing, metric calculation, and result visualization using Python, pandas, FAISS, sentence-transformers, and matplotlib.

- Created a Streamlit demo that retrieves relevant official UIC source chunks for student advising questions and displays source IDs, URLs, text previews, and safety disclaimers.

## Short Version

Built AdvisingBench, a research-style LLM evaluation benchmark using official UIC documents, FAISS-based retrieval, RAG pipelines, citation grounding, and custom metrics for hallucination, groundedness, citation precision, and uncertainty handling.
