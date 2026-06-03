# AdvisingBench Annotation Guidelines

## Purpose

These guidelines explain how advising questions, reference answers, citations, and evaluation scores should be created.

## Question Categories

- Course prerequisites
- Transfer credit
- Degree requirements
- Major requirements
- Minor requirements
- Repeat policy
- Academic standing
- Petitions
- Graduation requirements
- Registration deadlines
- International student restrictions
- Online course rules
- General student support

## Risk Levels

### Low Risk

General informational questions that are unlikely to seriously affect student progress.

Example:
- Where can I find UIC's academic calendar?

### Medium Risk

Questions that may affect course planning, registration, or graduation progress.

Example:
- Can I repeat a course if I earned a C?

### High Risk

Questions that may affect visa status, graduation eligibility, academic standing, or financial/administrative consequences.

Example:
- Can an F-1 student take only online courses?

## Answer Types

### Direct

The official source gives a clear answer.

### Conditional

The answer depends on specific conditions.

### Uncertain

The official source does not provide enough information, so the answer should recommend contacting an advisor or office.

## Evaluation Metrics

### Factual Accuracy

Does the answer correctly reflect official UIC policy?

Score:
- 1 = correct
- 0.5 = partially correct
- 0 = incorrect

### Hallucination

Does the answer include unsupported claims?

Score:
- 1 = no hallucination
- 0.5 = minor unsupported claim
- 0 = major hallucination

### Groundedness

Is the answer supported by retrieved official sources?

Score:
- 1 = fully grounded
- 0.5 = partially grounded
- 0 = not grounded

### Citation Precision

Are the citations relevant and correct?

Score:
- 1 = correct citation
- 0.5 = partially relevant citation
- 0 = wrong or missing citation

### Completeness

Does the answer include important conditions and limitations?

Score:
- 1 = complete
- 0.5 = missing minor detail
- 0 = missing major condition

### Uncertainty Handling

Does the answer avoid overconfidence when the policy is unclear?

Score:
- 1 = handles uncertainty well
- 0.5 = somewhat overconfident
- 0 = dangerously overconfident

## Error Types

- Unsupported claim
- Wrong policy interpretation
- Missing condition
- Outdated information
- Wrong citation
- Overconfident answer
- Irrelevant retrieval
- Incomplete answer
- Unsafe advising
