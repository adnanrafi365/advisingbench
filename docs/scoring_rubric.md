# AdvisingBench Scoring Rubric

## Purpose

This rubric explains how model-generated advising answers will be evaluated.

Each answer will be scored using six metrics:

1. Factual accuracy
2. Hallucination
3. Groundedness
4. Citation precision
5. Completeness
6. Uncertainty handling

Each metric uses a 0, 0.5, or 1 score.

## Metrics

### 1. Factual Accuracy

Measures whether the answer is correct according to official UIC sources.

- 1 = Correct
- 0.5 = Partially correct
- 0 = Incorrect

### 2. Hallucination

Measures whether the answer includes unsupported claims.

- 1 = No hallucination
- 0.5 = Minor hallucination
- 0 = Major hallucination

### 3. Groundedness

Measures whether the answer is based on retrieved official source text.

- 1 = Fully grounded
- 0.5 = Partially grounded
- 0 = Not grounded

### 4. Citation Precision

Measures whether the citation actually supports the claim.

- 1 = Correct citation
- 0.5 = Partially relevant citation
- 0 = Wrong or missing citation

### 5. Completeness

Measures whether the answer includes important conditions, exceptions, and limitations.

- 1 = Complete
- 0.5 = Partially complete
- 0 = Incomplete

### 6. Uncertainty Handling

Measures whether the answer avoids overconfidence when the policy is unclear or student-specific.

- 1 = Good uncertainty handling
- 0.5 = Somewhat overconfident
- 0 = Dangerously overconfident

## Error Labels

Each incorrect or partially incorrect answer may receive one or more error labels:

- Unsupported claim
- Wrong policy interpretation
- Missing condition
- Outdated information
- Wrong citation
- Overconfident answer
- Irrelevant retrieval
- Incomplete answer
- Unsafe advising

## Overall Answer Quality

For each answer, the total score will be calculated as:

total_score = factual_accuracy + hallucination + groundedness + citation_precision + completeness + uncertainty_handling

Maximum score: 6.0

Minimum score: 0.0

A strong answer should score close to 6.0.

A risky answer is any answer with:

- factual_accuracy = 0
- hallucination = 0
- uncertainty_handling = 0
- unsafe_advising error label
