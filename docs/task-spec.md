# Task Spec — Financial QA (draft)

- **Input:** financial question + context passage/table from SEC filings
- **Output:** reasoning, numeric program, then a final answer line: `ANSWER: <value>`
- **Scoring:** exact match against gold after rule-based extraction
- **Open questions:** (fill in on Day 2) percentage handling, rounding tolerance,
  multi-answer disambiguation