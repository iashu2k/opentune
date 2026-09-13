# Task Spec — Financial QA (v1.1, locked 2026-09-13)

This document is the frozen contract that `src/opentune/extract.py` implements.
Changes to scoring semantics require a version bump here AND updates to the test
suite in the same commit.

## Input

- A financial question plus context (passage and/or table) drawn from SEC
  filings (FinQA format for Phase 0).

## Output contract

The model must produce, in order:

1. **Reasoning** — free-form chain of thought (required for prompted baselines;
   optional post-SFT).
2. **Program** — one line, `Program: op(args)[, op(args)...]` FinQA-style
   (e.g. `Program: subtract(5000, 3200), divide(#0, 3200)`). Not scored in
   Phase 0; parsed by the Phase 2 format reward, so the grammar is locked now.
3. **Answer line** — final line matching `answer\s*[:\-–]?\s*<value>`
   (case-insensitive, word-bounded so `Answers:` does not match). `<value>` is
   a bare number with optional `$`, commas, `%`, or accounting-parentheses:

   - `ANSWER: 42`
   - `ANSWER: -0.125`
   - `ANSWER: 12.5%`
   - `ANSWER: $1,234,567`
   - `ANSWER: (1,234)`  → −1234 (accounting negative)

The answer line must be on its own line. The string `answer:` occurring
mid-paragraph does NOT count.

## Extraction rules

1. Collect all answer lines; use the **last non-empty** one (self-correction
   wins over first answer).
2. If no answer line exists → status `NO_ANSWER`, scored incorrect.
3. If the answer text can't be parsed as a number → status `NON_NUMERIC`,
   scored incorrect. `N/A`, empty values, and unit-burdened strings like
   `42 million` are all `NON_NUMERIC`.
4. Misplaced commas are normalized, not rejected: `1,2,3` → `123`.
5. Unicode minus signs (−, –, —) normalize to ASCII `-`.

## Canonicalization (both gold and prediction, identical code path)

- Strip currency symbols, commas, backticks, quotes.
- Parenthesized values are negative.
- Values carrying `%` or `percent` are divided by 100 (canonical space is
  fraction-of-1). Consequence: gold `"37.5%"`, gold `"0.375"`, pred
  `"37.5%"`, and pred `"0.375"` are all equal; pred `"37.5"` vs gold
  `"0.375"` is a MISS — by design, documented here.

## Scoring rule

```
correct  ⟺  |pred − gold| ≤ tol,  where
tol = 0.001 · |gold|   if |gold| ≥ 1000
      0.01             otherwise
```

Cent-level strictness (0.01 absolute) for typical answers; the 0.1% relative
tolerance only kicks in for magnitudes ≥ 1000. Computed in canonical space on
`Decimal` values (no float drift).

**Design note (2026-09-13):** the original draft used `max(0.01, 0.001·|gold|)`,
which let the relative arm dominate from |gold| ≥ 10 — making e.g. gold 42.45
accept predictions off by 0.042. Caught by boundary tests before any baseline
ran; switched to the piecewise rule above. Logged in the README experiment log.

## Correct examples

| Prediction ending | Gold | Correct? | Why |
|---|---|---|---|
| `ANSWER: 42.0` | `42` | ✓ | numeric identity |
| `ANSWER: 0.375` | `37.5%` | ✓ | canonical percent space |
| `ANSWER: (1,234)` | `-1234.0` | ✓ | accounting negative |
| `ANSWER: 1,000,500` | `1000000` | ✓ | diff 500 ≤ rel tol 1000 |
| `ANSWER: 1,2,3` | `123` | ✓ | comma misplacement normalized |

## Rejected examples

| Prediction ending | Gold | Correct? | Why |
|---|---|---|---|
| `ANSWER: 42.461` | `42.45` | ✗ | diff 0.011 > 0.01 |
| `ANSWER: 1,002,000` | `1000000` | ✗ | diff 2000 > rel tol 1000 |
| `ANSWER: 37.5` | `0.375` | ✗ | bare number treated literally |
| `ANSWER: N/A` | `42` | ✗ | NON_NUMERIC |
| `The answer: 42.` (mid-paragraph) | `42` | ✗ | no answer line |
| *(no answer line)* | `42` | ✗ | NO_ANSWER |