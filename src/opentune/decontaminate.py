"""Word-level n-gram decontamination: FinQA train vs. eval-side corpus.

Method (locked for dataset card):
  - 8-gram word-level shingles over normalized text
  - Eval corpus = dev + test (question + context); test is blind-checked only
    for contamination statistics, never content
  - Per train example: question-containment = |q-grams ∩ eval| / |q-grams|,
    context-containment likewise (reported, not dropped on — table boilerplate
    legitimately recurs across filings)
  - Drop rule: question-containment >= QUESTION_CONTAINMENT_THRESHOLD

Outputs: data/processed/finqa_train_decontaminated.parquet
         data/processed/decontamination_report.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
N_GRAM = 8
QUESTION_CONTAINMENT_THRESHOLD = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9$.%-]+")


def _tokens(text: str) -> list[str]:
  return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int = N_GRAM) -> frozenset[str]:
  if len(tokens) < n:
    return frozenset()
  return frozenset(" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _eval_corpus(dev: pd.DataFrame, test: pd.DataFrame) -> frozenset[str]:
  grams: set[str] = set()
  for df in (dev, test):
    for _, row in df.iterrows():
      grams |= _ngrams(_tokens(row["question"]))
      grams |= _ngrams(_tokens(row["context"]))
  return frozenset(grams)


def _containment(text: str, corpus: frozenset[str]) -> float:
  grams = _ngrams(_tokens(text))
  if not grams:  # too short to shingle: no evidence of overlap
    return 0.0
  return sum(1 for g in grams if g in corpus) / len(grams)


def main() -> None:
  train = pd.read_parquet(PROCESSED / "finqa_train.parquet")
  dev = pd.read_parquet(PROCESSED / "finqa_dev.parquet")
  test = pd.read_parquet(PROCESSED / "finqa_test.parquet")

  corpus = _eval_corpus(dev, test)
  print(f"eval corpus: {len(corpus):,} distinct {N_GRAM}-grams "
        f"(dev {len(dev)} + test {len(test)})")

  q_scores, c_scores = [], []
  for _, row in train.iterrows():
    q_scores.append(_containment(row["question"], corpus))
    c_scores.append(_containment(row["context"], corpus))
  train["q_containment"] = q_scores
  train["c_containment"] = c_scores

  flagged = train["q_containment"] >= QUESTION_CONTAINMENT_THRESHOLD
  clean = train.loc[~flagged].drop(columns=["q_containment", "c_containment"])
  clean.to_parquet(
    PROCESSED / "finqa_train_decontaminated.parquet", index=False)

  report = {
      "n_gram": N_GRAM,
      "threshold": QUESTION_CONTAINMENT_THRESHOLD,
      "train_raw": len(train),
      "dropped_contaminated": int(flagged.sum()),
      "train_decontaminated": len(clean),
      "q_containment_percentiles": {
          str(p): round(float(train["q_containment"].quantile(p / 100)), 4)
          for p in (50, 90, 95, 99)
      },
      "c_containment_percentiles": {
          str(p): round(float(train["c_containment"].quantile(p / 100)), 4)
          for p in (50, 90, 95, 99)
      },
      "worst_offenders": (
          train.nlargest(5, "q_containment")[
              ["id", "question", "q_containment"]]
          .to_dict(orient="records")
      ),
  }
  with open(PROCESSED / "decontamination_report.json", "w") as f:
    json.dump(report, f, indent=2)

  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
