"""Smoke test: does every FinQA gold score ITSELF correct through the extractor?

Any gold format that fails here biases every downstream baseline number.
Run: uv run python scripts/smoke_golds.py [--n 200]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from opentune.extract import score

FINQA_DIR = Path("data/raw/finqa")


def load_split(split: str) -> list[dict]:
  with open(FINQA_DIR / f"{split}.json") as f:
    return json.load(f)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=200,
                  help="examples per split to check")
  ap.add_argument("--splits", nargs="+",
                  default=["dev"], choices=["train", "dev", "test"])
  args = ap.parse_args()

  failures: Counter = Counter()
  checked = 0
  bad_examples: list[tuple[str, str, object]] = []

  for split in args.splits:
    data = load_split(split)
    for ex in data[: args.n]:
      for field in ("answer", "exe_ans"):
        gold = ex["qa"].get(field)
        if gold is None:
          continue
        gold_str = str(gold)
        fake_pred = f"reasoning...\nANSWER: {gold_str}"  # identity check
        result = score(fake_pred, gold_str)
        checked += 1
        if not result.correct:
          failures[f"{split}.{field}:{result.extraction.status}"] += 1
          if len(bad_examples) < 15:
            bad_examples.append((split, field, gold))

  print(f"checked {checked} gold values")
  if not failures:
    print("ALL GOLDS PARSE: extractor covers observed FinQA gold formats")
    return
  print("FAILURES (must fix before any baseline runs):")
  for kind, count in failures.most_common():
    print(f"  {count:4d}  {kind}")
  print("\nexample failing golds:")
  for split, field, gold in bad_examples:
    print(f"  [{split}.{field}] {gold!r}")


if __name__ == "__main__":
  main()
