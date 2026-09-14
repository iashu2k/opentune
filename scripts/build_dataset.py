"""Build processed FinQA parquets with the locked gold-selection policy.

Policy (docs/task-spec.md v1.1, README §4):
  1. Primary gold: qa.exe_ans (raw numeric)
  2. Fallback: qa.answer string if exe_ans fails extraction
  3. Drop the example if neither parses — identical for every model/arm

Text normalization: FinQA raw text contains LaTeX-escape artifacts (literal
\n, \t sequences) in questions, table cells, and pre/post text. clean_text()
removes them and collapses whitespace. Golds are NOT cleaned here — the
extractor v1.1 handles their artifacts at scoring time.

Run: uv run python scripts/build_dataset.py
Writes: data/processed/finqa_{train,dev,test}.parquet
Prints: per-split exclusion accounting for the dataset card.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from opentune.extract import _parse_number

RAW = Path("data/raw/finqa")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

_ESCAPES_RE = re.compile(r"\\\\[nt]|\\[nt]")
_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
  """Remove literal \n / \t escape artifacts; collapse whitespace."""
  return _WS_RE.sub(" ", _ESCAPES_RE.sub(" ", s)).strip()


def serialize_example(ex: dict) -> str:
  """Deterministic context serialization: pre_text + table + post_text."""
  parts: list[str] = []
  pre = " ".join(clean_text(t) for t in ex.get("pre_text", []) if t.strip())
  if pre:
    parts.append(pre)
  table = ex.get("table", [])
  if table:
    rows = [" | ".join(clean_text(str(c)) for c in row) for row in table]
    parts.append("\n".join(r for r in rows if r))
  post = " ".join(clean_text(t) for t in ex.get("post_text", []) if t.strip())
  if post:
    parts.append(post)
  return "\n".join(parts)


def build_split(split: str) -> tuple[pd.DataFrame, Counter]:
  with open(RAW / f"{split}.json") as f:
    raw = json.load(f)

  records = []
  dropped = Counter()
  gold_sources = Counter()

  for i, ex in enumerate(raw):
    qa = ex["qa"]
    gold, source = None, None

    exe = qa.get("exe_ans")
    if exe is not None and _parse_number(str(exe)) is not None:
      gold, source = str(exe), "exe_ans"
    else:
      ans = qa.get("answer")
      if ans is not None and _parse_number(str(ans)) is not None:
        gold, source = str(ans), "answer_fallback"

    if gold is None:
      reason = "empty_gold" if not (
        qa.get("answer") or qa.get("exe_ans")) else "non_numeric_gold"
      dropped[reason] += 1
      continue

    gold_sources[source] += 1
    records.append(
        {
            "id": f"{split}-{i}",
            "split": split,
            "question": clean_text(str(qa["question"])),
            "context": serialize_example(ex),
            "gold": gold,
            "gold_source": source,
            "program": clean_text(str(qa.get("program", ""))),
        }
    )

  df = pd.DataFrame(records)
  stats = Counter()
  stats["raw"] = len(raw)
  stats["kept"] = len(records)
  stats.update({f"dropped:{k}": v for k, v in dropped.items()})
  stats.update({f"gold:{k}": v for k, v in gold_sources.items()})
  return df, stats


def main() -> None:
  for split in ("train", "dev", "test"):
    df, stats = build_split(split)
    out_path = OUT / f"finqa_{split}.parquet"
    df.to_parquet(out_path, index=False)
    total_drop = sum(v for k, v in stats.items() if k.startswith("dropped:"))
    print(f"\n[{split}] -> {out_path}")
    for k, v in sorted(stats.items()):
      print(f"  {k}: {v}")
    print(f"  exclusion_rate: {total_drop / stats['raw']:.2%}")


if __name__ == "__main__":
  main()
