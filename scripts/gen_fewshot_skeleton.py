"""Generate fewshot.json skeleton from verified train exemplars.

Pulls context/question/program/gold verbatim from the decontaminated train
parquet (no transcription error on long figures). Solution fields are filled
by hand afterward and MUST pass tests/test_prompts.py self-consistency checks.

Run: uv run python scripts/gen_fewshot_skeleton.py
"""

import json
from pathlib import Path

import pandas as pd

EXEMPLAR_IDS = ["train-2024", "train-527", "train-1758"]
SRC = Path("data/processed/finqa_train_decontaminated.parquet")
DST = Path("src/opentune/prompts/fewshot.json")

df = pd.read_parquet(SRC)
sel = df[df["id"].isin(EXEMPLAR_IDS)]

missing = set(EXEMPLAR_IDS) - set(sel["id"])
if missing:
    raise SystemExit(f"exemplar ids not found: {missing}")

records = [
    {
        "id": r["id"],
        "context": r["context"],
        "question": r["question"],
        "solution": "",  # FILL: reasoning lines + "Program: ..." + "ANSWER: ..."
        "_program_ref": r["program"],  # reference for hand-verification; delete after
        "_gold_ref": r["gold"],
    }
    for r in sel.to_dict(orient="records")
]

DST.write_text(json.dumps(records, indent=2))
print(f"wrote {DST}")
for r in records:
    print(f"{r['id']}: context_len={len(r['context'])}, "
          f"program={r['_program_ref']}, gold={r['_gold_ref']}")
