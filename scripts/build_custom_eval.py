"""Build the custom eval set (n≈450) from the FinQA dev split.

Design (recorded in dataset card):
  - Source: FinQA dev, post gold-filter (873 rows). Decontaminated vs train
    by construction — the decontamination corpus included dev+test shingles.
  - Stratified sampling on (primary_operator, context_length_quartile) so
    the sample mirrors dev composition; proportional allocation per stratum.
  - Fixed seed recorded in the output filename metadata + dataset card.
  - Test split untouched. Remaining dev rows are an unused holdout.

Run: uv run python scripts/build_custom_eval.py
Writes: data/eval/custom_eval.parquet  (COMMIT this — eval sets are versioned)
"""

import re
from pathlib import Path

import pandas as pd

N_TARGET = 450
SEED = 42
DEV = Path("data/processed/finqa_dev.parquet")
OUT = Path("data/eval/custom_eval.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

_OP_RE = re.compile(r"^([a-zA-Z_]+)\(")


def op_group(program: str) -> str:
    m = _OP_RE.search(program.strip())
    if not m:
        return "none"
    op = m.group(1)
    if op in ("divide", "subtract", "add", "multiply"):
        return op
    return "table_or_tail" if op.startswith("table_") else "tail"


def main() -> None:
    dev = pd.read_parquet(DEV)
    dev = dev.assign(
        op_group=dev["program"].map(op_group),
        ctx_quartile=pd.qcut(dev["context"].str.len(), 4, labels=False),
    )

    dev["stratum"] = dev["op_group"].astype(str) + "|" + dev["ctx_quartile"].astype(str)
    frac = N_TARGET / len(dev)
    sampled = (
        dev.groupby("stratum", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=SEED))
        .reset_index(drop=True)
    )
    # top up / trim to exactly N_TARGET if rounding drifted
    if len(sampled) > N_TARGET:
        sampled = sampled.sample(N_TARGET, random_state=SEED).reset_index(drop=True)
    elif len(sampled) < N_TARGET:
        rest = dev[~dev["id"].isin(sampled["id"])]
        extra = rest.sample(N_TARGET - len(sampled), random_state=SEED + 1)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled[["id", "split", "question", "context", "gold", "gold_source", "program"]].to_parquet(
        OUT, index=False
    )

    print(f"wrote {OUT} (n={len(sampled)}, seed={SEED})")
    print("\noperator composition (sampled vs dev):")
    comp = pd.DataFrame(
        {
            "sampled": sampled["op_group"].value_counts(normalize=True),
            "dev": dev["op_group"].value_counts(normalize=True),
        }
    ).fillna(0)
    print((comp * 100).round(1).to_string(float_format=lambda x: f"{x:.1f}%"))
    print("\nctx length p50: sampled",
          int(sampled["context"].str.len().median()),
          "| dev", int(dev["context"].str.len().median()))


if __name__ == "__main__":
    main()
