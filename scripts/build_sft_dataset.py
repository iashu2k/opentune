"""
scripts/build_sft_dataset.py

Builds the Phase 1 SFT training set (v1) from
data/processed/finqa_train_decontaminated.parquet.

v1 scope decisions (confirmed 2026-09-22, see docs/phase1_gate.md context):
  - Excludes rows whose program uses table_average / table_max / table_min /
    table_sum (~3.3% of rows, 195/5828 confirmed on the real run). These ops
    reference table row LABELS, not literal numbers, and require parsing the
    linearized table in `context` to resolve -- out of scope for v1.
    Excluded rows are logged, not dropped silently.
  - Handles const_<N> constants (e.g. const_100, const_1000000) AND the
    const_m<N> family (e.g. const_m1 == -1), confirmed 2026-09-22 by
    re-executing real rows using const_m1 and matching gold exactly.
  - Every remaining row's `program` is re-executed against its stored `gold`
    as a hard correctness gate. On the full 2026-09-22 run: 5,633/5,633
    arithmetic-op rows matched gold exactly (0 parse errors, 0 mismatches).
    Rows that fail this gate are dropped and logged, never silently included,
    and never allowed to crash the whole run.
  - Reasoning traces are synthesized deterministically from the `program`
    DSL (terse, mechanical, step-by-step) -- never freely generated -- so
    the reasoning can never contradict the gold answer.
  - Output target format matches the FROZEN task-spec output contract
    exactly: reasoning -> "Program: ..." line -> "ANSWER: <number>" line.
  - Prompts are rendered via the FROZEN render_prompt("cot", question,
    context) -- not reimplemented here.
  - A small deterministic validation slice (--val-fraction, default 0.05)
    is held out purely for training-loss monitoring during SFT (e.g. W&B
    eval_loss curves). This is NOT a substitute for the frozen finqa_test /
    custom_eval / sec_2026 gate sets -- the pre-registered Phase 1 gate
    (docs/phase1_gate.md) is evaluated only on those frozen sets.

Usage:
    uv run python scripts/build_sft_dataset.py \\
        --input data/processed/finqa_train_decontaminated.parquet \\
        --output data/processed/sft_train_v1.jsonl \\
        --val-output data/processed/sft_val_v1.jsonl \\
        --val-fraction 0.05 \\
        --seed 42 \\
        --report data/processed/sft_train_v1_report.json

Do NOT edit the frozen task-spec, extractor, prompt templates, or
decontamination logic to make this script's output "look better" -- any
row this script can't handle correctly should be excluded and logged, not
forced.
"""

import argparse
import json
import random
import re
from decimal import Decimal
from pathlib import Path

import pandas as pd

try:
  from opentune.prompts.templates import render_prompt
except ImportError as e:
  raise SystemExit(
      "Could not import render_prompt from opentune.prompts.templates. "
      "Run this script from the repo root with `uv run python scripts/build_sft_dataset.py` "
      "so the local package is importable. Original error: " + str(e)
  )

TABLE_OPS = ("table_average", "table_max", "table_min", "table_sum")

OP_TEXT = {
    "add": "add {a} and {b}",
    "subtract": "subtract {b} from {a}",
    "multiply": "multiply {a} by {b}",
    "divide": "divide {a} by {b}",
    "exp": "raise {a} to the power of {b}",
}

CALL_RE = re.compile(r"(\w+)\(([^)]*)\)")
CONST_M_RE = re.compile(r"^m(\d*)$")


class ProgramError(Exception):
  pass


def uses_table_op(program_str: str) -> bool:
  return any(op in program_str for op in TABLE_OPS)


def parse_const(suffix: str):
  """
  Resolves the part after 'const_'. Handles:
    - plain numbers: '100' -> 100.0, '1000000' -> 1000000.0
    - the m<N> negative-constant family confirmed 2026-09-22:
      'm1' -> -1.0 (re-executed and verified against real rows)
      'm2' -> -2.0 (generalized form, supported defensively)
  """
  m = CONST_M_RE.match(suffix)
  if m:
    digits = m.group(1)
    magnitude = float(digits) if digits else 1.0
    return -magnitude
  return float(suffix.replace(",", ""))


def parse_arg(tok: str, values: list):
  """
  Resolve a single program argument to (numeric_value, display_label).
  Raises ProgramError (never a raw ValueError) on any token this parser
  doesn't recognize, so callers can log-and-skip instead of crashing.
  """
  tok = tok.strip()
  try:
    if tok.startswith("#"):
      idx = int(tok[1:])
      if idx >= len(values):
        raise ProgramError(
          f"Reference #{idx} out of range (have {len(values)} prior steps)")
      return values[idx], f"the result of step {idx + 1}"
    if tok.startswith("const_"):
      suffix = tok.replace("const_", "")
      val = parse_const(suffix)
      return val, tok
    if tok.endswith("%"):
      num_str = tok[:-1].replace(",", "")
      return float(num_str) / 100.0, tok
    num_str = tok.replace(",", "")
    return float(num_str), tok
  except ValueError:
    raise ProgramError(f"Unrecognized argument token: {tok!r}")


def parse_and_execute_program(program_str: str):
  """
  Executes a program string of the form
  'op(arg, arg), op(arg, arg), ...' and returns (values, sentences).
  Raises ProgramError on any op not in OP_TEXT, any malformed call, or
  any unparseable argument token (table_* ops must be filtered out
  before calling this).
  """
  calls = CALL_RE.findall(program_str)
  if not calls:
    raise ProgramError(f"No parseable op() calls found in: {program_str!r}")

  values, sentences = [], []
  for i, (op, argstr) in enumerate(calls):
    if op not in OP_TEXT:
      raise ProgramError(f"Unsupported op '{op}' in program: {program_str!r}")

    raw_args = [a for a in argstr.split(",")]
    if len(raw_args) != 2:
      raise ProgramError(
        f"Expected 2 args for op '{op}', got {raw_args} in {program_str!r}")

    parsed = [parse_arg(a, values) for a in raw_args]
    (a_val, a_label), (b_val, b_label) = parsed

    if op == "add":
      result = a_val + b_val
    elif op == "subtract":
      result = a_val - b_val
    elif op == "multiply":
      result = a_val * b_val
    elif op == "divide":
      if b_val == 0:
        raise ProgramError(f"Division by zero in step {i} of {program_str!r}")
      result = a_val / b_val
    elif op == "exp":
      result = a_val ** b_val
    else:
      raise ProgramError(f"Unhandled op '{op}'")

    values.append(result)
    text = OP_TEXT[op].format(a=a_label, b=b_label)
    sentences.append(f"Step {i + 1}: {text} to get {result:.5g} (#{i}).")

  return values, sentences


def gold_tolerance(gold: float) -> float:
  """Mirrors the frozen extractor's piecewise tolerance rule (task-spec v1.1)."""
  gold_d = Decimal(str(abs(gold)))
  if gold_d >= 1000:
    return float(gold_d * Decimal("0.001"))
  return 0.01


def build_target_text(program_str: str, gold_str: str):
  """
  Returns (target_text, computed_value, matches_gold: bool).
  Raises ProgramError if the program can't be executed at all -- callers
  must catch ProgramError, never let it propagate.
  """
  values, sentences = parse_and_execute_program(program_str)
  computed = values[-1]
  gold_val = float(gold_str)
  tol = gold_tolerance(gold_val)
  matches = abs(computed - gold_val) <= tol

  reasoning = " ".join(sentences)
  target_text = f"{reasoning}\nProgram: {program_str}\nANSWER: {gold_str}"
  return target_text, computed, matches


def write_jsonl(path: Path, rows: list):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input", default="data/processed/finqa_train_decontaminated.parquet")
  parser.add_argument("--output", default="data/processed/sft_train_v1.jsonl")
  parser.add_argument(
    "--val-output", default="data/processed/sft_val_v1.jsonl")
  parser.add_argument("--val-fraction", type=float, default=0.05,
                      help="Fraction of kept rows held out for training-loss monitoring only. "
                      "This is NOT the pre-registered gate eval -- that stays on the frozen "
                      "finqa_test/custom_eval/sec_2026 sets per docs/phase1_gate.md.")
  parser.add_argument("--seed", type=int, default=42,
                      help="Deterministic shuffle seed for the val split.")
  parser.add_argument(
    "--report", default="data/processed/sft_train_v1_report.json")
  parser.add_argument("--prompt-arm", default="cot",
                      help="Prompt template arm to render targets against")
  args = parser.parse_args()

  input_path = Path(args.input)
  output_path = Path(args.output)
  val_output_path = Path(args.val_output)
  report_path = Path(args.report)

  df = pd.read_parquet(input_path)
  total_rows = len(df)

  excluded_table_ids = []
  parse_error_rows = []  # (id, error message)
  mismatch_rows = []     # (id, computed, gold)
  kept_rows = []

  for _, row in df.iterrows():
    row_id = row["id"]
    program_str = row["program"]
    gold_str = str(row["gold"])
    question = row["question"]
    context = row["context"]

    if uses_table_op(program_str):
      excluded_table_ids.append(row_id)
      continue

    try:
      target_text, computed, matches = build_target_text(program_str, gold_str)
    except ProgramError as e:
      parse_error_rows.append(
        {"id": row_id, "error": str(e), "program": program_str})
      continue
    except Exception as e:
      parse_error_rows.append({
          "id": row_id,
          "error": f"Unexpected {type(e).__name__}: {e}",
          "program": program_str,
      })
      continue

    if not matches:
      mismatch_rows.append({
          "id": row_id,
          "program": program_str,
          "computed": computed,
          "gold": gold_str,
      })
      continue

    prompt_text = render_prompt(args.prompt_arm, question, context)
    kept_rows.append({
        "id": row_id,
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": target_text},
        ],
    })

  # Deterministic train/val split -- val slice is for loss-curve monitoring
  # only, never for the pre-registered gate.
  rng = random.Random(args.seed)
  shuffled = kept_rows[:]
  rng.shuffle(shuffled)
  n_val = int(round(len(shuffled) * args.val_fraction))
  val_rows = shuffled[:n_val]
  train_rows = shuffled[n_val:]

  write_jsonl(output_path, train_rows)
  write_jsonl(val_output_path, val_rows)

  report = {
      "input_file": str(input_path),
      "total_rows_in_input": total_rows,
      "excluded_table_op_rows": len(excluded_table_ids),
      "excluded_table_op_ids": excluded_table_ids,
      "parse_error_rows": len(parse_error_rows),
      "parse_error_details": parse_error_rows,
      "gold_mismatch_rows": len(mismatch_rows),
      "gold_mismatch_details": mismatch_rows,
      "kept_rows_total": len(kept_rows),
      "kept_fraction_of_input": round(len(kept_rows) / total_rows, 4) if total_rows else 0,
      "val_fraction": args.val_fraction,
      "val_split_seed": args.seed,
      "train_rows": len(train_rows),
      "val_rows": len(val_rows),
      "prompt_arm_used": args.prompt_arm,
  }
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

  print(f"Input rows:               {total_rows}")
  print(f"Excluded (table_* ops):   {len(excluded_table_ids)}")
  print(f"Excluded (parse error):   {len(parse_error_rows)}")
  print(f"Excluded (gold mismatch): {len(mismatch_rows)}")
  print(f"Kept (total):             {len(kept_rows)}")
  print(f"  -> train split:         {len(train_rows)} -> {output_path}")
  print(f"  -> val split (monitor): {len(val_rows)} -> {val_output_path}")
  print(f"Report written to:       {report_path}")


if __name__ == "__main__":
  main()
