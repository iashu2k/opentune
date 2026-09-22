"""Analyze which completions in the trace log were score()-correct vs
rejected by is_clean() vs is_valid_program() -- the three tiers of gating.
Reads results/raw/sft_build_trace_log_shard0.jsonl (or any trace log)."""

import json
import re
import sys
from opentune.extract import score


def is_clean(text):
  return text.count("ANSWER:") <= 1


def is_valid_program(text):
  if text.count("Program:") != 1:
    return False
  m = re.search(r"Program:\s*(.+)", text)
  if not m:
    return False
  program_str = m.group(1).strip()
  steps = re.findall(r"[a-z_]+\([^()]*\)", program_str)
  reconstructed = ", ".join(steps)
  return bool(steps) and reconstructed == program_str


path = sys.argv[1] if len(
  sys.argv) > 1 else "results/raw/sft_build_trace_log_shard0.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]

n = len(rows)
score_correct = 0
clean_only = 0
valid_only = 0
clean_and_valid = 0
all_three_gates = 0
star_rows = 0

for r in rows:
  res = score(r["completion"], r["gold"])
  is_score_ok = res.correct and res.extraction.status.value == "ok"
  is_clean_ok = is_clean(r["completion"])
  is_valid_ok = is_valid_program(r["completion"])

  if is_score_ok:
    score_correct += 1
  if is_score_ok and is_clean_ok:
    clean_only += 1
  if is_score_ok and is_valid_ok:
    valid_only += 1
  if is_score_ok and is_clean_ok and is_valid_ok:
    clean_and_valid += 1
  if is_score_ok and is_clean_ok and is_valid_ok and r["source"] == "star":
    all_three_gates += 1

  if r["source"] == "star":
    star_rows += 1

print(f"total rows analyzed: {n}")
print(f"rows source=star (STaR accepted, all 3 gates passed): {star_rows}")
print(
  f"score-correct regardless of source: {score_correct}  ({100 * score_correct / n:.1f}%)")
print(f"score-correct AND clean: {clean_only}  ({100 * clean_only / n:.1f}%)")
print(
  f"score-correct AND valid-program: {valid_only}  ({100 * valid_only / n:.1f}%)")
print(
  f"score-correct AND clean AND valid: {clean_and_valid}  ({100 * clean_and_valid / n:.1f}%)")
print(
  f"score-correct AND clean AND valid AND source=star: {all_three_gates}  ({100 * all_three_gates / n:.1f}%)")
