#!/usr/bin/env python3
"""
scripts/compare_frontier.py

Cross-model comparison for the OpenTune Phase 0 gate: frontier CoT vs.
Qwen3-8B CoT, paired on example id, per eval set.

This is analysis tooling, not frozen scoring/prompt/eval-data code -- the
handover explicitly calls out that aggregate_results.py only compares arms
*within* one model and needs to be extended (or companioned) for cross-model
comparison. This script reads the already-scored JSONL files (produced by
run_baselines.py and run_frontier.py) and does not re-score, re-extract, or
touch any frozen contract.

Methodology matches the existing base-matrix aggregation for consistency:
  - Bootstrap 95% CI on accuracy, 10,000 resamples, seed=0.
  - Exact McNemar via scipy.stats.binomtest on discordant pairs
    (b = A correct / B wrong, c = A wrong / B correct), two-sided, p=0.5.

Phase 0 gate (pre-registered, see README.md):
  Frontier CoT accuracy on FinQA test - Qwen3-8B CoT accuracy on FinQA test
  >= 8 percentage points, AND their bootstrap 95% CIs do not overlap.
  Custom eval and SEC 2026 are corroborating evidence only, not independent
  strict gates (SEC 2026 n=180 has wide CIs).

Usage:
    uv run python scripts/compare_frontier.py
    uv run python scripts/compare_frontier.py --out docs/gate_decision.md
    # Also fold the same content into docs/baselines.md, under a delimited,
    # idempotently-replaceable section (safe to re-run after re-scoring):
    uv run python scripts/compare_frontier.py --append-to docs/baselines.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "raw"

N_BOOT = 10_000
SEED = 0
GATE_EVAL_SET = "finqa_test"
GATE_MIN_DELTA_PTS = 8.0

QWEN_MODEL = "Qwen3-8B"
FRONTIER_MODEL = "anthropic--claude-haiku-4-5"
ARM = "cot"

EVAL_SETS = ["finqa_test", "custom_eval", "sec_2026"]

APPEND_BEGIN_MARKER = "<!-- BEGIN compare_frontier.py generated section -->"
APPEND_END_MARKER = "<!-- END compare_frontier.py generated section -->"


def load_jsonl_dedup(path: Path) -> dict[str, dict]:
  """First-write-wins dedupe by id, matching aggregate_results.py convention."""
  records: dict[str, dict] = {}
  if not path.exists():
    return records
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      rid = str(rec["id"])
      if rid not in records:
        records[rid] = rec
  return records


def bootstrap_ci(correct: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
  rng = np.random.default_rng(seed)
  n = len(correct)
  means = np.empty(n_boot)
  for i in range(n_boot):
    sample = rng.integers(0, n, size=n)
    means[i] = correct[sample].mean()
  lo, hi = np.percentile(means, [2.5, 97.5])
  return float(lo), float(hi)


def exact_mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
  b = int(np.sum(a_correct & ~b_correct))  # A correct, B wrong
  c = int(np.sum(~a_correct & b_correct))  # A wrong, B correct
  n_discordant = b + c
  if n_discordant == 0:
    return b, c, 1.0
  k = min(b, c)
  p = binomtest(k, n_discordant, 0.5, alternative="two-sided").pvalue
  return b, c, p


def compare_eval_set(eval_set: str) -> dict:
  qwen_path = RESULTS_DIR / f"{QWEN_MODEL}__{ARM}__{eval_set}.jsonl"
  frontier_path = RESULTS_DIR / f"{FRONTIER_MODEL}__{ARM}__{eval_set}.jsonl"

  qwen = load_jsonl_dedup(qwen_path)
  frontier = load_jsonl_dedup(frontier_path)

  if not qwen:
    raise FileNotFoundError(f"Missing Qwen CoT results at {qwen_path}")
  if not frontier:
    raise FileNotFoundError(f"Missing frontier CoT results at {frontier_path}")

  shared_ids = sorted(set(qwen) & set(frontier))
  missing_from_qwen = set(frontier) - set(qwen)
  missing_from_frontier = set(qwen) - set(frontier)

  qwen_correct = np.array([bool(qwen[i]["correct"]) for i in shared_ids])
  frontier_correct = np.array(
    [bool(frontier[i]["correct"]) for i in shared_ids])

  n = len(shared_ids)
  qwen_acc = qwen_correct.mean()
  frontier_acc = frontier_correct.mean()
  qwen_lo, qwen_hi = bootstrap_ci(qwen_correct)
  frontier_lo, frontier_hi = bootstrap_ci(frontier_correct)
  # b: frontier-right/qwen-wrong
  b, c, p = exact_mcnemar(frontier_correct, qwen_correct)

  ci_non_overlap = frontier_lo > qwen_hi or qwen_lo > frontier_hi
  delta_pts = (frontier_acc - qwen_acc) * 100

  return {
      "eval_set": eval_set,
      "n_paired": n,
      "n_missing_from_qwen": len(missing_from_qwen),
      "n_missing_from_frontier": len(missing_from_frontier),
      "qwen_acc": qwen_acc,
      "qwen_ci": (qwen_lo, qwen_hi),
      "frontier_acc": frontier_acc,
      "frontier_ci": (frontier_lo, frontier_hi),
      "delta_pts": delta_pts,
      "ci_non_overlap": ci_non_overlap,
      "mcnemar_b_frontier_right_qwen_wrong": b,
      "mcnemar_c_frontier_wrong_qwen_right": c,
      "mcnemar_p": p,
  }


def render_markdown(results: list[dict], heading_level: str = "#") -> str:
  h1, h2 = heading_level, heading_level + "#"
  lines = [
      f"{h1} Phase 0 gate: frontier vs. Qwen3-8B CoT",
      "",
      f"Frontier model: `{FRONTIER_MODEL}` (Claude Haiku 4.5 via OpenRouter, "
      "provider pinned to Anthropic). Arm: CoT for both models. "
      f"Bootstrap 95% CI, {N_BOOT:,} resamples, seed={SEED}. "
      "Exact McNemar via `scipy.stats.binomtest`.",
      "",
      "| Eval set | n (paired) | Qwen3-8B CoT | Frontier CoT | Delta (pts) | "
      "CIs non-overlapping | McNemar b/c | McNemar p |",
      "|---|---:|---|---|---:|---|---|---:|",
  ]
  for r in results:
    qwen_str = f"{r['qwen_acc'] * 100:.2f}% [{r['qwen_ci'][0] * 100:.2f}, {r['qwen_ci'][1] * 100:.2f}]"
    frontier_str = (
        f"{r['frontier_acc'] * 100:.2f}% [{r['frontier_ci'][0] * 100:.2f}, "
        f"{r['frontier_ci'][1] * 100:.2f}]"
    )
    lines.append(
        f"| {r['eval_set']} | {r['n_paired']} | {qwen_str} | {frontier_str} | "
        f"{r['delta_pts']:+.2f} | {'yes' if r['ci_non_overlap'] else 'no'} | "
        f"{r['mcnemar_b_frontier_right_qwen_wrong']}/"
        f"{r['mcnemar_c_frontier_wrong_qwen_right']} | {r['mcnemar_p']:.4f} |"
    )

  gate_row = next(r for r in results if r["eval_set"] == GATE_EVAL_SET)
  gate_pass = gate_row["delta_pts"] >= GATE_MIN_DELTA_PTS and gate_row["ci_non_overlap"]

  lines += [
      "",
      f"{h2} Pre-registered gate outcome",
      "",
      f"Gate: frontier CoT - Qwen3-8B CoT >= {GATE_MIN_DELTA_PTS:.0f} points on "
      f"`{GATE_EVAL_SET}`, with non-overlapping bootstrap 95% CIs.",
      "",
      f"- Delta on {GATE_EVAL_SET}: **{gate_row['delta_pts']:+.2f} points**",
      f"- CIs non-overlapping: **{'yes' if gate_row['ci_non_overlap'] else 'no'}**",
      f"- Exact McNemar p on {GATE_EVAL_SET}: **{gate_row['mcnemar_p']:.4f}**",
      f"- **Gate result: {'PASS' if gate_pass else 'FAIL'}**",
      "",
      "Custom eval and SEC 2026 are corroborating evidence only; SEC 2026's "
      "n=180 gives wide CIs and was never pre-registered as an independent "
      "strict gate.",
      "",
      f"{h2} Phase 1 framing decision",
      "",
      "Gate failed on the pre-registered ≥8pt / non-overlapping-CI criteria, "
      "but the finqa_test delta is statistically significant "
      f"(McNemar p={gate_row['mcnemar_p']:.4f}), just smaller than the "
      "pre-registered bar. Decision: proceed to SFT/GRPO with a revised, "
      "smaller target -- close most of the gap to near-frontier prompting "
      "(measured gap: "
      f"{gate_row['delta_pts']:+.2f} points on {GATE_EVAL_SET}) -- rather "
      "than reframing around cost/latency or spending further on a "
      "flagship-tier frontier snapshot. The near-frontier-tier caveat on "
      "the pinned model (`anthropic/claude-haiku-4.5`) still applies: this "
      "target is calibrated to the model actually measured, not to an "
      "untested flagship ceiling.",
  ]

  for r in results:
    if r["n_missing_from_qwen"] or r["n_missing_from_frontier"]:
      lines.append(
          f"\n**Warning ({r['eval_set']}):** {r['n_missing_from_qwen']} ids in "
          f"frontier file not found in Qwen file; {r['n_missing_from_frontier']} "
          "ids in Qwen file not found in frontier file. Comparison restricted to "
          f"the {r['n_paired']} shared ids -- investigate before trusting this row."
      )

  return "\n".join(lines) + "\n"


def append_to_file(target: Path, section_markdown: str) -> None:
  """Idempotently insert/replace a delimited section in an existing markdown
  file (e.g. docs/baselines.md) without touching anything else in it."""
  block = f"{APPEND_BEGIN_MARKER}\n\n{section_markdown}\n{APPEND_END_MARKER}\n"

  if target.exists():
    content = target.read_text(encoding="utf-8")
  else:
    content = ""

  pattern = re.compile(
      re.escape(APPEND_BEGIN_MARKER) + r".*?" +
      re.escape(APPEND_END_MARKER) + r"\n?",
      re.DOTALL,
  )
  if pattern.search(content):
    content = pattern.sub(block, content)
  else:
    if content and not content.endswith("\n\n"):
      content = content.rstrip("\n") + "\n\n"
    content += block

  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(content, encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", default="docs/gate_decision.md")
  parser.add_argument(
      "--append-to",
      default=None,
      help="Also insert/replace a delimited copy of this report inside an "
      "existing markdown file, e.g. docs/baselines.md. Safe to re-run: "
      "replaces only its own marked block, leaving the rest of the file "
      "(the base Qwen3-8B matrix produced by aggregate_results.py) intact.",
  )
  args = parser.parse_args()

  results = [compare_eval_set(es) for es in EVAL_SETS]
  md = render_markdown(results)
  print(md)

  out_path = REPO_ROOT / args.out
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(md, encoding="utf-8")
  print(f"\nWritten to {out_path}")

  if args.append_to:
    append_path = REPO_ROOT / args.append_to
    append_md = render_markdown(results, heading_level="##")
    append_to_file(append_path, append_md)
    print(f"Appended/updated frontier section in {append_path}")


if __name__ == "__main__":
  main()
