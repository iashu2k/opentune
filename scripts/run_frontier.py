#!/usr/bin/env python3
"""
scripts/run_frontier.py

Resumable frontier-API runner for OpenTune Phase 0, via OpenRouter.

Mirrors scripts/run_baselines.py's contract exactly so results/aggregates.json
and aggregate_results.py work unchanged:

  - Uses the SAME frozen prompt renderer render_prompt(arm, question, context)
    from src/opentune/prompts/templates.py, and the SAME frozen
    score(output_text, gold_str) from src/opentune/extract.py. This script
    does not define, alter, or duplicate prompt text, scoring rules, or eval
    data.
  - Writes results/raw/{sanitized-model}__{arm}__{eval-set}.jsonl using the
    SAME field names as run_baselines.py's output:
        id, arm, eval_set, model, output, gold, correct, status, extracted,
        n_prompt_tokens, latency_s, ts
    plus frontier-only bookkeeping fields appended (api_model_slug,
    provider_pinned, provider_used, generation_id, completion_tokens,
    cost_usd, cumulative_cost_usd, temperature) -- extra keys, no renames of
    the shared ones, so aggregate_results.py needs zero changes.
  - Resumable: reads existing JSONL IDs on startup and skips them; appends
    and flush()es after every record.
  - Retries rate limits/5xx/network errors with capped exponential backoff.
  - Pins the OpenRouter *provider* to Anthropic directly (not Bedrock/GCP
    Vertex/etc. re-listings of the same model), because OpenRouter shows
    those alternate providers at $1.10 / $5.50 per MTok vs. Anthropic's own
    $1.00 / $5.00 -- pinning keeps both cost and inference behavior
    reproducible.

API key handling:
  - Reads OPENROUTER_API_KEY from a local .env file via python-dotenv (never
    hardcoded, never committed). Add `.env` to .gitignore if it isn't
    already there.

    .env (not committed):
        OPENROUTER_API_KEY=sk-or-v1-...

Pinned snapshot (2026-09-16 decision, see README.md "Frontier baseline"):
    provider (via)   : OpenRouter -> Anthropic
    model            : Claude Haiku 4.5
    OpenRouter slug  : anthropic/claude-haiku-4.5
    pricing (Anthropic endpoint) : $1.00 / MTok input, $5.00 / MTok output
    context / max out: 200K / 64K tokens
    decoding         : temperature=0, no `reasoning` param set (extended
                        thinking stays off by default) -- we want prompted
                        CoT text in the transcript, for parity with the Qwen
                        CoT arm, not the provider's internal reasoning-token
                        mode.

Usage:
    # .env in repo root (gitignored):
    #   OPENROUTER_API_KEY=sk-or-v1-...
    uv run python scripts/run_frontier.py --estimate --eval-set all
    uv run python scripts/run_frontier.py --eval-set finqa_test
    uv run python scripts/run_frontier.py --eval-set all --max-tokens 700
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Frozen imports -- do not modify prompts, scoring, or eval data.
# Signature confirmed against templates.py / run_baselines.py:
#   render_prompt(name: str, question: str, context: str) -> str
#   score(output_text: str, gold_str: str) -> ScoreResult
#     ScoreResult.correct: bool
#     ScoreResult.extraction.status: Status (has .value)
#     ScoreResult.extraction.value: numeric | None
# --------------------------------------------------------------------------
try:
  from opentune.prompts.templates import render_prompt  # frozen
  from opentune.extract import score  # frozen
except ImportError as exc:  # pragma: no cover
  print(
      "FATAL: could not import frozen render_prompt / score from "
      "src/opentune. This script must reuse the exact same prompt "
      "template and scoring contract as run_baselines.py -- fix the import "
      f"path, do not reimplement. Original error: {exc}",
      file=sys.stderr,
  )
  raise

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from repo root before reading any secrets.
load_dotenv(REPO_ROOT / ".env")

# --------------------------------------------------------------------------
# Pinned snapshot + pricing (2026-09-16). Update this block only with a new
# dated decision entry in README.md's Decisions log -- never silently swap
# the snapshot.
# --------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PINNED_MODEL_SLUG = "anthropic/claude-haiku-4.5"  # OpenRouter model slug
PINNED_PROVIDER = "anthropic"  # pin to Anthropic's own endpoint, not resellers
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00
# Phase 0 gate only needs frontier CoT; do not spend on other arms yet.
ARM = "cot"

EVAL_SET_PATHS: dict[str, Path] = {
    "finqa_test": REPO_ROOT / "data" / "processed" / "finqa_test.parquet",
    "custom_eval": REPO_ROOT / "data" / "eval" / "custom_eval.parquet",
    "sec_2026": REPO_ROOT / "data" / "eval" / "sec_2026_eval.parquet",
}

MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 1.5
MAX_BACKOFF_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 120


def sanitize_model_name(model_slug: str) -> str:
  return model_slug.replace("/", "--").replace(".", "-")


def output_path(eval_set: str) -> Path:
  return RESULTS_DIR / f"{sanitize_model_name(PINNED_MODEL_SLUG)}__{ARM}__{eval_set}.jsonl"


def manifest_path() -> Path:
  return RESULTS_DIR / f"manifest_frontier__{sanitize_model_name(PINNED_MODEL_SLUG)}.json"


def load_done_ids(path: Path) -> set[str]:
  if not path.exists():
    return set()
  done: set[str] = set()
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
      except json.JSONDecodeError:
        continue
      if "id" in rec:
        done.add(str(rec["id"]))
  return done


def load_eval_df(eval_set: str) -> pd.DataFrame:
  path = EVAL_SET_PATHS[eval_set]
  if not path.exists():
    raise FileNotFoundError(
        f"Frozen eval set not found at {path}. Do not regenerate eval data "
        "for this task -- confirm the path/build step from handover.md."
    )
  return pd.read_parquet(path)


def approx_token_count(text: str) -> int:
  """Cheap, model-agnostic estimate for --estimate mode only (no API call).
  Actual billed tokens come from the real response usage block."""
  return max(1, round(len(text) / 3.6))


@dataclass
class CallResult:
  output_text: str
  prompt_tokens: int
  completion_tokens: int
  latency_seconds: float
  finish_reason: str
  provider_used: str
  generation_id: str


def call_frontier(
    api_key: str, prompt: str, max_tokens: int, temperature: float = 0.0
) -> CallResult:
  """Single retrying call to the pinned OpenRouter model+provider,
  CoT text mode (no reasoning param), deterministic decoding."""
  headers = {
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/iashu2k/opentune",
      "X-Title": "OpenTune Phase 0 frontier baseline",
  }
  payload = {
      "model": PINNED_MODEL_SLUG,
      "messages": [{"role": "user", "content": prompt}],
      "temperature": temperature,
      "max_tokens": max_tokens,
      "provider": {"order": [PINNED_PROVIDER], "allow_fallbacks": False},
  }

  attempt = 0
  while True:
    attempt += 1
    start = time.monotonic()
    try:
      resp = requests.post(
          OPENROUTER_API_URL,
          headers=headers,
          json=payload,
          timeout=REQUEST_TIMEOUT_SECONDS,
      )
      latency = time.monotonic() - start

      if resp.status_code >= 400:
        retryable = resp.status_code in (429, 500, 502, 503, 529)
        if not retryable or attempt >= MAX_RETRIES:
          resp.raise_for_status()
        backoff = min(MAX_BACKOFF_SECONDS,
                      BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        backoff *= 1.0 + random.uniform(-0.2, 0.2)
        print(
            f"  [retry] attempt {attempt}/{MAX_RETRIES} after HTTP "
            f"{resp.status_code}; sleeping {backoff:.1f}s",
            file=sys.stderr,
        )
        time.sleep(backoff)
        continue

      data = resp.json()
      choice = data["choices"][0]
      output_text = choice["message"]["content"] or ""
      usage = data.get("usage", {}) or {}
      return CallResult(
          output_text=output_text,
          prompt_tokens=usage.get("prompt_tokens", 0) or 0,
          completion_tokens=usage.get("completion_tokens", 0) or 0,
          latency_seconds=latency,
          finish_reason=choice.get("finish_reason", "") or "",
          provider_used=data.get(
            "provider", PINNED_PROVIDER) or PINNED_PROVIDER,
          generation_id=data.get("id", "") or "",
      )

    except requests.exceptions.RequestException as exc:
      if attempt >= MAX_RETRIES:
        raise
      backoff = min(MAX_BACKOFF_SECONDS,
                    BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
      backoff *= 1.0 + random.uniform(-0.2, 0.2)
      print(
          f"  [retry] attempt {attempt}/{MAX_RETRIES} after network error "
          f"({type(exc).__name__}: {exc}); sleeping {backoff:.1f}s",
          file=sys.stderr,
      )
      time.sleep(backoff)


def run_eval_set(api_key: str, eval_set: str, max_tokens: int, temperature: float) -> None:
  df = load_eval_df(eval_set)
  out_path = output_path(eval_set)
  done_ids = load_done_ids(out_path)
  todo = df[~df["id"].astype(str).isin(done_ids)]
  print(
      f"[{eval_set}] {len(df)} total, {len(done_ids)} already done, "
      f"{len(todo)} remaining -> {out_path}"
  )
  if len(todo) == 0:
    print(f"[{eval_set}] cell already complete")
    return

  manifest = load_manifest()
  cumulative_cost = manifest.get("cumulative_cost_usd", 0.0)

  with out_path.open("a", encoding="utf-8") as f:
    for _, row in todo.iterrows():
      example_id = str(row["id"])

      prompt = render_prompt(ARM, row["question"], row["context"])  # frozen
      call = call_frontier(
        api_key, prompt, max_tokens=max_tokens, temperature=temperature)

      # frozen: extracts + scores
      r = score(call.output_text, str(row["gold"]))

      cost_usd = (
          call.prompt_tokens / 1_000_000 * PRICE_IN_PER_MTOK
          + call.completion_tokens / 1_000_000 * PRICE_OUT_PER_MTOK
      )
      cumulative_cost += cost_usd

      out_rec = {
          # -- fields matching run_baselines.py's schema exactly --
          "id": example_id,
          "arm": ARM,
          "eval_set": eval_set,
          "model": PINNED_MODEL_SLUG,
          "output": call.output_text,
          "gold": str(row["gold"]),
          "correct": bool(r.correct),
          "status": r.extraction.status.value,
          "extracted": str(r.extraction.value) if r.extraction.value is not None else None,
          "n_prompt_tokens": call.prompt_tokens,
          "latency_s": round(call.latency_seconds, 3),
          "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
          # -- frontier-only bookkeeping (extra keys, nothing renamed) --
          "api_model_slug": PINNED_MODEL_SLUG,
          "provider_pinned": PINNED_PROVIDER,
          "provider_used": call.provider_used,
          "generation_id": call.generation_id,
          "completion_tokens": call.completion_tokens,
          "finish_reason": call.finish_reason,
          "cost_usd": round(cost_usd, 6),
          "cumulative_cost_usd": round(cumulative_cost, 4),
          "temperature": temperature,
      }
      f.write(json.dumps(out_rec) + "\n")
      f.flush()
      done_ids.add(example_id)

      manifest["cumulative_cost_usd"] = cumulative_cost
      manifest["last_updated"] = out_rec["ts"]
      manifest.setdefault("eval_sets", {})[eval_set] = {
          "n_total": len(df),
          "n_done": len(done_ids),
      }
      save_manifest(manifest)

  print(f"[{eval_set}] done. cumulative cost so far: ${cumulative_cost:.4f}")


def load_manifest() -> dict[str, Any]:
  path = manifest_path()
  if path.exists():
    return json.loads(path.read_text(encoding="utf-8"))
  return {
      "api_model_slug": PINNED_MODEL_SLUG,
      "provider_pinned": PINNED_PROVIDER,
      "arm": ARM,
      "price_in_per_mtok": PRICE_IN_PER_MTOK,
      "price_out_per_mtok": PRICE_OUT_PER_MTOK,
      "cumulative_cost_usd": 0.0,
      "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
  }


def save_manifest(manifest: dict[str, Any]) -> None:
  manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def estimate(eval_sets: Iterable[str], max_tokens: int) -> None:
  print(
      f"--estimate mode: no API calls made. Pinned model: {PINNED_MODEL_SLUG} "
      f"(provider pinned: {PINNED_PROVIDER})\n"
  )
  total_in = total_out_at_cap = 0
  for eval_set in eval_sets:
    df = load_eval_df(eval_set)
    in_tokens = 0
    for _, row in df.iterrows():
      prompt = render_prompt(ARM, row["question"], row["context"])
      in_tokens += approx_token_count(prompt)
    out_tokens_cap = len(df) * max_tokens
    total_in += in_tokens
    total_out_at_cap += out_tokens_cap
    cost_in = in_tokens / 1_000_000 * PRICE_IN_PER_MTOK
    cost_out_cap = out_tokens_cap / 1_000_000 * PRICE_OUT_PER_MTOK
    print(
        f"  {eval_set:12s} n={len(df):5d}  "
        f"approx_input_tokens={in_tokens:9,d}  "
        f"worst_case_output_tokens={out_tokens_cap:9,d}  "
        f"worst_case_cost=${cost_in + cost_out_cap:6.2f}"
    )
  cost_in_total = total_in / 1_000_000 * PRICE_IN_PER_MTOK
  cost_out_total_cap = total_out_at_cap / 1_000_000 * PRICE_OUT_PER_MTOK
  print(
      f"\n  TOTAL approx_input_tokens={total_in:,}  "
      f"worst_case_output_tokens={total_out_at_cap:,}\n"
      f"  TOTAL worst_case_cost=${cost_in_total + cost_out_total_cap:.2f}  "
      "(actual cost will track real completion length, typically well under this cap)"
  )


def get_api_key() -> str:
  api_key = os.environ.get("OPENROUTER_API_KEY")
  if not api_key:
    print(
        "FATAL: OPENROUTER_API_KEY not found. Create a .env file in the repo "
        "root (never commit it -- add `.env` to .gitignore) containing:\n\n"
        "    OPENROUTER_API_KEY=sk-or-v1-...\n",
        file=sys.stderr,
    )
    sys.exit(1)
  return api_key


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Resumable frontier CoT runner via OpenRouter (OpenTune Phase 0)."
  )
  parser.add_argument(
      "--eval-set",
      choices=["finqa_test", "custom_eval", "sec_2026", "all"],
      default="all",
      help="Which frozen eval set(s) to run CoT on.",
  )
  parser.add_argument(
      "--max-tokens",
      type=int,
      default=700,
      help="Max completion tokens per call (slightly above the local 640 cap "
      "for safety margin; frontier models ramble less under a clean CoT+"
      "ANSWER-line contract).",
  )
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument(
      "--estimate",
      action="store_true",
      help="Zero-cost dry pass: projects input/output tokens and worst-case "
      "$ cost without calling the API. Run this before spending money.",
  )
  args = parser.parse_args()

  eval_sets = list(EVAL_SET_PATHS.keys()) if args.eval_set == "all" else [
      args.eval_set]

  if args.estimate:
    estimate(eval_sets, args.max_tokens)
    return

  api_key = get_api_key()
  for eval_set in eval_sets:
    run_eval_set(api_key, eval_set, max_tokens=args.max_tokens,
                 temperature=args.temperature)


if __name__ == "__main__":
  main()
