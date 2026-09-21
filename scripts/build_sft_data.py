"""Build SFT training data via STaR-style rejection sampling from Qwen3-8B,
with deterministic template-trace fallback for questions with 0 correct
samples in k attempts.

v2: fixed CUDA OOM from v1 — num_return_sequences=k combined with
batch_size=8 questions meant 32 concurrent sequences per generate() call,
which exceeded a single T4's 14.56GB budget on top of the 4-bit 8B model's
own footprint. Fix: process ONE question at a time (num_return_sequences=k
only, no cross-question batching), with cache clearing between questions.
This is slower per-row but was the actual bottleneck, not a nice-to-have.

Design decisions (see docs/phase1_prereg.md, amended arm=cot):
  - Generation arm: "cot" (matches the 65.57% baseline arm; keeps prompt/
    completion pairs honest).
  - Anti-degeneration filter: reject candidates with >1 "ANSWER:" line.
  - One accepted trace per question (first clean+correct candidate of k).
  - Frozen contract is sole acceptance authority: score() correct=True,
    status="ok".

Usage (Kaggle, single GPU):
    uv run python scripts/build_sft_data.py --limit 50 --k 4            # dry run
    uv run python scripts/build_sft_data.py --k 4                        # full run

Resumable via results/raw/sft_build_trace_log.jsonl (skip completed ids).
Stats -> docs/sft_data_card_stats.json.
"""
from __future__ import annotations
from opentune.extract import score
from opentune.prompts.templates import render_prompt
import torch
import pandas as pd
from pathlib import Path
import collections
import argparse
import re
import json
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


TRAIN_PATH = "data/processed/finqa_train_decontaminated.parquet"
OUT_JSONL = Path("results/raw/sft_build_trace_log.jsonl")
OUT_PARQUET = Path("data/processed/sft_train_v1.parquet")
STATS_PATH = Path("docs/sft_data_card_stats.json")
MAX_NEW_TOKENS = 640
STOP_STRINGS = ["\n### ", "\n\nANSWER:", "\nOkay,"]
ARM = "cot"

OP_RE = re.compile(r"([a-z_]+)\(")


def load_model(model_id: str):
  from unsloth import FastLanguageModel

  model, tokenizer = FastLanguageModel.from_pretrained(
      model_id, max_seq_length=8192, load_in_4bit=True
  )
  FastLanguageModel.for_inference(model)
  tokenizer.padding_side = "left"
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  return model, tokenizer


def completed_ids(path: Path) -> set[str]:
  if not path.exists():
    return set()
  with open(path) as f:
    return {json.loads(l)["id"] for l in f if l.strip()}


def is_clean(text: str) -> bool:
  return text.count("ANSWER:") <= 1


def trim_after_answer(text: str) -> str:
  idx = text.find("ANSWER:")
  if idx == -1:
    return text
  end = text.find("\n", idx)
  return text[: end if end != -1 else len(text)].rstrip()


def template_trace(program: str, gold: str) -> str:
  steps = re.findall(r"([a-z_]+)\([^)]*\)", program)
  verb = {
      "divide": "divide the values", "subtract": "subtract the values",
      "add": "add the values", "multiply": "multiply the values",
      "table_average": "average the table values",
      "table_max": "take the table maximum", "table_min": "take the table minimum",
      "table_sum": "sum the table values", "exp": "apply the exponent",
  }
  reasoning = "Following the program, I " + "; then I ".join(
      verb.get(s, s) for s in steps
  ) + "."
  return f"{reasoning}\nProgram: {program}\nANSWER: {gold}"


def op_counts(program: str) -> collections.Counter:
  return collections.Counter(OP_RE.findall(program))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen3-8B")
  ap.add_argument("--k", type=int, default=4)
  ap.add_argument("--temperature", type=float, default=0.7)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--log-every", type=int, default=10)
  args = ap.parse_args()

  df = pd.read_parquet(TRAIN_PATH)
  if args.limit:
    df = df.head(args.limit)

  OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
  done = completed_ids(OUT_JSONL)
  todo = df[~df["id"].isin(done)]
  print(f"sft_build: {len(done)} done, {len(todo)} to go")
  if not len(todo):
    print("already complete")
    return

  model, tokenizer = load_model(args.model)

  op_stat = collections.Counter()
  source_stat = collections.Counter()
  degenerate_rejected = 0

  rows = list(todo.iterrows())
  with open(OUT_JSONL, "a") as f:
    for n, (_, row) in enumerate(rows, 1):
      prompt = render_prompt(ARM, row["question"], row["context"])
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

      try:
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=args.k,
            stop_strings=STOP_STRINGS,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = out[:, inputs["input_ids"].shape[1]:]
        candidates = tokenizer.batch_decode(new, skip_special_tokens=True)
      except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        candidates = []  # forces template fallback for this row
        print(f"  [{n}] OOM on row {row['id']} — falling back to template")

      accepted = None
      for cand in candidates:
        r = score(cand, str(row["gold"]))
        if not (r.correct and r.extraction.status.value == "ok"):
          continue
        if not is_clean(cand):
          degenerate_rejected += 1
          continue
        accepted = trim_after_answer(cand)
        break

      if accepted is not None:
        source = "star"
      else:
        accepted = template_trace(row["program"], row["gold"])
        source = "template_fallback"

      op_stat.update(op_counts(row["program"]))
      source_stat[source] += 1

      f.write(json.dumps({
          "id": row["id"], "prompt": prompt, "completion": accepted,
          "source": source, "gold": str(row["gold"]), "program": row["program"],
      }) + "\n")
      f.flush()

      del inputs
      if 'out' in dir():
        del out
      torch.cuda.empty_cache()

      if n % args.log_every == 0 or n == len(rows):
        print(f"  [{n}/{len(rows)}] star={source_stat['star']} "
              f"fallback={source_stat['template_fallback']} "
              f"degenerate_rejected={degenerate_rejected}")

  all_rows = [json.loads(l) for l in open(OUT_JSONL)]
  out_df = pd.DataFrame(all_rows)
  out_df.to_parquet(OUT_PARQUET, index=False)

  total = len(out_df)
  star_n = int((out_df["source"] == "star").sum())
  fallback_n = int((out_df["source"] == "template_fallback").sum())
  stats = {
      "total_rows": total,
      "star_accepted": star_n,
      "template_fallback": fallback_n,
      "star_acceptance_rate": round(star_n / total, 4) if total else None,
      "degenerate_candidates_rejected": degenerate_rejected,
      "k": args.k,
      "temperature": args.temperature,
      "arm": ARM,
      "operator_counts_in_source_rows": dict(op_stat),
  }
  STATS_PATH.write_text(json.dumps(stats, indent=2))
  print(json.dumps(stats, indent=2))


if __name__ == "__main__":
  main()
