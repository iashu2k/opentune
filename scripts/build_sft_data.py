"""Build SFT training data via STaR-style rejection sampling from Qwen3-8B,
with deterministic template-trace fallback for questions with 0 correct
samples in k attempts.

v7: restored batched generation for k=1, matching run_baselines.py's proven
config exactly (batch_size=8, left-padded, greedy). Diagnostic history:
  - k=4 sampled: 52% star-accept (after is_valid_program fix) -- below the
    65.57% baseline, plausible given best-of-4 vs single greedy comparison.
  - k=1 sampled (t=0.7): 28% -- much lower, blamed on sampling noise vs.
    greedy at the time.
  - k=1 greedy, same head-of-file 50 rows: 30% -- barely moved. Ruled out
    sampling-vs-greedy as the dominant cause.
  - k=1 greedy, SHUFFLED random 50 rows: 24% -- ruled out head-of-file bias
    in the test rows too.
  - Remaining architectural difference vs. the baseline run that produced
    65.57%: this script has generated ONE question at a time (batch=1)
    since the OOM fix several iterations back. run_baselines.py always
    batches 8 questions with left-padding. At k=1 there is no multiplicative
    OOM risk (batch_size * k = 8 * 1 = 8, the baseline's own proven-safe
    config) -- so batching was never actually necessary to avoid here, and
    if Unsloth's generate path behaves differently for batch=1 vs padded
    batch=8, that would explain the persistent gap this v7 is testing for.

Usage:
    uv run python scripts/build_sft_data.py --k 1                    # batched greedy (recommended)
    uv run python scripts/build_sft_data.py --k 4 --temperature 0.7   # per-row sampled rejection sampling
    uv run python scripts/build_sft_data.py --k 1 --limit 50
    uv run python scripts/build_sft_data.py --merge-only
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import collections
import subprocess
import argparse
import time
import sys
import re
import json
import glob
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


TRAIN_PATH = "data/processed/finqa_train_decontaminated.parquet"
RESULTS_DIR = Path("results/raw")
OUT_PARQUET = Path("data/processed/sft_train_v1.parquet")
STATS_PATH = Path("docs/sft_data_card_stats.json")
MAX_NEW_TOKENS = 640
STOP_STRINGS = ["\n### ", "\n\nANSWER:", "\nOkay,"]
ARM = "cot"

OP_RE = re.compile(r"([a-z_]+)\(")


def detect_gpu_count() -> int:
  try:
    out = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
    )
    n = len([l for l in out.stdout.splitlines() if l.strip()])
    return max(n, 1)
  except Exception:
    return 1


def is_clean(text: str) -> bool:
  return text.count("ANSWER:") <= 1


def is_valid_program(text: str) -> bool:
  if text.count("Program:") != 1:
    return False
  m = re.search(r"Program:\s*(.+)", text)
  if not m:
    return False
  program_str = m.group(1).strip()
  steps = re.findall(r"[a-z_]+\([^()]*\)", program_str)
  reconstructed = ", ".join(steps)
  return bool(steps) and reconstructed == program_str


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


def completed_ids(path: Path) -> set[str]:
  if not path.exists():
    return set()
  with open(path) as f:
    return {json.loads(l)["id"] for l in f if l.strip()}


def merge_shards() -> dict:
  shard_files = sorted(
    glob.glob(str(RESULTS_DIR / "sft_build_trace_log_shard*.jsonl")))
  if not shard_files:
    print("no shard files found")
    return {}
  print(f"merging: {shard_files}")

  all_rows = []
  for fp in shard_files:
    with open(fp) as f:
      all_rows.extend(json.loads(l) for l in f if l.strip())

  ids_seen, deduped = set(), []
  for r in all_rows:
    if r["id"] in ids_seen:
      continue
    ids_seen.add(r["id"])
    deduped.append(r)

  out_df = pd.DataFrame(deduped)
  out_df.to_parquet(OUT_PARQUET, index=False)

  total = len(out_df)
  star_n = int((out_df["source"] == "star").sum())
  fallback_n = int((out_df["source"] == "template_fallback").sum())
  oom_n = int(out_df["oom"].sum()) if "oom" in out_df.columns else None
  op_stat = collections.Counter()
  for prog in out_df["program"]:
    op_stat.update(op_counts(prog))

  ts_span = None
  if "ts" in out_df.columns and total:
    ts_span = float(out_df["ts"].max() - out_df["ts"].min())

  stats = {
      "total_rows": total,
      "star_accepted": star_n,
      "template_fallback": fallback_n,
      "star_acceptance_rate": round(star_n / total, 4) if total else None,
      "oom_fallbacks": oom_n,
      "n_shards_merged": len(shard_files),
      "wall_clock_span_s_across_all_shards": ts_span,
      "operator_counts_in_source_rows": dict(op_stat),
  }
  STATS_PATH.write_text(json.dumps(stats, indent=2))
  print(f"merged {total} rows -> {OUT_PARQUET}")
  print(json.dumps(stats, indent=2))
  return stats


def _accept_or_fallback(cand_texts, row, score_fn):
  accepted = None
  n_rejected = 0
  for cand in cand_texts:
    r = score_fn(cand, str(row["gold"]))
    if not (r.correct and r.extraction.status.value == "ok"):
      continue
    if not (is_clean(cand) and is_valid_program(cand)):
      n_rejected += 1
      continue
    accepted = trim_after_answer(cand)
    break
  return accepted, n_rejected


def run_worker(args) -> None:
  import torch
  from opentune.prompts.templates import render_prompt
  from opentune.extract import score

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

  df = pd.read_parquet(TRAIN_PATH)
  df = df.sample(frac=1, random_state=42).reset_index(
    drop=True)  # shuffle, fixed seed
  if args.num_shards > 1:
    df = df.iloc[args.shard_id::args.num_shards].reset_index(drop=True)
  if args.limit:
    df = df.head(args.limit)

  suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
  out_jsonl = RESULTS_DIR / f"sft_build_trace_log{suffix}.jsonl"
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)

  done = completed_ids(out_jsonl)
  todo = df[~df["id"].isin(done)]
  print(
    f"shard {args.shard_id}/{args.num_shards}: {len(done)} done, {len(todo)} to go")
  if not len(todo):
    print("shard already complete")
    return

  model, tokenizer = load_model(args.model)

  do_sample = args.k > 1
  # k=1 (greedy): batch like run_baselines.py; k>1 (sampled): per-row (avoids batch*k OOM)
  use_batching = not do_sample
  batch_size = args.batch_size if use_batching else 1

  print(f"[shard {args.shard_id}] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"torch sees {torch.cuda.device_count()} device(s), "
        f"using: {torch.cuda.get_device_name(0)}, "
        f"decode: {'sampled t=' + str(args.temperature) if do_sample else 'greedy'}, "
        f"k={args.k}, batch_size={batch_size}")

  source_stat = collections.Counter()
  degenerate_rejected = 0
  oom_count = 0
  rows = list(todo.iterrows())

  with open(out_jsonl, "a") as f:
    for i in range(0, len(rows), batch_size):
      batch = rows[i: i + batch_size]
      prompts = [render_prompt(ARM, r["question"], r["context"])
                 for _, r in batch]

      oom_this_batch = False
      try:
        if use_batching:
          inputs = tokenizer(prompts, return_tensors="pt",
                             padding=True).to(model.device)
          out = model.generate(
              **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
              stop_strings=STOP_STRINGS, tokenizer=tokenizer,
              pad_token_id=tokenizer.pad_token_id,
          )
          new = out[:, inputs["input_ids"].shape[1]:]
          decoded = tokenizer.batch_decode(new, skip_special_tokens=True)
          # k=1 -> one candidate list per row
          per_row_candidates = [[d] for d in decoded]
        else:
          inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
          out = model.generate(
              **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
              temperature=args.temperature, num_return_sequences=args.k,
              stop_strings=STOP_STRINGS, tokenizer=tokenizer,
              pad_token_id=tokenizer.pad_token_id,
          )
          new = out[:, inputs["input_ids"].shape[1]:]
          per_row_candidates = [tokenizer.batch_decode(
            new, skip_special_tokens=True)]
      except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        per_row_candidates = [[] for _ in batch]
        oom_count += len(batch)
        oom_this_batch = True
        print(f"  [{i + len(batch)}] OOM on batch starting row "
              f"{batch[0][1]['id']} -- falling back to template for {len(batch)} row(s)")

      for (_, row), candidates in zip(batch, per_row_candidates):
        accepted, n_rej = _accept_or_fallback(candidates, row, score)
        degenerate_rejected += n_rej
        source = "star" if accepted is not None else "template_fallback"
        if accepted is None:
          accepted = template_trace(row["program"], row["gold"])
        source_stat[source] += 1

        f.write(json.dumps({
            "id": row["id"], "prompt": render_prompt(ARM, row["question"], row["context"]),
            "completion": accepted, "source": source, "gold": str(row["gold"]),
            "program": row["program"], "ts": time.time(), "oom": oom_this_batch,
        }) + "\n")
      f.flush()

      del inputs
      if 'out' in dir():
        del out
      torch.cuda.empty_cache()

      n_done = min(i + batch_size, len(rows))
      if n_done % (args.log_every) < batch_size or n_done == len(rows):
        print(f"  [{n_done}/{len(rows)}] star={source_stat['star']} "
              f"fallback={source_stat['template_fallback']} "
              f"degenerate_rejected={degenerate_rejected} oom_rows={oom_count}")

  print(f"shard {args.shard_id} done: {dict(source_stat)} oom_count={oom_count}")


def run_orchestrator(args) -> None:
  num_shards = args.num_shards or detect_gpu_count()
  print(f"orchestrator: launching {num_shards} shard(s)")

  if num_shards == 1:
    ns = argparse.Namespace(**vars(args), shard_id=0, num_shards=1)
    run_worker(ns)
    _finalize_single_shard()
    return

  procs = []
  for shard_id in range(num_shards):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(shard_id)
    cmd = [
        sys.executable, __file__,
        "--shard-id", str(shard_id), "--num-shards", str(num_shards),
        "--k", str(args.k), "--temperature", str(args.temperature),
        "--model", args.model, "--log-every", str(args.log_every),
        "--batch-size", str(args.batch_size),
    ]
    if args.limit:
      cmd += ["--limit", str(args.limit)]
    log_path = f"shard{shard_id}.log"
    print(f"  launching shard {shard_id} on GPU {shard_id} -> {log_path}")
    log_f = open(log_path, "w")
    p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    procs.append((p, log_f))

  for p, log_f in procs:
    p.wait()
    log_f.close()
    print(f"  shard exited with code {p.returncode}")

  merge_shards()


def _finalize_single_shard() -> None:
  path = RESULTS_DIR / "sft_build_trace_log.jsonl"
  if not path.exists():
    print("no trace log found")
    return
  rows = [json.loads(l) for l in open(path) if l.strip()]
  out_df = pd.DataFrame(rows)
  out_df.to_parquet(OUT_PARQUET, index=False)
  total = len(out_df)
  star_n = int((out_df["source"] == "star").sum())
  oom_n = int(out_df["oom"].sum()) if "oom" in out_df.columns else None
  op_stat = collections.Counter()
  for prog in out_df["program"]:
    op_stat.update(op_counts(prog))
  ts_span = float(out_df["ts"].max() - out_df["ts"].min()
                  ) if "ts" in out_df.columns and total else None
  stats = {
      "total_rows": total, "star_accepted": star_n,
      "template_fallback": total - star_n,
      "star_acceptance_rate": round(star_n / total, 4) if total else None,
      "oom_fallbacks": oom_n,
      "n_shards_merged": 1, "wall_clock_span_s_across_all_shards": ts_span,
      "operator_counts_in_source_rows": dict(op_stat),
  }
  STATS_PATH.write_text(json.dumps(stats, indent=2))
  print(json.dumps(stats, indent=2))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen3-8B")
  ap.add_argument("--k", type=int, default=1)
  ap.add_argument("--temperature", type=float, default=0.7)
  ap.add_argument("--batch-size", type=int, default=8)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--log-every", type=int, default=10)
  ap.add_argument("--num-shards", type=int, default=None)
  ap.add_argument("--shard-id", type=int, default=None)
  ap.add_argument("--merge-only", action="store_true")
  args = ap.parse_args()

  if args.merge_only:
    merge_shards()
    return

  if args.shard_id is not None:
    args.num_shards = args.num_shards or 1
    run_worker(args)
    return

  run_orchestrator(args)


if __name__ == "__main__":
  main()
