"""
check_seq_lengths.py

Quick, dependency-light sanity check on prompt/target lengths in the
Phase 1 SFT dataset, to inform max_seq_length for the Unsloth/TRL training
config. Uses a rough chars-per-token approximation (3.7) since loading the
real Qwen3 tokenizer isn't necessary for a ballpark estimate -- rerun with
the real tokenizer later if the training script needs exact numbers.

Usage (from repo root):
    uv run python check_seq_lengths.py
"""

import json

CHARS_PER_TOKEN = 3.7  # rough approximation for English/financial text


def load(path):
  with open(path) as f:
    return [json.loads(l) for l in f]


def char_lens(rows):
  out = []
  for ex in rows:
    user_len = len(ex["messages"][0]["content"])
    asst_len = len(ex["messages"][1]["content"])
    out.append((user_len, asst_len, user_len + asst_len))
  return out


def pct(sorted_list, p):
  idx = int(len(sorted_list) * p)
  return sorted_list[min(idx, len(sorted_list) - 1)]


def report(name, arr):
  arr = sorted(arr)
  p50, p90, p95, p99, mx = pct(arr, 0.5), pct(
    arr, 0.9), pct(arr, 0.95), pct(arr, 0.99), arr[-1]
  print(f"{name}: p50={p50} p90={p90} p95={p95} p99={p99} max={mx}")
  print(
      f"  est. tokens (chars/{CHARS_PER_TOKEN}): "
      f"p50={p50 / CHARS_PER_TOKEN:.0f} p90={p90 / CHARS_PER_TOKEN:.0f} "
      f"p95={p95 / CHARS_PER_TOKEN:.0f} p99={p99 / CHARS_PER_TOKEN:.0f} "
      f"max={mx / CHARS_PER_TOKEN:.0f}"
  )


def main():
  train = load("data/processed/sft_train_v1.jsonl")
  val = load("data/processed/sft_val_v1.jsonl")
  all_rows = train + val
  print(
    f"train rows: {len(train)}, val rows: {len(val)}, total: {len(all_rows)}\n")

  lens = char_lens(all_rows)
  user_lens = [l[0] for l in lens]
  asst_lens = [l[1] for l in lens]
  total_lens = [l[2] for l in lens]

  report("user (prompt)", user_lens)
  report("assistant (target)", asst_lens)
  report("total (prompt + target)", total_lens)


if __name__ == "__main__":
  main()
