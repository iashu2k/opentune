"""Baseline evaluation runner v2 (Kaggle/Colab GPU).

Changes vs v1:
  - stop_strings cut generation at the format boundary (new section header or
    a second ANSWER line). Rationale: dry-run showed greedy decode rambling to
    the 640-token cap after the answer; a second ANSWER line in the tail would
    override the real one under task-spec last-line-wins.
  - batched generation (left padding) for T4 throughput; ~52 s/row serial was
    projected at ~77 GPU-hours for the full matrix — far over weekly quota.

Resumable: predictions append to results/raw/{model}__{arm}__{evalset}.jsonl;
reruns skip completed ids.
"""
from __future__ import annotations
import os
from opentune.prompts.templates import render_prompt
from opentune.extract import score
import pandas as pd
from pathlib import Path
import time
import json
import argparse

# see experiment log: multi-GPU mask bug
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


RESULTS_DIR = Path("results/raw")
MAX_NEW_TOKENS = 640
STOP_STRINGS = ["\n### ", "\n\nANSWER:", "\nOkay,"]

EVAL_SETS = {
    "custom_eval": "data/eval/custom_eval.parquet",
    "finqa_test": "data/processed/finqa_test.parquet",
    "sec_2026": "data/eval/sec_2026_eval.parquet",
}


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


def completed_ids(out_path: Path) -> set[str]:
  if not out_path.exists():
    return set()
  with open(out_path) as f:
    return {json.loads(line)["id"] for line in f if line.strip()}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen3-8B")
  ap.add_argument(
    "--arm", choices=["zero_shot", "cot", "few_shot"], required=True)
  ap.add_argument("--eval-set", choices=EVAL_SETS.keys(), required=True)
  ap.add_argument("--batch-size", type=int, default=8)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--wandb", action="store_true")
  args = ap.parse_args()

  run_id = f"{args.model.split('/')[-1]}__{args.arm}__{args.eval_set}"
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  out_path = RESULTS_DIR / f"{run_id}.jsonl"

  df = pd.read_parquet(EVAL_SETS[args.eval_set])
  if args.limit:
    df = df.head(args.limit)

  done = completed_ids(out_path)
  todo = df[~df["id"].isin(done)]
  print(f"{run_id}: {len(done)} done, {len(todo)} to go")
  if not len(todo):
    print("cell already complete")
    return

  model, tokenizer = load_model(args.model)

  wb = None
  if args.wandb:
    import wandb

    wb = wandb.init(
        project="opentune-phase0",
        name=run_id,
        config={"arm": args.arm, "eval_set": args.eval_set, "model": args.model,
                "decode": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
                "stop_strings": STOP_STRINGS, "batch_size": args.batch_size,
                "quantization": "nf4-4bit", "resume_n_done": len(done)},
    )

  rows = list(todo.iterrows())
  n_correct, n_done = 0, 0
  with open(out_path, "a") as f:
    for i in range(0, len(rows), args.batch_size):
      batch = rows[i: i + args.batch_size]
      prompts = [render_prompt(args.arm, r["question"], r["context"])
                 for _, r in batch]
      inputs = tokenizer(prompts, return_tensors="pt",
                         padding=True).to(model.device)
      t0 = time.time()
      out = model.generate(
          **inputs,
          max_new_tokens=MAX_NEW_TOKENS,
          do_sample=False,
          stop_strings=STOP_STRINGS,
          tokenizer=tokenizer,
          pad_token_id=tokenizer.pad_token_id,
      )
      latency = time.time() - t0
      new = out[:, inputs["input_ids"].shape[1]:]
      texts = tokenizer.batch_decode(new, skip_special_tokens=True)

      for (_, row), text in zip(batch, texts):
        r = score(text, str(row["gold"]))
        n_correct += int(r.correct)
        n_done += 1
        f.write(json.dumps({
            "id": row["id"], "arm": args.arm, "eval_set": args.eval_set,
            "model": args.model, "output": text, "gold": str(row["gold"]),
            "correct": r.correct, "status": r.extraction.status.value,
            "extracted": str(r.extraction.value) if r.extraction.value is not None else None,
            "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
            "latency_s": round(latency / len(batch), 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")
      f.flush()

      if n_done % 25 < args.batch_size:
        print(f"  [{n_done}/{len(rows)}] running acc {n_correct / n_done:.3f}")
        if wb:
          wb.log({"running_acc": n_correct / n_done, "i": len(done) + n_done})

  acc = n_correct / n_done
  print(f"DONE {run_id}: acc on new rows {acc:.4f}")
  if wb:
    wb.summary["final_acc_new_rows"] = acc
    wb.finish()


if __name__ == "__main__":
  main()
