"""Baseline evaluation runner (Kaggle/Colab GPU).

Runs one (model, prompt_arm, eval_set) cell, resumably: predictions append to
results/raw/{model}__{arm}__{evalset}.jsonl; reruns skip completed ids.
Deterministic decoding (greedy), per-example records for CIs + McNemar.

Kaggle setup (launcher cell):
    !git clone https://github.com/iashu2k/opentune /kaggle/working/opentune
    !pip install -q unsloth
    %cd /kaggle/working/opentune
    !pip install -q -e . --no-deps   # deps (pandas/numpy) preinstalled on Kaggle

Run one cell:
    !python scripts/run_baselines.py --arm cot --eval-set custom_eval \
        --model Qwen/Qwen3-8B --wandb

Run all 9 cells (3 arms x 3 sets) as separate Kaggle sessions to survive quota.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from opentune.extract import score
from opentune.prompts.templates import render_prompt

RESULTS_DIR = Path("results/raw")
MAX_NEW_TOKENS = 640

EVAL_SETS = {
    "custom_eval": "data/eval/custom_eval.parquet",
    "finqa_test": "data/processed/finqa_test.parquet",
    "sec_2026": "data/eval/sec_2026_eval.parquet",
}


def load_model(model_id: str):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=8192,  # p95 context + few-shot overhead + CoT headroom
        load_in_4bit=True,  # NF4: same stack as Phase 1 SFT (decisions log)
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def completed_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with open(out_path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--arm", choices=["zero_shot", "cot", "few_shot"], required=True)
    ap.add_argument("--eval-set", choices=EVAL_SETS.keys(), required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test cap")
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

    if len(todo) == 0:
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
                    "quantization": "nf4-4bit", "resume_n_done": len(done)},
        )

    n_correct = 0
    with open(out_path, "a") as f:
        for j, (_, row) in enumerate(todo.iterrows(), 1):
            prompt = render_prompt(args.arm, question=row["question"], context=row["context"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            t0 = time.time()
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            latency = time.time() - t0
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

            r = score(text, str(row["gold"]))
            n_correct += int(r.correct)
            record = {
                "id": row["id"],
                "arm": args.arm,
                "eval_set": args.eval_set,
                "model": args.model,
                "output": text,
                "gold": str(row["gold"]),
                "correct": r.correct,
                "status": r.extraction.status.value,
                "extracted": str(r.extraction.value) if r.extraction.value is not None else None,
                "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
                "latency_s": round(latency, 3),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()  # survive session death mid-cell

            if wb and j % 25 == 0:
                wb.log({"running_acc": n_correct / j, "i": len(done) + j})
            if j % 25 == 0:
                print(f"  [{j}/{len(todo)}] running acc {n_correct / j:.3f}")

    acc = n_correct / len(todo) if len(todo) else 0.0
    print(f"DONE {run_id}: acc on new rows {acc:.4f}")
    if wb:
        wb.summary["final_acc_new_rows"] = acc
        wb.finish()


if __name__ == "__main__":
    main()
