"""
scripts/train_sft.py — OpenTune Phase 1: QLoRA SFT on Qwen3-8B

RUN THIS ON KAGGLE (GPU T4 x2 notebook), NOT ON YOUR MAC.
Your Mac has no CUDA GPU -- torch/unsloth are not installed there and
should not be. Per the project plan's hardware split: Mac = scripting/
data curation only; Kaggle/Colab = anything touching the 7-9B model.

Confirmed decisions baked into this script (2026-09-22):
  - GPU pinning is now a CLI flag (--multi-gpu), not hardcoded, so we can
    deliberately test whether Unsloth's known Qwen3 multi-GPU
    attention-mask bug (previously hit at INFERENCE time in Phase 0) also
    affects TRAINING. Default is single-GPU (safe, matches Phase 0 fix).
    Pass --multi-gpu to test 2-GPU training explicitly, per your request
    to spend one quick Kaggle session verifying this before committing to
    the single-GPU assumption long-term.
  - Qwen3's native "thinking mode" (<think>...</think>) is explicitly
    DISABLED (enable_thinking=False) when rendering the chat template.
    The task-spec's own reasoning -> Program: -> ANSWER: contract is the
    only reasoning format we want the model producing.
  - max_seq_length = 3072, chosen from real dataset percentiles
    (p95 ~=1955 tokens, p99 ~=2639, max ~=4616 by char-based estimate).
    Any example whose REAL tokenized length exceeds this is dropped and
    logged here (not truncated -- truncation risks cutting off the
    ANSWER line, which would corrupt the training signal).
  - Training data: data/processed/sft_train_v1.jsonl (5,351 rows) /
    data/processed/sft_val_v1.jsonl (282 rows, loss-monitoring only --
    NOT the pre-registered gate, which is evaluated separately on the
    frozen finqa_test/custom_eval/sec_2026 sets per docs/phase1_gate.md).
  - Loss is masked to assistant-turn tokens only (completion-only SFT),
    via Unsloth's train_on_responses_only helper.
  - W&B project: opentune-phase1.
  - HF Hub target: iashu2k/opentune-qwen3-8b-sft (confirmed 2026-09-22).
    Checkpoints pushed at the end of every epoch, so a Kaggle session
    disconnect never loses more than one epoch of progress.

Usage on Kaggle (after git clone + pip installs, GPU pin decided by flag):

    # Dry run, single-GPU (safe default), no Hub/W&B writes:
    python scripts/train_sft.py --max-train-samples 100 --epochs 1 \\
        --hub-repo-id "" --wandb-project ""

    # Dry run, MULTI-GPU test (checks the Unsloth Qwen3 bug at training time):
    python scripts/train_sft.py --multi-gpu --max-train-samples 100 --epochs 1 \\
        --hub-repo-id "" --wandb-project ""

    # Real run, single-GPU, full dataset:
    python scripts/train_sft.py \\
        --hub-repo-id iashu2k/opentune-qwen3-8b-sft \\
        --wandb-project opentune-phase1 --epochs 2
"""

import argparse
import json
import os


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--base-model", default="Qwen/Qwen3-8B")
  p.add_argument("--train-file", default="data/processed/sft_train_v1.jsonl")
  p.add_argument("--val-file", default="data/processed/sft_val_v1.jsonl")
  p.add_argument("--output-dir", default="outputs/sft_v1")
  p.add_argument("--max-seq-length", type=int, default=3072)
  p.add_argument("--lora-r", type=int, default=16)
  p.add_argument("--lora-alpha", type=int, default=16)
  p.add_argument("--lora-dropout", type=float, default=0.0)
  p.add_argument("--learning-rate", type=float, default=2e-4)
  p.add_argument("--per-device-batch-size", type=int, default=1)
  p.add_argument("--grad-accum-steps", type=int, default=8)
  p.add_argument("--epochs", type=int, default=2)
  p.add_argument("--max-steps", type=int, default=-1,
                 help="Hard cap on training steps; -1 = no cap beyond --epochs.")
  p.add_argument("--max-train-samples", type=int, default=-1,
                 help="For dry runs: subsample the train set to this many rows.")
  p.add_argument("--wandb-project", default="opentune-phase1")
  p.add_argument("--hub-repo-id", default="iashu2k/opentune-qwen3-8b-sft",
                 help="Empty string skips HF Hub push (dry runs).")
  p.add_argument(
    "--report", default="outputs/sft_v1/length_filter_report.json")
  p.add_argument("--multi-gpu", action="store_true",
                 help="If set, does NOT pin to a single GPU -- used to deliberately test "
                 "whether the Unsloth Qwen3 multi-GPU attention-mask bug (hit at "
                 "inference in Phase 0) also occurs during training. Default: off "
                 "(single-GPU pin, the safe assumption).")
  return p.parse_args()


def main():
  args = parse_args()

  # GPU pin decision happens BEFORE any torch/unsloth import, per the
  # Phase 0 lesson: the pin has no effect if set after CUDA context exists.
  if args.multi_gpu:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    print("MULTI-GPU TEST MODE: no CUDA_VISIBLE_DEVICES pin set. "
          "Watch closely for the attention-mask device-mismatch crash "
          "seen at inference time in Phase 0.")
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Single-GPU pin active: CUDA_VISIBLE_DEVICES=0")

  import torch
  from datasets import load_dataset
  from unsloth import FastLanguageModel
  from unsloth.chat_templates import get_chat_template, train_on_responses_only
  from trl import SFTTrainer, SFTConfig

  print(
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
  print(f"torch.cuda.device_count()={torch.cuda.device_count()}")

  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name=args.base_model,
      max_seq_length=args.max_seq_length,
      load_in_4bit=True,
      dtype=None,
  )

  tokenizer = get_chat_template(tokenizer, chat_template="qwen3")

  model = FastLanguageModel.get_peft_model(
      model,
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      lora_dropout=args.lora_dropout,
      target_modules=[
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj",
      ],
      bias="none",
      use_gradient_checkpointing="unsloth",
      random_state=42,
  )

  train_ds = load_dataset("json", data_files=args.train_file, split="train")
  val_ds = load_dataset("json", data_files=args.val_file, split="train")

  if args.max_train_samples > 0:
    train_ds = train_ds.select(
      range(min(args.max_train_samples, len(train_ds))))

  def render_and_filter(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # Qwen3 native thinking mode: OFF, per 2026-09-22 decision
    )
    n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return {"text": text, "n_tokens": n_tokens}

  train_ds = train_ds.map(render_and_filter)
  val_ds = val_ds.map(render_and_filter)

  over_limit_train = train_ds.filter(
    lambda ex: ex["n_tokens"] > args.max_seq_length)
  over_limit_val = val_ds.filter(
    lambda ex: ex["n_tokens"] > args.max_seq_length)

  dropped_ids = list(over_limit_train["id"]) + list(over_limit_val["id"])

  train_ds = train_ds.filter(lambda ex: ex["n_tokens"] <= args.max_seq_length)
  val_ds = val_ds.filter(lambda ex: ex["n_tokens"] <= args.max_seq_length)

  os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
  with open(args.report, "w") as f:
    json.dump({
        "max_seq_length": args.max_seq_length,
        "multi_gpu_test": args.multi_gpu,
        "train_rows_before_filter": len(train_ds) + len(over_limit_train),
        "train_rows_dropped_over_limit": len(over_limit_train),
        "val_rows_before_filter": len(val_ds) + len(over_limit_val),
        "val_rows_dropped_over_limit": len(over_limit_val),
        "dropped_ids": dropped_ids,
    }, f, indent=2)

  print(
    f"Train rows after length filter: {len(train_ds)} (dropped {len(over_limit_train)})")
  print(
    f"Val rows after length filter:   {len(val_ds)} (dropped {len(over_limit_val)})")

  report_to = "wandb" if args.wandb_project else "none"
  if report_to == "wandb":
    os.environ["WANDB_PROJECT"] = args.wandb_project

  sft_config = SFTConfig(
      output_dir=args.output_dir,
      per_device_train_batch_size=args.per_device_batch_size,
      gradient_accumulation_steps=args.grad_accum_steps,
      num_train_epochs=args.epochs,
      max_steps=args.max_steps,
      learning_rate=args.learning_rate,
      optim="adamw_8bit",
      lr_scheduler_type="cosine",
      warmup_ratio=0.03,
      logging_steps=10,
      eval_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=2,
      bf16=torch.cuda.is_bf16_supported(),
      fp16=not torch.cuda.is_bf16_supported(),
      report_to=report_to,
      dataset_text_field="text",
      max_seq_length=args.max_seq_length,
      packing=False,
      push_to_hub=bool(args.hub_repo_id),
      hub_model_id=args.hub_repo_id or None,
  )

  trainer = SFTTrainer(
      model=model,
      tokenizer=tokenizer,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      args=sft_config,
  )

  # Completion-only loss masking: only assistant-turn tokens contribute to
  # the loss. Instruction/response markers match the qwen3 chat template.
  trainer = train_on_responses_only(
      trainer,
      instruction_part="<|im_start|>user\n",
      response_part="<|im_start|>assistant\n",
  )

  trainer_stats = trainer.train()
  print(trainer_stats)

  final_dir = os.path.join(args.output_dir, "final")
  model.save_pretrained(final_dir)
  tokenizer.save_pretrained(final_dir)
  print(f"Final adapter saved to {final_dir}")

  if args.hub_repo_id:
    model.push_to_hub(args.hub_repo_id)
    tokenizer.push_to_hub(args.hub_repo_id)
    print(f"Pushed to HF Hub: {args.hub_repo_id}")


if __name__ == "__main__":
  main()
