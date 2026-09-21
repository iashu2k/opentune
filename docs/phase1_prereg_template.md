# OpenTune — Phase 1 Pre-Registration

**Status:** PRE-REGISTERED — committed before any Phase 1 training data, training
code, or SFT results existed.
**Date written:** 2026-09-21
**Commit:** 
**Baseline anchor:** (hash: 7e75a2dfb6ba2ba2ad23611cbfffa7863520d3e6) — all Phase 1 claims are
measured against results reproducible from that tag.

---

## 1. Fixed anchors (constants, from Phase 0)

These numbers are inputs to the criteria below. They are not updated, revised,
or recomputed in this phase.

| Quantity | Value | Source |
|---|---|---|
| Qwen3-8B CoT, finqa_test (n=1,127) | 65.57% [62.82, 68.32] | `docs/baselines.md` @ `phase0-closed` |
| Haiku 4.5 CoT, finqa_test | 70.90% [68.23, 73.56] | `docs/gate_decision.md` @ `phase0-closed` |
| Measured gap (frontier − base) | +5.32 pts | idem |
| Qwen3-8B CoT, sec_2026_eval (n=180) | 73.33% | `docs/baselines.md` @ `phase0-closed` |
| Qwen3-8B CoT contract-failure rate, finqa_test | 6.39% (72/1127) | recomputed from results/raw/Qwen3-8B__cot__finqa_test.jsonl via stored status field |

Statistics protocol (identical to Phase 0): paired bootstrap 95% CIs with
10,000 resamples, seed 0; exact McNemar via `scipy.stats.binomtest`.

## 2. Primary criterion (gate)

The SFT model (evaluated zero-shot, greedy, temperature=0) is a Phase 1
success if **both** hold on finqa_test (n=1,127):

- (a) Paired point estimate: Δ(SFT − Qwen3-8B CoT) ≥ **+3.2 pts**
  — this equals ≥60% closure of the 5.32-pt gap → SFT ≥ 68.77%
- (b) Paired bootstrap 95% CI on that delta has lower bound **> 0**

Rationale: the Phase 0 gate failed on the original "beat frontier by 8+ pts"
framing; the logged decision reframed the target as closing most of the
measured +5.32-pt gap. "Most" is operationalized here as ≥60% of the point
estimate plus a CI that excludes zero — significance alone would pass a
+0.5-pt improvement, which closes 9% of the gap and supports no claim.

## 3. Secondary (reported, not gating)

- Exact McNemar p, SFT vs Qwen3-8B CoT (paired).
- Paired delta SFT vs Haiku 4.5 CoT on finqa_test (sign-agnostic; narrative).
- SFT performance on custom_eval (n=450) with CI.

## 4. Safety floors (gating)

- **S1 (generalization):** SFT accuracy on sec_2026_eval must not be more
  than **2 pts** below the Qwen3-8B CoT anchor (73.33%), judged on point
  estimate only. Point-estimate-only is registered deliberately: n=180 makes
  CI-gated regressions a noise lottery on either side.
- **S2 (contract compliance):** SFT combined no_answer_line + non_numeric_answer rate on finqa_test ≤ 7.39% (72/1127 + 1.0 pt). SFT on
  contract-formatted data should reduce parse failures; an increase indicates
  a training-format bug and invalidates the run for gating purposes.

## 5. Eval protocol (frozen for this phase)

- Prompt arm: `zero_shot` via frozen `render_prompt(name, question, context)`
  (separate string args). Byte-identical context serialization to
  `run_baselines.py`.
- Decoding: greedy, temperature 0. Same runner pattern as Phase 0; SFT
  checkpoint is the only changed input.
- Scoring: frozen `extract.py` `score(text, gold_str)` — no Phase 1 edits.
- Eval sets: frozen `finqa_test` (n=1,127), `custom_eval` (n=450),
  `sec_2026_eval` (n=180), read-only.
- Output: `results/raw/` JSONL + manifest, same schema as Phase 0 runs; stats
  via new `scripts/compare_sft.py` (Phase 0 tools untouched).

## 6. Training-data commitments

- Base: `data/processed/finqa_train_decontaminated.parquet` (5,828 rows) —
  read-only.
- Trace construction: STaR-style rejection sampling from Qwen3-8B CoT on
  Kaggle, k=4, keep completions where frozen `score()`
  returns `correct == True` and `status == OK`. Questions with zero correct
  samples fall back to deterministic template traces rendered from gold
  `program`/`exe_ans`; fallback count logged.
- Validation: **every** final training completion passes frozen `score()`
  to `correct == True`; parse failures are excluded, never repaired by hand.
- Output artifact: `data/processed/sft_train_v1.parquet` (new, versioned),
  plus `docs/sft_data_card.md` (sizes, acceptance rate, k, operator
  distribution vs §6-of-handover inventory, fallback count).
- Synthetic SEC examples: **deferred to iteration 2**, used only if the
  primary criterion fails on data-quality grounds; capped at 30–40% of mix
  and passed through the same decontamination gate if used at all.

## 7. Training-run policy

- Stack: Unsloth QLoRA + TRL `SFTTrainer`, prompt-completion format with
  completion-only loss; single GPU (`CUDA_VISIBLE_DEVICES=0`).
- Max **3** SFT runs, in this exact order per the project plan:
  1. v1 data/config as committed §6.
  2. data-quality or trace-composition fix (requires logged evidence for
     what was wrong with v1 data, not vibes).
  3. at most one hyperparameter pass (lr and/or LoRA rank), only if run 2
     criteria-fail is plausibly optimization-limited.
- Every run: `max_steps` hard cap in config, W&B logging to project
  `opentune-phase1`, checkpoints pushed to
  `iashu2k/opentune-qwen3-8b-sft` (private) at least every 500 steps and at
  session end.
- Per-run experiment-log entry regardless of outcome (kept/rolled back).

## 8. Budget commitments

- Spent to date: $3.9994 (Phase 0 frontier API).
- Phase 1 target: $0–12; free-tier Kaggle primary, single RunPod run ≤$5 as
  the only anticipated paid path; alert at **$30 total project**.
- Flagship-tier frontier re-test (~$15–20) remains deferred per the 2026-09-16
  decision; revisit only after Phase 1 results.

## 9. Failure handling (pre-committed)

If the primary criterion fails after the iteration budget:

1. Verify no format regression (S2) or data bug first, per §7 order.
2. Publish the negative result with root-cause analysis in the README
   experiment log and `docs/sft_results.md` — the Living README policy
   requires negative results stay in.
3. Allowed pivots, in preference order: (a) raise synthetic SEC share within
   the 30–40% cap; (b) consider k>4 rejection sampling; (c) accept the
   Phase 1 negative finding and carry the SFT checkpoint forward as the
   Phase 2 GRPO warm-start anyway, with the gap closure now a Phase 2 target.
   (d) is explicit non-option: **no re-defining §2 after results exist.**

## 10. Sign-off

Written and committed before any Phase 1 training artifact.
- Date: 2026-09-21
- Commit: FILL-IN
- Author: iashu2k
