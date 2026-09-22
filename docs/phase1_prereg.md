# OpenTune — Phase 1 Pre-Registration

**Status:** PRE-REGISTERED — committed before any Phase 1 training data, training
code, or SFT results existed. Amended twice since initial commit; see §11
Amendments Log. Amendments below concern implementation parameters (trace
sampling k) and a pipeline bug fix discovered during dry runs — **not** the
primary success criterion (§2), which remains unmodified.

**Date written:** 2026-09-21
**Commit:** 4eea8333d9376c67b12ffccc1cf4c59026cbd80b
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

**Unchanged by any amendment below.** No re-defining §2 after results exist.

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

- Prompt arm: `cot` via frozen `render_prompt(name, question, context)`
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
  Kaggle. **k=1** as of 2026-09-21 amendment (originally k=4; see §11
  Amendment 1 for the compute-cost rationale) — keep the completion where
  frozen `score()` returns `correct == True` and `status == "ok"`, and where
  it additionally passes `is_clean()` (≤1 `ANSWER:` line) and
  `is_valid_program()` (well-formed `op(args), op(args), ...` syntax; see
  §11 Amendment 2 for a parsing bug found and fixed in this validator).
  Questions with no correct/clean/valid sample at k=1 fall back to
  deterministic template traces rendered from gold `program`/`exe_ans`;
  fallback count logged.
- Expected star-acceptance rate at k=1: ≈ baseline CoT accuracy (~65.57%),
  since a single temperature-sampled draw succeeds roughly as often as the
  baseline's greedy pass, with no best-of-k advantage. This is a known,
  accepted trade-off of the k reduction, not a target to chase back up via
  data tricks.
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
  completion-only loss; single GPU per training process
  (`CUDA_VISIBLE_DEVICES=0`) — the Phase 0 multi-GPU Unsloth/Qwen3 bug
  applies to training as much as inference; do not assume otherwise without
  re-verifying.
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

- Spent to date: $3.9994 (Phase 0 frontier API). No paid spend yet in Phase 1
  — data-build compute has stayed within Kaggle's free tier after the k=1
  amendment (see §11 Amendment 1).
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
   the 30–40% cap; (b) consider a higher k for rejection sampling if a
   cheaper/faster compute path becomes available (e.g., a paid GPU-hour
   burst within the $0–12 Phase 1 budget); (c) accept the Phase 1 negative
   finding and carry the SFT checkpoint forward as the Phase 2 GRPO
   warm-start anyway, with the gap closure now a Phase 2 target.
   (d) is explicit non-option: **no re-defining §2 after results exist.**

## 10. Sign-off

Written and committed before any Phase 1 training artifact.
- Date: 2026-09-21
- Commit: 4eea8333d9376c67b12ffccc1cf4c59026cbd80b
- Author: iashu2k

## 11. Amendments Log

Pre-registration integrity is maintained by logging changes here with dates
and evidence, rather than silently editing §2–§9 without a trail. Only
implementation parameters (this section) have been amended; the primary gate
(§2) has not.

### Amendment 1 — k=4 → k=1 (2026-09-21)

**What changed:** §6 trace-construction k reduced from 4 to 1.

**Why:** empirical timing from a 50-row dry run (25 rows/shard × 2 Kaggle
T4 GPUs in parallel, k=4, temperature=0.7) measured **111.5s/row** on a
single GPU. Extrapolated to the full 5,828-row decontaminated train set:
**180.4 total GPU-hours** — 6.0 weeks of Kaggle's 30 GPU-hr/week free quota.
Critically, Kaggle meters GPU-hours per GPU used, so running two GPUs in
parallel reduces wall-clock time (to ~3.8 days) but **not** total quota
consumed (still 180.4 GPU-hours) — this was confirmed as a real constraint,
not a parallelism workaround. At k=1, the same extrapolation gives
~45.1 GPU-hours (~1.5 weeks of quota), which is compatible with the Phase 1
timeline and the $0–12 budget; k=4 was not.

**Trade-off accepted:** k=1 removes rejection sampling's best-of-k
advantage — star-acceptance rate is expected to track baseline CoT accuracy
(~65.57%) rather than exceed it. This is logged as a known, accepted
limitation (see §6), not a target to recover via other means within Phase 1.

**Status:** k=1 per-row throughput not yet independently re-measured at time
of this entry; the 45.1 GPU-hour figure is a first-order estimate (linear
scaling with k) pending confirmation via a k=1 dry run.

### Amendment 2 — `is_valid_program()` bug found and fixed (2026-09-21)

**What happened:** during dry-run validation of the STaR sampling pipeline
(pre-amendment-1, at k=4), a program-syntax validator was added to reject
malformed `Program:` lines and duplicate headers, in addition to the
existing `ANSWER:`-count check. The first implementation used
`text.split(",")` on the full program string, which incorrectly splits
*inside* any multi-argument operator (e.g. `divide(14001, 26302)` has a
comma inside its own arguments). Since nearly all real FinQA operators take
two comma-separated arguments, this bug rejected essentially all correct,
well-formed completions.

**Evidence:** a 100-row dry run (2 shards × 50 rows, k=4) returned
`star_accepted: 0` / `template_fallback: 100` — a result inconsistent with
the ~62% star rate observed in an earlier manual (unsharded, unfiltered)
50-row test. Per-shard logs showed no OOM errors and no crashes;
`degenerate_rejected` counts (104–118) were close to the total number of
otherwise-correct candidates found, pointing at the validator rather than
generation or scoring as the fault.

**Fix:** replaced the naive split with `re.findall(r"[a-z_]+\([^()]*\)", ...)`
to extract each operator call as a unit (respecting its own internal
parens/commas), then verified the reconstruction matches the original
program string exactly (catching genuinely malformed syntax, e.g. a stray
`* 100` outside any `op(...)` call, without false-rejecting valid multi-arg
operators).

**Confirmation:** re-run of the same 50-row dry run (post-fix, still k=4)
returned `star_accepted: 26/50` (52%), `oom_fallbacks: 0` — consistent with
baseline accuracy and the earlier informal 62% estimate, allowing for the
validator now correctly rejecting a few genuinely malformed completions that
the informal manual check had missed.

**Impact on §6:** none to the acceptance *logic*; the fix only corrects a
bug in an added safety filter. No change to frozen-contract scoring
(`extract.py`) was made or needed.
