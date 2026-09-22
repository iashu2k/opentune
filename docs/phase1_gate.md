# Phase 1 Gate — Pre-Registered Success Criterion

**Status: PRE-REGISTERED, NO RESULTS EXIST YET**
**Date logged:** 2026-09-22
**Logged by:** OpenTune project (see git commit hash for this file as proof of timing)
**Applies to:** SFT/QLoRA checkpoint(s) trained on `Qwen/Qwen3-8B`, evaluated on the frozen
`data/processed/finqa_test.parquet` (n=1,127), using the frozen extractor
(`src/opentune/extract.py`) and the frozen `cot` prompt template
(`src/opentune/prompts/templates.py`).

This document is written and committed **before** `build_sft_dataset.py`,
any training script, or any SFT checkpoint exists. It must not be edited
after SFT results are produced. If the criterion needs to change, log a new
dated entry in the README Decisions Log explaining why, and leave this file
untouched as the historical record — same practice as the Phase 0 gate.

---

## 1. Context this gate is reframing

Phase 0 pre-registered gate (frontier vs. base, best-effort prompting):
required frontier to beat Qwen3-8B CoT by ≥8 pts with non-overlapping CIs.

**Actual Phase 0 result on finqa_test (n=1,127):**

| Model | Accuracy | 95% CI |
|---|---|---|
| Qwen3-8B CoT (baseline) | 65.57% | [62.82, 68.32] |
| Claude Haiku 4.5 CoT (frontier reference) | 70.90% | [68.23, 73.56] |

Observed gap: **+5.32 points**, statistically significant (McNemar p<0.0001)
but CIs overlap and the gap is below the pre-registered 8-point bar →
Phase 0 gate **FAILED**. Decision: proceed to Phase 1 with a reframed,
smaller target (this document).

---

## 2. Phase 1 pre-registered target

> The SFT checkpoint must beat the Qwen3-8B CoT baseline (65.57% on
> finqa_test) by **at least 3.5 percentage points** (i.e., reach **≥69.07%**
> on finqa_test), **with non-overlapping bootstrap 95% confidence intervals**
> relative to the 65.57% [62.82, 68.32] baseline.

This target is chosen to represent closing **at least 60% of the measured
5.32-point gap** to the near-frontier CoT reference (70.90%):
3.5 / 5.32 ≈ 65.8%, comfortably inside the "close ≥60–70% of the gap"
range floated informally at Phase 0 close.

### Rationale for these exact numbers
- In-domain supervised fine-tuning on task-formatted data typically recovers
  a larger fraction of a prompting gap than generic instruction-following
  differences would predict, because SFT directly targets the output format
  and reasoning style the eval rewards — this is a defensible, not
  guaranteed, expectation, hence the buffer below 100% gap closure.
- Leaving ~35–40% of the gap unclosed after SFT alone is intentional: Phase 2
  (GRPO with verifiable rewards) is expected to close more of the remainder,
  so Phase 1 is not required to solve the entire gap by itself.
- 3.5 points is large enough that, given Phase 0's observed CI widths
  (~2.7-point half-width on this n=1,127 set), a true effect of this size is
  plausible to detect as statistically significant without requiring an
  implausibly large true effect.

---

## 3. Statistical methodology (must match Phase 0 exactly)

- **Point estimate:** accuracy = correct / n on `finqa_test` (n=1,127),
  using the frozen `score(text, gold_str)` extractor, deterministic decoding
  (temperature=0), same as the Phase 0 base-model CoT run.
- **Confidence intervals:** bootstrap 95% CI, same resampling procedure as
  `scripts/aggregate_results.py` / `scripts/compare_frontier.py` (reuse the
  existing bootstrap function, do not reimplement it).
- **Paired significance test:** McNemar's test, SFT vs. Qwen3-8B CoT
  baseline, on the same n=1,127 items, same procedure as
  `compare_frontier.py`'s paired comparison.
- **Gate passes only if both hold:**
  1. Point estimate ≥ 69.07% (i.e., delta ≥ +3.5 pts), AND
  2. The 95% CI for the SFT checkpoint does not overlap the 95% CI
     [62.82, 68.32] for the Qwen3-8B CoT baseline.
- **Secondary, non-gating reporting:** also report deltas and CIs on
  `data/eval/custom_eval.parquet` (n=450) and `data/eval/sec_2026_eval.parquet`
  (n=180) for context. These are informational only — noisier n, not part of
  the pass/fail decision — consistent with Phase 0 treating finqa_test as
  the sole gate set.

---

## 4. Decision branches (pre-committed, before results exist)

| Outcome | Definition | Action |
|---|---|---|
| **Full pass** | Delta ≥ +3.5 pts AND non-overlapping CIs | Ship SFT checkpoint to HF Hub, update README results table + decisions log, proceed to Phase 2 (GRPO) using this checkpoint as warm-start |
| **Partial pass** | Delta between +1.0 and +3.5 pts, regardless of CI overlap | Ship as an honest partial result, document explicitly as "did not clear the pre-registered bar," proceed to Phase 2 anyway — GRPO is expected to close more of the gap; do not retroactively lower this document's bar to call it a full pass |
| **Fail** | Delta < +1.0 pt, or negative, or CI entirely below baseline | Do NOT proceed to hyperparameter tuning first. Per the original project plan's own rule, diagnose SFT data quality first (noisy examples are the dominant SFT failure mode) — inspect a sample of generated training completions for factual/formatting errors before touching learning rate, rank, or epochs. Re-attempt at most 2–3 iterations total (time-boxed), then document as a negative result with root-cause analysis if still failing |

## 5. What must NOT change after this point

- The frozen task-spec, extractor, prompt templates, decontamination logic,
  and eval sets (finqa_test, custom_eval, sec_2026_eval) are unaffected by
  Phase 1 and must not be edited to make this gate easier to pass.
- This target (+3.5 pts / ≥69.07% / non-overlapping CIs) is not to be
  adjusted after seeing any SFT training or eval result. If the team later
  decides the target was wrong, that decision and its date go in the README
  Decisions Log as a new entry — this file stays as the historical record
  of what was pre-registered and when.
</content>