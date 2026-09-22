# OpenTune — SFT Training Data Card (v1)

**Status:** TEMPLATE — fields marked FILL-IN require `scripts/build_sft_data.py`
to have been run. Do not commit with FILL-IN values present.
**Source artifact:** `data/processed/sft_train_v1.parquet`
**Built from:** `data/processed/finqa_train_decontaminated.parquet` (5,828 rows,
frozen, read-only)
**Pre-registration this card is accountable to:** `docs/phase1_prereg.md` §6

---

## 1. Provenance

| Field | Value |
|---|---|
| Base corpus | `finqa_train_decontaminated.parquet`, 5,828 rows |
| Decontamination method (inherited, unchanged) | 8-gram word-level containment, question-containment ≥0.5 drop rule, vs dev+test corpus |
| Build script | `scripts/build_sft_data.py` |
| Build date | FILL-IN |
| Git commit at build time | FILL-IN |

## 2. Trace construction

| Field | Value |
|---|---|
| Strategy | STaR-style rejection sampling from Qwen3-8B CoT (primary), template-from-gold-`program` fallback (secondary) |
| Samples per question (k) | 4 |
| Sampling temperature | FILL-IN |
| Acceptance rule | frozen `score(output, gold)` returns `correct == True` and `extraction.status == "ok"` |
| Anti-degeneration filter | Reject candidates where `output.count("ANSWER:") > 1` — see §6 rationale below |
| Rows with ≥1 accepted STaR sample | FILL-IN / 5,828 (FILL-IN %) |
| Rows falling back to template trace (0 correct in k=4) | FILL-IN / 5,828 (FILL-IN %) |
| Rows dropped entirely (fallback also unusable) | FILL-IN / 5,828 (FILL-IN %) |
| Final training set size | FILL-IN rows |

## 3. Output format compliance

Every row in `sft_train_v1.parquet` must independently satisfy the frozen
contract before inclusion — this is not a sampled spot-check, it's a
per-row gate.

| Check | Result |
|---|---|
| 100% of completions parse via frozen `extract_answer()` | FILL-IN (pass/fail) |
| 100% of completions score `correct == True` against gold | FILL-IN (pass/fail) |
| 0% of completions contain >1 `ANSWER:` line (post-filter) | FILL-IN (pass/fail) |
| Prompt arm used for training | `zero_shot` via frozen `render_prompt(name, question, context)` |
| Context serialization | Matches `run_baselines.py` byte-for-byte (verified: FILL-IN) |

## 4. Operator distribution (vs. Phase 0 inventory)

Reference distribution from decontaminated train (§6 of Phase 0→1 handover):
divide 4,175 | subtract 2,540 | add 1,480 | multiply 550 | table_average 92 |
table_max 48 | table_sum 34 | table_min 27 | exp 5. `power`/`greater` absent.

| Operator | Handover count | sft_train_v1 count | Δ vs. proportional expectation |
|---|---:|---:|---|
| divide | 4,175 | FILL-IN | FILL-IN |
| subtract | 2,540 | FILL-IN | FILL-IN |
| add | 1,480 | FILL-IN | FILL-IN |
| multiply | 550 | FILL-IN | FILL-IN |
| table_average | 92 | FILL-IN | FILL-IN |
| table_max | 48 | FILL-IN | FILL-IN |
| table_sum | 34 | FILL-IN | FILL-IN |
| table_min | 27 | FILL-IN | FILL-IN |
| exp | 5 | FILL-IN | FILL-IN |
| power / greater | 0 | FILL-IN (flag if nonzero — should not appear) | — |

**Finding (fill in after build):** does rejection sampling disproportionately
drop rare operators (table_min/table_max/exp/table_sum)? If the tail
operators shrink far below their already-small proportional share, log it
here rather than silently upsampling — it's a real property of what the base
model can solve, not a bug to hide.

## 5. Synthetic data

| Field | Value |
|---|---|
| Synthetic SEC-filing examples included in v1 | No — deferred to iteration 2 per prereg §6 |
| Cap if introduced later | 30–40% of mix, passed through same decontamination gate |

## 6. Known limitations / decisions

- **Degenerate-repetition baseline context:** Qwen3-8B CoT baseline
  predictions on finqa_test show a 28.39% (320/1127) rate of looped
  `Program:`/`ANSWER:` blocks (harmless to eval accuracy due to the frozen
  "last ANSWER line wins" rule, but undesirable as a training target). The
  anti-degeneration filter in §2 exists specifically to prevent this habit
  from propagating into the SFT model via STaR sampling from the same base
  model.
- **Template-fallback trace quality:** template traces are deterministic
  renderings of the gold `program` into short prose and are lower lexical
  diversity than STaR traces by construction — flagged, not treated as a
  defect, since they only fill true model-failure gaps.
- Any other build-time anomaly: FILL-IN.

## 7. Sign-off

- Built: FILL-IN (date)
- Verified against §3 checks: FILL-IN (pass/fail, all four)
- Commit: FILL-IN
