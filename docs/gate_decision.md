# Phase 0 gate: frontier vs. Qwen3-8B CoT

Frontier model: `anthropic--claude-haiku-4-5` (Claude Haiku 4.5 via OpenRouter, provider pinned to Anthropic). Arm: CoT for both models. Bootstrap 95% CI, 10,000 resamples, seed=0. Exact McNemar via `scipy.stats.binomtest`.

| Eval set | n (paired) | Qwen3-8B CoT | Frontier CoT | Delta (pts) | CIs non-overlapping | McNemar b/c | McNemar p |
|---|---:|---|---|---:|---|---|---:|
| finqa_test | 1127 | 65.57% [62.82, 68.32] | 70.90% [68.23, 73.56] | +5.32 | no | 134/74 | 0.0000 |
| custom_eval | 450 | 62.89% [58.44, 67.33] | 68.89% [64.67, 73.12] | +6.00 | no | 52/25 | 0.0028 |
| sec_2026 | 180 | 73.33% [66.67, 79.44] | 66.11% [59.44, 72.78] | -7.22 | no | 19/32 | 0.0919 |

## Pre-registered gate outcome

Gate: frontier CoT - Qwen3-8B CoT >= 8 points on `finqa_test`, with non-overlapping bootstrap 95% CIs.

- Delta on finqa_test: **+5.32 points**
- CIs non-overlapping: **no**
- Exact McNemar p on finqa_test: **0.0000**
- **Gate result: FAIL**

Custom eval and SEC 2026 are corroborating evidence only; SEC 2026's n=180 gives wide CIs and was never pre-registered as an independent strict gate.

## Phase 1 framing decision

Gate failed on the pre-registered ≥8pt / non-overlapping-CI criteria, but the finqa_test delta is statistically significant (McNemar p=0.0000), just smaller than the pre-registered bar. Decision: proceed to SFT/GRPO with a revised, smaller target -- close most of the gap to near-frontier prompting (measured gap: +5.32 points on finqa_test) -- rather than reframing around cost/latency or spending further on a flagship-tier frontier snapshot. The near-frontier-tier caveat on the pinned model (`anthropic/claude-haiku-4.5`) still applies: this target is calibrated to the model actually measured, not to an untested flagship ceiling.
