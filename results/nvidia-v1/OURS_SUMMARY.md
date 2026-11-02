# nvidia-v1 — our systems (LIVE, 2026-11-02)

Task: binary moderation under **NVIDIA's content-safety policy** (Nemotron Content Safety 3.5 taxonomy).
All numbers **LIVE** (our runs on Fireworks), 0 MOCK. Frontier systems in this folder (`frontier-*`) belong
to another worker and are not summarised here.

## Data

- `nvidia_test` = `data/nvidia/test.jsonl`, 3,340 rows, held out (never trained or calibrated on;
  `data/LEAKAGE.md`: 4 near-duplicates already removed from train). 590 English + 2,750 non-English
  (11 languages × 250) → `nvidia_test_non_english` slice.
- `disguised_v1` = `data/frozen/disguised_v1.jsonl`, 2,222 rows, 8 perturbations (frozen eval set).

## Systems

| system | what | where it ran |
|---|---|---|
| **`final`** | tuned NVIDIA Nemotron Nano 3 30B-A3B LoRA (`accounts/kldjalkdjwwpadjapdj/models/safety-one-nano-safety`, live-merge, 1× B200) + clean-up (stage-0 normalisation) + rules + router + gray-zone escalation to NVIDIA Nemotron 3 Ultra serverless = registry `full` via `scripts.nano_ladder:full` | nvidia_test: deployment `qd2gx85j` today. disguised_v1: **reused** from the earlier `frozen-v1` run (r4 receipts, not shipped here; same adapter/pipeline, deployment `ier7tjje`) |
| **`nano-untuned`** | base `accounts/fireworks/models/nemotron-nano-3-30b-a3b`, no adapter, same prompt + clean-up + rules + router, **no escalation** (`scripts.nano_ladder:nano-untuned` = `+router` wiring) | own 1× B200 deployment `o4ubjdmu`, both datasets |
| **`ultra-untuned`** | NVIDIA Nemotron 3 Ultra 550B-A55B zero-shot, serverless (`nvidia-ultra-zeroshot`) | nvidia_test today; disguised_v1 **reused** from the earlier `frozen-v1` run (`nvidia-ultra-zeroshot.jsonl`, not shipped here) |
| `specialist-only` (internal) | tuned Nano alone, no clean-up/rules/router/escalation | `qd2gx85j`, nvidia_test |
| `final-rawtext` (internal) | `final` with `router.provider_text: raw` = clean-up OFF (models see the original disguised text) | `qd2gx85j`, disguised_v1 |

## Headline (LIVE)

| system | dataset | n | err | acc | macro-F1 | unsafe recall | over-block | p50 ms | p95 ms | $/1M msgs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **final** | nvidia_test | 3,340 | 0 | 97.0 | **95.3** | 97.6 | 5.5 | 644 | 1,193 | **56** |
| nano-untuned | nvidia_test | 3,340 | 1 | 89.2 | 82.7 | 93.3 | 27.8 | 804 | 1,157 | 64 |
| ultra-untuned | nvidia_test | 3,340 | 0 | 88.6 | 84.5 | 86.7 | 3.3 | 710 | 1,594 | 108 |
| **final** | nvidia_test non-English | 2,750 | 0 | 97.8 | **95.9** | 98.6 | 6.4 | 652 | 1,190 | 56 |
| nano-untuned | nvidia_test non-English | 2,750 | 0 | 89.5 | 81.3 | 92.6 | 26.9 | 805 | 1,148 | 64 |
| ultra-untuned | nvidia_test non-English | 2,750 | 0 | 87.5 | 81.8 | 85.4 | 1.1 | 713 | 1,644 | 112 |
| **final** | disguised_v1 | 2,222 | 0 | 82.1 | 82.0 | 89.4 | 25.3 | 571 | 1,115 | **69** |
| nano-untuned | disguised_v1 | 2,222 | 11 | 66.1 | 63.3 | 94.0 | 62.2 | 775 | 1,086 | 63 |
| ultra-untuned | disguised_v1 | 2,222 | 0 | 82.0 | 82.0 | 83.8 | 19.9 | 713 | 1,013 | 148 |

Paired on the same rows (bootstrap 95 % CI, exact McNemar), `final` minus other, macro-F1 points:

| other | dataset | Δ macro-F1 | 95 % CI | McNemar p |
|---|---|---:|---|---:|
| ultra-untuned | nvidia_test | **+10.8** | [+9.4, +12.4] | 2.5e-54 |
| ultra-untuned | disguised_v1 | −0.0 | [−1.6, +1.6] | 0.96 |
| nano-untuned | nvidia_test | **+12.6** | [+11.0, +14.4] | 3.5e-49 |
| nano-untuned | disguised_v1 | **+18.7** | [+16.5, +20.8] | 7.8e-55 |
| specialist-only | nvidia_test | +0.2 | [−0.4, +0.7] | 0.73 |
| final-rawtext (clean-up OFF) | disguised_v1 | **+8.0** | [+6.4, +9.7] | 8.7e-21 |

Reading:
- **Fine-tuning is the win under NVIDIA's policy:** +12.6 macro-F1 over the same base model and
  +10.8 over the 18× larger Ultra zero-shot, at ~half Ultra's $/msg. Over-block falls from 27.8 %
  (untuned Nano) to 5.5 %. Ultra blocks less (3.3 %) but misses 13 % of unsafe content (recall 86.7).
- **Non-English holds** (95.9 macro-F1 on 2,750 rows in 11 languages); Ultra drops to 81.8 there.
- **Disguised text:** `final` ties Ultra zero-shot on macro-F1 (82.0 vs 82.0, CI spans 0) with higher
  recall (89.4 vs 83.8) and more over-block (25.3 vs 19.9), at 47 % of the cost.
- **Clean-up step is worth +8.0 macro-F1 on disguised_v1** (74.0 → 82.0). It is decisive on
  base64: with clean-up OFF, recall on `base64_wrap` collapses to 0.7 % (macro-F1 34.1 vs 82.6).
- Gray-zone escalation adds little on NVIDIA-policy data (+0.2, n.s.; escalates 3.2 % of rows, cuts
  over-block 7.3 → 5.5); it mattered on the mixed-policy frozen set (r4).

## Clean-up ablation and per-perturbation (disguised_v1, LIVE)

| system | slice | n | err | acc | macro-F1 | unsafe recall | over-block | p50 ms | p95 ms | esc % | $/1M msgs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final | disguised_v1 | 2,222 | 0 | 82.1 | 82.0 | 89.4 | 25.3 | 571 | 1,115 | 9.8 | 69 |
| final | disguised_v1:base64_wrap | 300 | 0 | 82.7 | 82.6 | 89.3 | 24.0 | 570 | 1,097 | 8.7 | 68 |
| final | disguised_v1:char_spacing | 291 | 0 | 84.9 | 84.8 | 90.3 | 20.5 | 564 | 1,139 | 10.3 | 70 |
| final | disguised_v1:emoji_sub | 138 | 0 | 74.6 | 73.5 | 84.6 | 38.3 | 581 | 1,020 | 8.0 | 68 |
| final | disguised_v1:homoglyph | 300 | 0 | 83.7 | 83.6 | 90.7 | 23.3 | 561 | 1,122 | 10.0 | 70 |
| final | disguised_v1:leetspeak | 300 | 0 | 79.7 | 79.5 | 88.0 | 28.7 | 589 | 1,089 | 12.7 | 69 |
| final | disguised_v1:mixed_case | 300 | 0 | 79.3 | 79.2 | 88.0 | 29.3 | 573 | 1,110 | 10.0 | 70 |
| final | disguised_v1:none | 300 | 0 | 83.7 | 83.6 | 90.7 | 23.3 | 569 | 1,083 | 8.7 | 68 |
| final | disguised_v1:zero_width | 293 | 0 | 84.3 | 84.2 | 91.1 | 22.4 | 569 | 1,103 | 9.2 | 69 |
| nano-untuned | disguised_v1 | 2,222 | 11 | 66.1 | 63.3 | 94.0 | 62.2 | 775 | 1,086 | 0.0 | 63 |
| nano-untuned | disguised_v1:base64_wrap | 300 | 2 | 68.0 | 66.3 | 92.0 | 56.0 | 754 | 1,027 | 0.0 | 62 |
| nano-untuned | disguised_v1:char_spacing | 291 | 2 | 67.7 | 65.8 | 93.1 | 57.5 | 757 | 1,080 | 0.0 | 62 |
| nano-untuned | disguised_v1:emoji_sub | 138 | 1 | 64.5 | 58.0 | 92.3 | 71.7 | 780 | 1,068 | 0.0 | 63 |
| nano-untuned | disguised_v1:homoglyph | 300 | 2 | 68.3 | 66.2 | 94.7 | 58.0 | 765 | 994 | 0.0 | 62 |
| nano-untuned | disguised_v1:leetspeak | 300 | 0 | 59.7 | 52.1 | 99.3 | 80.0 | 827 | 1,283 | 0.0 | 63 |
| nano-untuned | disguised_v1:mixed_case | 300 | 0 | 61.7 | 57.2 | 94.0 | 70.7 | 803 | 1,184 | 0.0 | 63 |
| nano-untuned | disguised_v1:none | 300 | 2 | 69.0 | 67.3 | 93.3 | 55.3 | 766 | 1,028 | 0.0 | 62 |
| nano-untuned | disguised_v1:zero_width | 293 | 2 | 69.3 | 67.9 | 92.5 | 53.7 | 755 | 973 | 0.0 | 62 |
| ultra-untuned | disguised_v1 | 2,222 | 0 | 82.0 | 82.0 | 83.8 | 19.9 | 713 | 1,013 | 0.0 | 148 |
| ultra-untuned | disguised_v1:base64_wrap | 300 | 0 | 68.3 | 68.3 | 73.3 | 36.7 | 719 | 1,061 | 0.0 | 229 |
| ultra-untuned | disguised_v1:char_spacing | 291 | 0 | 86.3 | 86.2 | 91.0 | 18.5 | 719 | 929 | 0.0 | 145 |
| ultra-untuned | disguised_v1:emoji_sub | 138 | 0 | 80.4 | 79.7 | 88.5 | 30.0 | 688 | 998 | 0.0 | 97 |
| ultra-untuned | disguised_v1:homoglyph | 300 | 0 | 83.0 | 83.0 | 84.0 | 18.0 | 710 | 1,028 | 0.0 | 173 |
| ultra-untuned | disguised_v1:leetspeak | 300 | 0 | 77.0 | 76.9 | 70.7 | 16.7 | 718 | 972 | 0.0 | 179 |
| ultra-untuned | disguised_v1:mixed_case | 300 | 0 | 86.0 | 86.0 | 88.0 | 16.0 | 719 | 942 | 0.0 | 126 |
| ultra-untuned | disguised_v1:none | 300 | 0 | 87.3 | 87.3 | 88.7 | 14.0 | 708 | 1,133 | 0.0 | 88 |
| ultra-untuned | disguised_v1:zero_width | 293 | 0 | 87.0 | 87.0 | 89.0 | 15.0 | 708 | 1,031 | 0.0 | 121 |
| final-rawtext | disguised_v1 | 2,222 | 0 | 74.0 | 74.0 | 76.6 | 28.6 | 629 | 1,152 | 11.3 | 68 |
| final-rawtext | disguised_v1:base64_wrap | 300 | 0 | 50.3 | 34.1 | 0.7 | 0.0 | 545 | 702 | 0.0 | 54 |
| final-rawtext | disguised_v1:char_spacing | 291 | 0 | 79.4 | 79.1 | 91.0 | 32.2 | 663 | 1,153 | 13.4 | 72 |
| final-rawtext | disguised_v1:emoji_sub | 138 | 0 | 75.4 | 73.8 | 88.5 | 41.7 | 641 | 1,043 | 6.5 | 59 |
| final-rawtext | disguised_v1:homoglyph | 300 | 0 | 75.3 | 75.0 | 87.3 | 36.7 | 690 | 1,292 | 17.7 | 80 |
| final-rawtext | disguised_v1:leetspeak | 300 | 0 | 63.0 | 62.1 | 78.0 | 52.0 | 700 | 1,262 | 20.3 | 81 |
| final-rawtext | disguised_v1:mixed_case | 300 | 0 | 81.0 | 80.9 | 88.7 | 26.7 | 624 | 1,155 | 12.7 | 68 |
| final-rawtext | disguised_v1:none | 300 | 0 | 85.0 | 84.9 | 92.7 | 22.7 | 629 | 1,063 | 7.7 | 62 |
| final-rawtext | disguised_v1:zero_width | 293 | 0 | 84.0 | 83.9 | 92.5 | 24.5 | 631 | 1,107 | 9.2 | 65 |
| ultra-untuned | disguised_v1 | 2,222 | -0.0 | [-1.6, +1.6] | 0.96 |
| nano-untuned | disguised_v1 | 2,222 | +18.7 | [+16.5, +20.8] | 7.8e-55 |
| final-rawtext | disguised_v1 | 2,222 | +8.0 | [+6.4, +9.7] | 8.7e-21 |

## nvidia_test, all systems incl. internal (LIVE)

| system | slice | n | err | acc | macro-F1 | unsafe recall | over-block | p50 ms | p95 ms | esc % | $/1M msgs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final | nvidia_test | 3,340 | 0 | 97.0 | 95.3 | 97.6 | 5.5 | 644 | 1,193 | 3.2 | 56 |
| final | nvidia_test_non_english | 2,750 | 0 | 97.8 | 95.9 | 98.6 | 6.4 | 652 | 1,190 | 2.7 | 56 |
| nano-untuned | nvidia_test | 3,340 | 1 | 89.2 | 82.7 | 93.3 | 27.8 | 804 | 1,157 | 0.0 | 64 |
| nano-untuned | nvidia_test_non_english | 2,750 | 0 | 89.5 | 81.3 | 92.6 | 26.9 | 805 | 1,148 | 0.0 | 64 |
| ultra-untuned | nvidia_test | 3,340 | 0 | 88.6 | 84.5 | 86.7 | 3.3 | 710 | 1,594 | 0.0 | 108 |
| ultra-untuned | nvidia_test_non_english | 2,750 | 0 | 87.5 | 81.8 | 85.4 | 1.1 | 713 | 1,644 | 0.0 | 112 |
| specialist-only | nvidia_test | 3,340 | 0 | 96.9 | 95.1 | 98.0 | 7.3 | 671 | 944 | 0.0 | 52 |
| specialist-only | nvidia_test_non_english | 2,750 | 0 | 97.9 | 95.9 | 99.2 | 9.1 | 682 | 966 | 0.0 | 52 |
|---|---|---:|---:|---|---:|
| ultra-untuned | nvidia_test | 3,340 | +10.8 | [+9.4, +12.4] | 2.5e-54 |

## Method notes

- Runner: `scripts/run_eval.py --concurrency 48` per system, one client box; latency = client-side
  wall per message at 48 in flight (includes queueing and, for `final`, the Ultra round trip on
  escalated rows). Runs on one deployment were sequential.
- **$ per 1M msgs** = measured per-token cost in the receipts (Ultra serverless $0.60/$2.40 per 1M
  tokens: zero-shot lane and `final`'s escalations) + allocated GPU time: $13/h (1× B200) × run
  wall-clock ÷ model calls in that run (`ours_timing.json`; e.g. `final` nvidia_test 3,340 calls in
  49.3 s → $53 / 1M calls). Reused frozen-v1 `final` rows use the r4 allocation ($60 / 1M calls).
  Spin-up/idle time is excluded (amortised away in steady-state serving).
- Errors: `nano-untuned` returned no `safe`/`unsafe` label on 12 rows (1 nvidia_test, 11
  disguised_v1; deterministic at temperature 0, not retried). They count as incorrect in every metric.
- `final` on disguised_v1 is the r4 receipt set; `final-rawtext` ran today on the same adapter with r5
  code. The only wiring difference is `provider_text`; r5's P(unsafe) parser fix changes only Ultra's
  recorded probs (the tuned lane's probs come from the same `nano_ladder` parser in both).
- Escalated share: `final` 3.2 % nvidia_test, 9.8 % disguised_v1; `final-rawtext` 11.3 %.
- No row of `nvidia_test` was routed to the rule gate (all model-decided).
- Reproduce tables ($0): `.venv/bin/python scripts/nvidia_summary.py` (writes `ours_summary.json`).

## Files

`final.jsonl`, `nano-untuned.jsonl`, `ultra-untuned.jsonl` (5,562 rows each = both datasets),
`specialist-only.jsonl` (3,340), `final-rawtext.jsonl` (2,222): compact receipts (provider response
bodies dropped; label, categories, probs, route, latency, cost, token usage kept; reused rows carry
`reused_from`). `ours_timing.json` (GPU allocation inputs), `ours_summary.json` (all slices + paired tests).
