# Demo-video numbers (LIVE receipts)

**Built on Fireworks AI · NVIDIA Nemotron.**

Regenerate charts + this file: `.venv/bin/python docs/video/make_charts.py`

Dataset: `nvidia_test` held-out NVIDIA-policy rows (target n=3,340).
Refusals/errors scored as **incorrect**. Cost footnote: ours = GPU rental amortized at full utilization.
Frontier $/1M = (verdict receipt cost + spend-ledger cost of refusal rows) ÷ all n (`results/spend.json`).
Bill @100M = $/1M × 100.

| System | n | Macro-F1 (95% CI) | $/1M msgs | Bill @100M/mo | p50 ms | p95 ms | refused % | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| House Rules | 3340 | 95.3 [94.4, 96.2] | 56.2 | 5,620 | 644 | 1193 | 0.0 | `results/nvidia-v1/ours_summary.json → systems.final.nvidia_test` |
| Nano untuned | 3340 | 82.7 [81.0, 84.2] | 64.3 | 6,430 | 804 | 1157 | 0.0 | `results/nvidia-v1/ours_summary.json → systems.nano-untuned.nvidia_test` |
| Ultra 550B | 3340 | 84.5 [83.0, 85.9] | 107.9 | 10,790 | 710 | 1594 | 0.0 | `results/nvidia-v1/ours_summary.json → systems.ultra-untuned.nvidia_test` |
| GPT-6.1 Sol | 3340 | 71.5 [69.8, 73.2] | 1,819.64 | 181,964 | 3235 | 6066 | 1.9 | `results/nvidia-v1-frontier/frontier-router-a.jsonl (refusal=wrong; $/1M = verdicts + ledger refusals ÷ n)` |
| Opus 5.5 | 3340 | 69.0 [67.3, 70.7] | 6,045.24 | 604,524 | 2966 | 5657 | 13.0 | `results/nvidia-v1-frontier/frontier-router-b.jsonl (refusal=wrong; $/1M = verdicts + ledger refusals ÷ n)` |

## Chart files

- `quality_vs_cost.png` — scatter, log $ / 1M vs macro-F1
- `monthly_bill.png` — bar chart at 100M messages/month
- `latency.png` — p50 / p95 bars
- `title_card.png` — title
- `architecture_card.png` — from `docs/img/architecture.svg`
