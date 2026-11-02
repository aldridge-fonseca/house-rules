# Baselines

## Arms in the shipped results

| Arm | Model | Access | Rate (per 1M tokens) | Rows | Receipts |
|---|---|---|---|---:|---|
| `frontier-router-a` | `openai/gpt-6.1-sol` | OpenAI-compatible token router (`baselines/run_frontier_router.py`) | $2 / $10 (`baselines/prices.yaml`, source `https://api.tokenrouter.com/api/pricing`) | 3,340 `nvidia_test` | `results/nvidia-v1-frontier/frontier-router-a.jsonl` |
| `frontier-router-b` | `anthropic/claude-opus-5.5` | same token router | $4 / $20 (`prices.yaml`) | 3,340 `nvidia_test` | `results/nvidia-v1-frontier/frontier-router-b.jsonl` |

Both arms used the full shared policy + 26-category taxonomy prompt
(`safety_one/providers/frontier.py` `moderation_prompt()`), `response_format: json_object`,
and temperature 0. Runner default concurrency is **4** (`baselines/run_frontier_router.py`);
the actual `--concurrency` flag used for the shipped receipts was **not retained**. Latency
includes the gateway hop. Cost is returned `usage` × the router rates above — an unpriced
model reports `cost_basis: unknown`, never a guessed number. Paid calls reserve+settle in
the shared frontier budget ledger.

**Prompt asymmetry vs NVIDIA arms:** GPT/Opus received the full policy prompt
(~390–725 extra input tokens vs the NVIDIA label-only / one-line training prompts). The
NVIDIA zero-shot baselines (Ultra, untuned Nano) did **not** get this policy text. See
`results/nvidia-v1/OURS_SUMMARY.md` Method / Systems.

No disguised_v1 frontier receipts are shipped (earlier run, receipts not published).

Driver: `baselines/run_frontier_router.py`. Summary: `baselines/frontier_summary.py`.
Machine-readable rates: `baselines/prices.yaml`.

## Registry entries not run in this submission

`config/systems.yaml` also defines direct-provider arms (`gpt-6-astra`, `claude-fable-5-1`,
`gemini-4-argon`), OpenAI native moderation (`omni-moderation-latest`), and Fireworks open
guards. None of those produced shipped `nvidia-v1-frontier` receipts. Requested open guards
are not callable through Fireworks serverless today (catalogue checked 2026-11-02); their
published scores elsewhere use different datasets/prompts and are not comparable to this
suite.
