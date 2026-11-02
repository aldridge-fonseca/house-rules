# Baseline plan and price notes

## Run list

- **GPT-6 Astra** (`gpt-6-astra`), **Claude Fable 5.1** (`claude-fable-5-1`), and **Gemini 4 Argon** (`gemini-4-argon`) are the closed frontier comparators. Their `FrontierProvider` instances receive one byte-for-byte shared system prompt generated from `data/TAXONOMY.md`, the same row, JSON schema, and temperature 0. Text and image input are sent together where supported. Billing uses provider-reported input/output token usage; Gemini thinking tokens are counted as output.
- **Token-router frontier arms** (`frontier-router`, `frontier-router-a`, `frontier-router-b`) run the same shared prompt through the user's OpenAI-compatible gateway: `ROUTER_BASE_URL` + `ROUTER_API_KEY` + per-arm `FRONTIER_MODEL*` env (chat completions, `response_format: json_object`, temperature 0, retries on transient/429/5xx). Cost comes from returned `usage` × `router.models` in `prices.yaml` — an unpriced model reports `cost_basis: unknown`, never a guessed number. Paid calls reserve+settle in the shared frontier budget ledger (default $40, hard max $150). Driver + dry-run plan: `baselines/run_frontier_router.py`.
- **OpenAI native moderation** (`omni-moderation-latest`) is an additional provider-native baseline. The OpenAI moderation endpoint is free and supports image input. It has its own native categories/decision logic, so it is reported separately from the shared-prompt frontier comparisons.
- **No requested guard is callable through Fireworks serverless today.** Fireworks marks [Llama Guard 3 8B](https://fireworks.ai/models/fireworks/llama-guard-3-8b) `Serverless: Not Supported`; the [Llama Guard 7B page](https://fireworks.ai/models/fireworks/llamaguard-7b) says the same. The Fireworks serverless catalogue has no model entries for Qwen3Guard, ShieldGemma, or Nemotron Safety Guard as checked 2026-11-02 ([catalogue](https://fireworks.ai/models?serverless=true)). Consequently `OpenGuardProvider` fails clearly instead of producing simulated benchmark verdicts. Published figures below are context only and are not results from this suite.

## Fairness and evaluation

Freeze row IDs and send identical rows to every live frontier lane. Use the same common prompt/taxonomy, strict JSON shape, temperature 0, and image bytes for the GPT/Claude/Gemini comparison. Do not silently drop provider errors or malformed JSON; keep them as failures. The provider-native moderation endpoint is an extra, separate comparator, not a replacement for the same-prompt frontier run. The open guards' published numbers use different datasets/prompts and must never be presented as directly comparable to our run.

## Price snapshot and approximate 2,000-row cost

Planning assumption from the project cost pass: **800 input and 30 output tokens per row**, including the shared fixed moderation instructions on an amortized basis. This gives 1.6M input and 60k output tokens over 2,000 rows. Costs below are estimates; calculate actual run cost from returned usage. Images, retries, and provider-side billed thinking tokens can change totals. `baselines/prices.yaml` is the machine-readable rate table.

| Baseline | Pinned model | Standard rate per 1M input/output | Approx. cost for 2,000 rows |
|---|---|---:|---:|
| OpenAI frontier | `gpt-6-astra` | $10 / $50 | **$163.00** |
| Anthropic frontier | `claude-fable-5-1` | $10 / $50 | **$163.00** |
| Gemini frontier | `gemini-4-argon` | $2 / $10 through Nov 30, 2026 | **$32.60** |
| OpenAI native moderation | `omni-moderation-latest` | Free | **$0.00** |
| Requested Fireworks open guards | not serverless-callable | no endpoint/rate applied | **Not run** |

Rates and request formats were checked against provider documentation: [OpenAI GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses), [Anthropic Claude Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/overview), [Anthropic structured outputs](https://platform.claude.com/docs/en/api/http/messages), [Google Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing), [Gemini structured output](https://ai.google.dev/gemini-api/docs/generate-content/structured-output), and [OpenAI Moderation](https://developers.openai.com/api/docs/guides/moderation). Gemini standard synchronous pricing is $2/$10 through November 30 and $4/$20 beginning December 1, 2026. No batch/flex discount is assumed.

## Published open-guard scores (not local results)

| Guard | Published result | Source and scope |
|---|---|---|
| Llama Guard 3 8B | No numeric benchmark score transcribed here; Fireworks listing only confirms model purpose/availability, not an eval score. | [Fireworks model page](https://fireworks.ai/models/fireworks/llama-guard-3-8b) |
| Qwen3Guard-8B | 85.3% overall accuracy (as quoted by an independent robustness paper); its split analysis reports 91.0% on public benchmark prompts and 33.8% on novel attacks. | [Robustness evaluation paper](https://arxiv.org/abs/2511.22047); different test protocol, not our score. |
| ShieldGemma 9B | OpenAI Mod optimal F1 0.821 and AU-PRC 0.907; SG Prompt optimal F1 0.828/AU-PRC 0.894. | [Official Google model card](https://gemma.ai/models/shieldgemma/model-card-v1/). These are benchmark-specific metrics. |
| Llama 3.1 Nemotron Safety Guard 8B V3 | 84.2% harmful-content classification accuracy. | [NVIDIA technical blog](https://developer.nvidia.com/blog/?p=107757); publisher-reported metric, dataset-specific. |

The figures have different datasets, prompts, metric definitions, and versions. They are background, not a leaderboard or substitute for running the same locked cases.
