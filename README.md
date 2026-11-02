# House Rules

**Your safety policy, enforced.**

House Rules is a content-moderation pipeline built on open NVIDIA models served by
Fireworks. It decides `safe` / `unsafe` under an explicit, configurable policy —
with an auditable route trail, measured latency, and measured cost per verdict —
instead of delegating the decision to a closed frontier API that applies a policy
you don't control and sometimes refuses to answer at all.

**Demo video:** _link to be added at submission_

## Project overview

- **Task.** Binary content moderation (`safe` / `unsafe`, plus taxonomy categories)
  for chat text and images, evaluated under NVIDIA's Nemotron Content Safety 3.5
  policy on held-out data.
- **Intended user.** Product and trust-and-safety teams that need a configurable,
  auditable moderation decision at message volume — where a closed API's fixed
  policy, refusal behavior, latency, and per-token pricing are the wrong fit.
- **Approach.** Deterministic clean-up and rules first; a small fine-tuned open
  model second; an expensive model only for the gray zone. A message is
  normalized (obfuscation unfolds), screened by a rule gate, routed by content
  type to a specialist, and escalated only when the specialist is uncertain.
- **Why it matters.** Fine-tuning a 30B open model on the target policy beat the
  same model untuned by **+12.6 macro-F1** and a 550B zero-shot model by **+10.8
  macro-F1** on 3,340 held-out cases, at roughly half the cost per message — and
  it answered every row, where closed frontier baselines refused 1.9–13% of the
  same requests and cost 33–114× more per message.

## Results

**Method.** Every system ran the same rows with the same prompt, taxonomy, and
deterministic scorer. Headline metric is binary macro-F1 over `safe`/`unsafe`,
reported with unsafe recall, over-block rate (share of gold-`safe` rows blocked),
p50/p95 latency, and USD per 1M messages. Provider refusals and errors are scored
as incorrect. Differences use paired bootstrap 95% CIs and exact McNemar tests on
id-matched rows (`results/nvidia-v1/ours_summary.json`).

**Datasets.**

- `nvidia_test` — 3,340 held-out rows from NVIDIA's Nemotron-3.5 Content Safety
  test split (`data/nvidia/test.jsonl`): 590 English + 2,750 non-English rows
  across 11 languages. Never trained or calibrated on.
- `disguised_v1` — 2,222 rows (`data/frozen/disguised_v1.jsonl`) attacking the
  pipeline with 8 obfuscation perturbations (base64, char-spacing, emoji
  substitution, homoglyphs, leetspeak, mixed case, zero-width chars).

### NVIDIA-policy held-out test (n=3,340)

| System | Macro-F1 | Unsafe recall | Over-block | p50 ms | p95 ms | $/1M msgs |
|---|---:|---:|---:|---:|---:|---:|
| **House Rules** (tuned Nano pipeline) | **95.3** | **97.6** | 5.5 | **644** | **1,193** | **56** |
| Nemotron Nano 3 30B, untuned | 82.7 | 93.3 | 27.8 | 804 | 1,157 | 64 |
| Nemotron 3 Ultra 550B, zero-shot | 84.5 | 86.7 | **3.3** | 710 | 1,594 | 108 |
| GPT-6.1 Sol (frontier router) | 71.5 | 70.0 | 1.1 | 3,235 | 6,067 | 1,854 |
| Claude Opus 5.5 (frontier router) | 69.0 | 67.6 | 3.9 | 2,966 | 5,658 | 6,375 |

Non-English slice (n=2,750, 11 languages): House Rules holds at **95.9 macro-F1 /
98.6 unsafe recall**; the untuned Nano drops to 81.3 and zero-shot Ultra to 81.8.

Refusals (scored as errors): GPT-6.1 Sol refused **1.9%** of these requests;
Claude Opus 5.5 refused **13.0%** of them. House Rules and both
NVIDIA baselines answered 100% of rows.

### Disguised attacks (n=2,222)

| System | Macro-F1 | Unsafe recall | Over-block | p50 ms | p95 ms | $/1M msgs |
|---|---:|---:|---:|---:|---:|---:|
| House Rules | 82.0 | **89.4** | 25.3 | **571** | **1,115** | **69** |
| **GPT-6.1 Sol** | **87.8** | 81.8 | **6.0** | 2,437 | — | 1,740 |
| Claude Opus 5.5 | 64.8 | 59.0 | 29.1 | 2,935 | — | 6,021 |
| Nemotron Nano 3 30B, untuned | 63.3 | 94.0 | 62.2 | 775 | 1,086 | 63 |
| Nemotron 3 Ultra 550B, zero-shot | 82.0 | 83.8 | 19.9 | 713 | 1,013 | 148 |

Opus refused **28.8%** of disguised rows (scored as errors); GPT-6.1 Sol refused
0.3%.

### Paired differences (House Rules − baseline, macro-F1 points)

| Baseline | Dataset | Δ | 95% CI | McNemar p |
|---|---|---:|---|---:|
| Nemotron Nano 30B untuned | nvidia_test | **+12.6** | [+11.0, +14.4] | 3.5e-49 |
| Nemotron 3 Ultra 550B zero-shot | nvidia_test | **+10.8** | [+9.4, +12.4] | 2.5e-54 |
| Nemotron Nano 30B untuned | disguised_v1 | **+18.7** | [+16.5, +20.8] | 7.8e-55 |
| Nemotron 3 Ultra 550B zero-shot | disguised_v1 | −0.0 | [−1.6, +1.6] | 0.96 |
| Pipeline − clean-up step (ablation) | disguised_v1 | **+8.0** | [+6.4, +9.7] | 8.7e-21 |

Paired frontier stats (id-matched, refusals scored as wrong) are in
`results/stats/vs_frontier.md`: ours − Opus 5.5 is **+26.3 macro-F1**
[+24.6, +28.0], McNemar p=7.0e-193 — far outside any CI above. Frontier
disguised-attack scores come from
`results/nvidia-v1/final_headline.json` (p95 not reported there, shown as —);
the NVIDIA-policy frontier rows are recomputed from the shipped per-case
receipts in `results/nvidia-v1-frontier/`.

### Sources

All numbers are computed from run receipts, not mocks:

- `results/nvidia-v1/ours_summary.json` — House Rules + NVIDIA baselines, all
  slices, paired tests (`scripts/nvidia_summary.py` regenerates it for $0).
- `results/nvidia-v1-frontier/frontier-router-{a,b}.jsonl` — per-case frontier
  receipts for the NVIDIA-policy test (3,340 rows each).
- `results/nvidia-v1/final_headline.json` — aggregated table incl. frontier
  disguised-attack scores (`scripts/final_headline.py` regenerates it for $0).
- `results/nvidia-v1/ours_timing.json` — GPU-allocation inputs for $/1M.

**Cost method.** $/1M messages = measured per-token spend in the receipts +
allocated dedicated-GPU time: $13/h (1× B200) × measured run wall-clock ÷ model
calls in that run, at concurrency 48. Deployment spin-up and idle time are
excluded, so dedicated-GPU costs assume full utilization (see Limitations).
Frontier costs are measured token usage × router prices. Latency is client-side
wall time per message including queueing and escalation round-trips.

## Architecture

![Architecture](docs/img/architecture.svg)

1. **Normalize (stage 0).** Unicode NFKC, zero-width removal, conservative
   leetspeak/confusable folding, bounded base64/ROT13 decode. The original text is
   always retained; the model sees the normalized view. This step is worth
   **+8.0 macro-F1** on disguised attacks — with it off, unsafe recall on
   base64-wrapped input collapses to 0.7%.
2. **Rule gate.** Deterministic PII / jailbreak / policy-pattern checks. A
   high-confidence hit short-circuits to a verdict (no NVIDIA-test row was
   rule-decided; the gate mostly matters for obvious cases and audit).
3. **Router.** Deterministic routing on content type: image or emoji-dense →
   vision lane; non-English → multilingual lane; otherwise → specialist.
4. **Specialist.** NVIDIA Nemotron Nano 3 30B-A3B with a LoRA adapter
   (rank 8, one epoch, 8K-example train cap) fine-tuned on NVIDIA Nemotron-3.5
   Content Safety train rows, served on a dedicated Fireworks deployment
   (1× B200). The multilingual lane uses the same tuned
   model; the vision lane is a Nemotron 3.5 Super VL 120B on-demand deployment.
5. **Confidence gate.** If P(unsafe) lands in the gray zone (default
   [0.35, 0.65], widened per high-stakes category), the row escalates to
   **Nemotron 3 Ultra 550B** on Fireworks serverless. Only 3.2% of
   NVIDIA-policy rows and 9.8% of disguised rows escalate — the rest pay the
   small model's price.

Every verdict carries `{label, categories, route, rule_hits, model, latency,
cost}` — an auditable path for each decision.

## Sponsor technology (actually used, with receipts)

- **Fireworks AI**
  - Managed LoRA fine-tuning (`firectl sftj`) of Nemotron Nano 3 30B-A3B.
  - Dedicated on-demand deployments: tuned specialist + untuned-Nano ablation
    (1× B200 each), Nemotron 3.5 Super VL vision (4× B200).
  - Serverless per-token inference: Nemotron 3 Ultra 550B for the zero-shot
    baseline and the escalation lane.
- **NVIDIA**
  - Nemotron Nano 3 30B-A3B (base model and tuned specialist), Nemotron 3 Ultra
    550B, Nemotron 3.5 Super VL 120B.
  - Nemotron-3.5 Content Safety Dataset (training + held-out eval) and the
    Aegis 2.0 test subset.
- **Microsoft:** no Microsoft services are used.

## Tradeoffs and limitations

- **Disguised attacks are the weak spot — GPT-6.1 Sol beats us there** (87.8 vs
  82.0 macro-F1), and zero-shot Ultra ties us (paired Δ −0.0, CI [−1.6, +1.6]).
  House Rules gets higher unsafe recall (89.4 vs 81.8/83.8) at the price of
  over-blocking one in four safe disguised rows (25.3%). If your traffic is
  mostly adversarial text and recall-tolerant, the frontier model is better —
  at ~25× the per-message cost and ~4× the p50 latency.
- **Over-block is the price of recall.** 5.5% on-policy, 25.3% under disguise.
  The operating point was chosen for unsafe recall on the dev split; the
  threshold and gray zone are config, not constants.
- **Cost assumes full utilization.** Dedicated-GPU $/1M amortizes $13/h over
  measured run wall-clock at concurrency 48 and excludes spin-up/idle. At low
  or bursty traffic, serverless pricing dominates and the gap narrows.
- **One policy.** Scores are under NVIDIA's content-safety taxonomy. Other labs'
  policies differ; the same system under another policy needs re-labeled data
  and a re-tuned threshold (the pipeline supports it; the measured numbers
  don't transfer).
- **Opus refusal rate.** Claude Opus 5.5 completed all 3,340 NVIDIA-policy
  rows but refused to classify 13.0% of them (28.8% of disguised rows);
  refusals are scored as incorrect. The refusal rate is itself the finding.
- **Escalation adds little on-policy** (+0.2 macro-F1, n.s.; it does cut
  over-block 7.3→5.5% and escalates only 3.2% of rows). It earned its keep on
  adversarial and mixed-policy inputs, not on the NVIDIA-policy test.
- **Human review.** This is a moderation component, not a trust-and-safety
  program; high-impact decisions still need appeal and human review paths.
- **Vision lane.** Image routing is wired and exercised by the demo's image
  cases; its separate image-eval A/B receipts are not part of the published
  `results/` set, so no vision score is claimed above.
- Untuned Nano produced an unparseable label on 12/5,562 rows (counted
  incorrect); all other reported runs had zero provider errors.

## Setup

Requires Python 3.12. The test suite, the offline eval, and the replay demo make
**no paid calls**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys for live runs only
python -m pytest -q
```

## Reproduce

```bash
# Offline eval on bundled fixtures + mock arms ($0)
python scripts/run_eval.py

# Eval any system on the prepared datasets; live systems need .env keys
python scripts/run_eval.py \
  --data data/nvidia/test.jsonl \
  --data data/frozen/disguised_v1.jsonl \
  --system full=safety_one.registry:full \
  --out results --run myrun --concurrency 48

# Regenerate the headline tables from receipts ($0, no model calls)
python scripts/nvidia_summary.py     # -> results/nvidia-v1/ours_summary.json
python scripts/final_headline.py     # -> results/nvidia-v1/final_headline.json
                                     #    (frontier disguised cells need the
                                     #    full frozen-v1 receipt set)

# Interactive "moderation race" demo (replay cache by default, loopback only)
python demo/server.py                # http://127.0.0.1:8787/
python demo/server.py --live         # real calls, needs keys
```

Frontier router arms are driven by `baselines/run_frontier_router.py`
(`ROUTER_BASE_URL`, `ROUTER_API_KEY`, `FRONTIER_MODEL_{A,B}`). Training/deploy
recipes live in `scripts/train_launch.sh`, `scripts/deploy_create.py`, and
`scripts/vision_deploy.sh`; data preparation in `scripts/prep_*.py`. See
`config/systems.yaml` for the full system registry (pipeline ladder `+rules` →
`+router` → `+escalation`/`full`, single-provider baselines) and `demo/README.md`
for live-mode details.

## Reproduce in a notebook

`notebooks/house_rules_eval.ipynb` is a keyless, top-to-bottom demonstration:
it reloads the run receipts in `results/nvidia-v1/` and
`results/nvidia-v1-frontier/`, recomputes every headline number above with the
repo's own scoring code (`safety_one.eval.metrics`,
`scripts/nvidia_summary.py`), and **asserts each one matches this README** —
quality/cost/latency charts, a monthly-bill table at 1M–100M messages, the
per-route breakdown, real example rows (errors included), and one optional
live-classification cell that is skipped unless `FIREWORKS_API_KEY` and
`FIREWORKS_MODEL` are set.

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/house_rules_eval.ipynb
```

## Data and licenses

Code is MIT (see `LICENSE`). Prepared datasets under `data/` keep their original
licenses (NVIDIA Nemotron-3.5 Content Safety CC-BY-4.0, Aegis 2.0 CC-BY-4.0,
OpenAI Moderation MIT, WildGuardTest ODC-BY, XSTest CC-BY-4.0, HatemojiCheck
CC-BY-4.0, VLGuard MIT) — see `data/README.md` and `data/BENCHMARKS.md` for
sources, counts, and terms.
