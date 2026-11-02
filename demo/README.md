# demo/ — live "moderation race" screen

A single-URL race: the **House Rules** pipeline vs a **closed frontier model**
(GPT-6.1 Sol) on the same locked chat cases. Every verdict card shows the full
route trail as badges naming the actual NVIDIA model on Fireworks, latency +
cost, the cleaned-up text when normalization changed it, and a running tally
(accuracy, $/1M messages, p50/p95) for ours vs frontier.

**Built on Fireworks AI · NVIDIA Nemotron** (text branding only — no logos, no
endorsement claim).

## Run

```bash
# Replay mode — offline, no API key. Serves demo/cached/results.jsonl.
python3 demo/server.py                     # default replay
python3 demo/server.py --replay-fallback   # stage-safe LIVE-RECORDED copy
python3 demo/smoke.py                      # self-check replay; leaves nothing running
python3 demo/smoke.py --replay-fallback    # self-check the frozen fallback cache

# Live mode — needs keys + a tuned-Nano Fireworks deployment.
# FIREWORKS_API_KEY, FIREWORKS_MODEL=accounts/<acct>/deployments/<id>,
# and for the frontier lane DEMO_LIVE_FRONTIER=1 + ROUTER_API_KEY + FRONTIER_MODEL_A.
python3 demo/server.py --live
python3 demo/smoke.py --live
```

Open `http://127.0.0.1:8787/?run=0` to autostart the race. Binds to 127.0.0.1;
stdlib only. Replay works fully offline from the LIVE-RECORDED cache.

## Lanes

| Lane | What it is |
|---|---|
| **ours (House Rules)** | Registry `full` via `scripts.nano_ladder`: normalize → rules → router → LoRA-tuned NVIDIA Nemotron Nano 3 on Fireworks, with gray-zone escalation to Nemotron 3 Ultra. |
| **frontier** | `frontier-router-a` = GPT-6.1 Sol, same zero-shot prompt/taxonomy as the NVIDIA arms in `results/`. |

Image routing / vision lane is **implemented and unit-tested**, but **no shipped
demo receipt classified an image on the vision lane** (the meme SVG case `c11`
was folded into text and routed `specialist` on the tuned Nano). The UI does not
claim a live vision/Super VL run.

## Cost & latency caveat (read this)

The meters show **two** $/1M numbers per lane, clearly labeled:

1. **this 100-message demo run (low utilization, GPU time ÷ 100 msgs)** — raw
   spend from the LIVE-RECORDED cache (`demo/cached/record_summary.json` /
   `results.jsonl`). Ours ≈ **$2,809/1M** because a dedicated B200 ($13/h) ran
   ~79 s of wall-clock for only 100 low-concurrency messages. GPT billed ≈
   **$1,541/1M** on the same 100.
2. **production basis (from results/)** — amortized at **full utilization** on
   the held-out NVIDIA test (n=3,340): House Rules **$56/1M** vs GPT-6.1 Sol
   **$1,820/1M**. Sources: `results/nvidia-v1/final_headline.json` /
   `ours_summary.json`, and `results/stats/vs_frontier.json`. Also shipped as
   `demo/cached/production_basis.json`; the server re-reads the results files
   at runtime when present.

Latency: the UI shows **p50 and p95** for the demo run. Ours p50 is fast
(~178 ms) but demo p95 is ~11.3 s because the entire tail is **7/100 messages
escalated to Ultra serverless** (11–35 s each: cold/queued). Every
non-escalated row finished in **≤ 425 ms** (counted from
`demo/cached/*.jsonl`). Production p95 on 3,340 rows is **1,193 ms** vs GPT
**6,066 ms**.

## Recording the replay cache (LIVE)

```bash
# Needs FIREWORKS_MODEL=accounts/<acct>/deployments/<id> (tuned Nano),
# FIREWORKS_API_KEY, ROUTER_API_KEY + ROUTER_BASE_URL, FRONTIER_MODEL_A=openai/gpt-6.1-sol.
# Caps Fireworks GPU ($13/h B200) + GPT spend at $12; freezes fallback on success.
python3 demo/record.py --live --require-live --workers 8 --budget 12 --fallback
```

Writes:

| file | what |
|---|---|
| `cached/results.jsonl` | default replay cache (LIVE classify of the 100 demo cases) |
| `cached/fallback_recorded.jsonl` | frozen copy for `--replay-fallback` (stage-safe) |
| `cached/fallback_meta.json` | recorded_at + LIVE-RECORDED label |
| `cached/record_summary.json` | accuracy / latency / cost receipt |
| `cached/production_basis.json` | production $/1M + demo escalation note (from `results/`) |

Lane mapping for classify: **ours** = registry `full` via `scripts.nano_ladder`
(fine-tuned NVIDIA Nemotron Nano 3 on Fireworks); **frontier** = `frontier-router-a`
(GPT-6.1 Sol, same prompt as results). UI badge: `LIVE-RECORDED · <ISO timestamp>`.

Optional offline path from eval receipts (not the default demo cache):

```bash
python3 demo/record.py --run results/nvidia-v1 [--ours-system full] \
    [--frontier-system frontier-router-a] [--select auto|cases]
```

`--run` never ships WildGuardTest / VLGuard / SIUO rows (filtered out). Image
cases from vision receipts are kept only when the file is already under
`demo/assets/` with a shareable license. Scripts that still default to
`results/frozen-v1` regenerate an earlier internal run whose receipts are
**not published** in this repo — prefer `results/nvidia-v1`.

## Cases + licensing

`cases.jsonl` (regenerate with `python3 demo/gen_cases.py`):

- ~20 original curated tricky rows (emoji threats, leet/homoglyph hate, PII,
  non-English, jailbreak, meme SVG under `assets/meme1.svg`).
- ~80 rows sampled from **permissively licensed** held-out sets only:
  NVIDIA Nemotron CS 3.5 test (CC-BY-4.0), XSTest (CC-BY-4.0), OpenAI
  Moderation (MIT), Aegis 2.0 (CC-BY-4.0), HatemojiCheck (CC-BY-4.0).

**Not included:** WildGuardTest, VLGuard, SIUO (gated / not shippable). The only
image is `/assets/meme1.svg` (drawn for this demo). No `/case-img/...` paths.

## Screen

- **Header** — `HOUSE RULES` + `Built on Fireworks AI · NVIDIA Nemotron`;
  `LIVE` / `CACHED REPLAY` / `FALLBACK REPLAY` + `LIVE-RECORDED · <ts>`
  (or a non-live warning if a lane was simulated).
- **Chat stream** — every case as it is scored (id, lang, source, gold).
- **House Rules lane** — verdict, categories, route trail (Rules → Nano SFT →
  Ultra escalation…), clean-up line, Σ latency · $cost.
- **Closed frontier lane** — GPT-6.1 Sol (or configured router arm), same cases.
- **Meters / tally** — accuracy, macro-F1, p50/p95, **raw demo $/1M** next to
  **production-basis $/1M**, plus an honest p95-tail note for Ultra escalations.
- **Threshold slider** — re-scores cached P(unsafe) client-side.
- **Footer** — Fireworks Training API (LoRA SFT; job id not recorded); no
  unevidenced vision / `firectl sftj` claim.

## Files

| file | what |
|---|---|
| `server.py` | stdlib HTTP + SSE (`--replay-fallback`, `--live`, `--record`); serves production_basis via `/api/meta`; refuses path traversal on `/assets/` and `/case-img/` |
| `index.html` | whole UI; metrics from cached P(unsafe) + production basis |
| `backends.py` | lane wiring (live nano_ladder + router, or mocks) |
| `cases.jsonl` | 100 locked rows (`gen_cases.py`) |
| `cached/results.jsonl` | LIVE-RECORDED replay cache |
| `cached/fallback_recorded.jsonl` | frozen LIVE copy for `--replay-fallback` |
| `cached/production_basis.json` | production $/1M + escalation stats |
| `record.py` | live classify (parallel) or `--run` from receipts |
| `smoke.py` | starts server, hits endpoints, checks path traversal + assets, exits |
| `assets/meme1.svg` | only image case (hate text embedded in the SVG) |
