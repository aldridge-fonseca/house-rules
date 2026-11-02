# demo/ — live "moderation race" screen

A single-URL race: **Safety One** pipeline vs a **closed frontier model** on the
same locked chat cases. Every verdict card shows the full route trail as badges
naming the actual NVIDIA model on Fireworks, latency + cost, the cleaned-up
text when normalization changed it, and a running tally (accuracy, $/1M
messages, p95) for ours vs frontier.

## Run

```bash
python3 demo/server.py                     # replay cached/results.jsonl
python3 demo/server.py --replay-fallback   # offline stage-safe LIVE-RECORDED copy
python3 demo/server.py --live              # classify cases now (needs keys + deployment)
python3 demo/smoke.py                      # self-check replay; leaves nothing running
python3 demo/smoke.py --replay-fallback    # self-check the frozen fallback cache
```

Open `http://127.0.0.1:8787/?run=0` to autostart the race. Binds to 127.0.0.1;
stdlib only; works offline from the LIVE-RECORDED cache.

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

Lane mapping for classify: **ours** = registry `full` via `scripts.nano_ladder`
(fine-tuned NVIDIA Nemotron Nano 3 on Fireworks); **frontier** = `frontier-router-a`
(GPT-6.1 Sol, same prompt as results). UI badge: `LIVE-RECORDED · <ISO timestamp>`.

Optional offline path from eval receipts (not the default demo cache):

```bash
python3 demo/record.py --run results/<run> [--vision results/<vision-run>] \
    [--ours-system full] [--frontier-system frontier-router-a] [--select auto|cases]
```

`--run` never ships WildGuardTest / VLGuard / SIUO rows (filtered out). Image
cases from vision receipts are kept only when the file is already under
`demo/assets/` with a shareable license.

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

- **Header** — `LIVE` / `CACHED REPLAY` / `FALLBACK REPLAY` + `LIVE-RECORDED · <ts>`
  (or a non-live warning if a lane was simulated).
- **Chat stream** — every case as it is scored (id, lang, source, gold).
- **Safety One lane** — verdict, categories, route trail (Rules → Nano SFT →
  Ultra escalation…), clean-up line, Σ latency · $cost.
- **Closed frontier lane** — GPT-6.1 Sol (or configured router arm), same cases.
- **Meters / tally** — accuracy, $/1M messages, p50/p95, cost/correct.
- **Threshold slider** — re-scores cached P(unsafe) client-side.

## Lanes (`backends.py`)

- **ours** — `scripts.nano_ladder` `full` pipeline when `FIREWORKS_API_KEY` +
  `FIREWORKS_MODEL` are set (mode `live`); otherwise deterministic mocks.
- **frontier** — `frontier-router-a` when `DEMO_LIVE_FRONTIER=1` + `ROUTER_API_KEY`
  (mode `live`); otherwise a `*-sim` mock. No paid calls by default.

## Files

| file | what |
|---|---|
| `server.py` | stdlib HTTP + SSE (`--replay-fallback`, `--live`, `--record`) |
| `index.html` | whole UI; metrics from cached P(unsafe) |
| `backends.py` | lane wiring (live nano_ladder + router, or mocks) |
| `cases.jsonl` | 100 locked rows (`gen_cases.py`) |
| `cached/results.jsonl` | LIVE-RECORDED replay cache |
| `cached/fallback_recorded.jsonl` | frozen LIVE copy for `--replay-fallback` |
| `record.py` | live classify (parallel) or `--run` from receipts |
| `smoke.py` | starts server, hits endpoints, checks shipped assets, exits |
| `assets/meme1.svg` | only image case (hate text embedded in the SVG) |
