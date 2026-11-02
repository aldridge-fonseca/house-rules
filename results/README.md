# Results — final runs only

Only the final reported runs live here: `nvidia-v1/` (our systems on the NVIDIA policy
task) and `nvidia-v1-frontier/` (frontier-model baselines on `nvidia_test`). Earlier
smoke/mock/dev runs are intentionally not published. Scripts that still default to
`results/frozen-v1` or `results/vision-r4` regenerate an earlier internal run whose
receipts are **not published**; prefer `results/nvidia-v1` for the submission figures.

Frontier **disguised_v1** scores (GPT / Opus) come from an earlier run; per-case receipts
are **not published** and are excluded from asserted numbers (`final_headline.json` →
`null` for those cells).

## Index

| path | rows / entries | what | produced by | recompute ($0) |
|---|---:|---|---|---|
| `spend.json` | 2,878 entries | Opus 5.5 per-row billed cost on `nvidia_test` (measured). Used by `scripts/stats_vs_frontier.py` / `scripts/final_headline.py` to bill **368 of 434** Opus refusals ($1.67). GPT refusals are HTTP-level blocks with no matching ledger cost and are counted at $0 (conservative in GPT's favor). | recorded during the frontier run | do not invent; ledger is the source of truth |
| `nvidia-v1/OURS_SUMMARY.md` | — | headline tables, paired + clustered notes, method | curated from receipts; numbers match `ours_summary.json` | `python scripts/nvidia_summary.py` (JSON); re-read this MD |
| `nvidia-v1/ours_summary.json` | — | all slices + row-level paired tests | `scripts/nvidia_summary.py` | same |
| `nvidia-v1/final_headline.json` | — | headline metrics incl. frontier comparison slots | `scripts/final_headline.py` | same (owned by that script's maintainer) |
| `nvidia-v1/ours_timing.json` | — | GPU-time allocation inputs for $/1M-msgs | recorded with the eval runs | — |
| `nvidia-v1/final.jsonl` | 5,562 | full tuned system on `nvidia_test` (3,340) + `disguised_v1` (2,222) | `scripts/run_eval.py` + compact | paid re-run |
| `nvidia-v1/nano-untuned.jsonl` | 5,562 | base Nano 30B, no adapter | same | paid re-run |
| `nvidia-v1/ultra-untuned.jsonl` | 5,562 | Nemotron 3 Ultra zero-shot | same | paid re-run |
| `nvidia-v1/specialist-only.jsonl` | 3,340 | tuned adapter alone on `nvidia_test` | same | paid re-run |
| `nvidia-v1/final-rawtext.jsonl` | 2,222 | `final` with clean-up off (disguised ablation) | same | paid re-run |
| `nvidia-v1-frontier/frontier-router-a.jsonl` | 3,340 | GPT-6.1 Sol on `nvidia_test` | `baselines/run_frontier_router.py` | paid re-run |
| `nvidia-v1-frontier/frontier-router-b.jsonl` | 3,340 | Claude Opus 5.5 on `nvidia_test` | same | paid re-run |
| `stats/vs_frontier.{json,md}` | — | paired ΔF1 / recall / over-block vs GPT & Opus; McNemar; $/1M | `scripts/stats_vs_frontier.py` | `python scripts/stats_vs_frontier.py` |
| `stats/disguised_clustered.{json,md}` | — | base-message cluster bootstrap on `disguised_v1` (ours vs Ultra / Nano / rawtext) | `scripts/stats_disguised_clustered.py` | `python scripts/stats_disguised_clustered.py` |

## `nvidia-v1/` detail

Reproduce the ours tables ($0): `python3 scripts/nvidia_summary.py`.

## `nvidia-v1-frontier/` detail

No disguised_v1 frontier receipts are shipped. nvidia_test only (3,340 each).

## Notes

- **Deduplication:** every JSONL was rewritten last-row-wins per `id` (retry policy).
  No duplicated ids were found in any copied file, so all row counts are unchanged.
- **Redaction:** rows whose eval text comes from access-controlled WildGuardTest
  (`source == "wildguard_test"`, 320 rows in each `nvidia_test`+`disguised` receipt file)
  have `text` replaced by `[redacted:<row-id>]`. Labels, predictions, routes, latency and
  cost are untouched, so all published metrics reproduce. See `data/README.md`.
- `reused_from` fields in some receipts name an earlier internal run (`frozen-v1`) whose
  receipts are not part of this submission; the field is kept as provenance.
- Paths written into `results/stats/*.json` are **repo-relative** (never absolute checkout paths).
