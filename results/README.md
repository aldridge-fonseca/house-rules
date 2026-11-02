# Results — final runs only

Only the final reported runs live here: `nvidia-v1/` (our systems on the NVIDIA policy
task) and `nvidia-v1-frontier/` (frontier-model baselines on `nvidia_test`). Earlier
smoke/mock/dev runs and spend tracking are intentionally not published.

## `nvidia-v1/`

| file | rows | what |
|---|---:|---|
| `OURS_SUMMARY.md` | — | headline tables, paired tests, method notes |
| `final.jsonl` | 5,562 | per-row receipts for the full tuned system on `nvidia_test` (3,340) + `disguised_v1` (2,222) |
| `nano-untuned.jsonl` | 5,562 | receipts, base Nano 30B no adapter |
| `ultra-untuned.jsonl` | 5,562 | receipts, Nemotron 3 Ultra zero-shot |
| `specialist-only.jsonl` | 3,340 | receipts, tuned adapter alone on `nvidia_test` |
| `final-rawtext.jsonl` | 2,222 | receipts, `final` with clean-up off (disguised ablation) |
| `ours_summary.json` | — | all slices + paired statistics |
| `final_headline.json` | — | headline metrics incl. frontier comparison |
| `ours_timing.json` | — | GPU-time allocation inputs for $/1M-msgs |

Reproduce the tables ($0): `python3 scripts/nvidia_summary.py`.

## `nvidia-v1-frontier/`

| file | rows | what |
|---|---:|---|
| `frontier-router-a.jsonl` | 3,340 | GPT-6.1 Sol receipts on `nvidia_test` |
| `frontier-router-b.jsonl` | 3,340 | Claude Opus 5.5 receipts on `nvidia_test` |

## Notes

- **Deduplication:** every JSONL was rewritten last-row-wins per `id` (retry policy).
  No duplicated ids were found in any copied file, so all row counts are unchanged.
- **Redaction:** rows whose eval text comes from access-controlled WildGuardTest
  (`source == "wildguard_test"`, 320 rows in each `nvidia_test`+`disguised` receipt file)
  have `text` replaced by `[redacted:<row-id>]`. Labels, predictions, routes, latency and
  cost are untouched, so all published metrics reproduce. See `data/README.md`.
- `reused_from` fields in some receipts name an earlier internal run (`frozen-v1`) whose
  receipts are not part of this submission; the field is kept as provenance.
