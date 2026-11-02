# Data

Eval rows use the shared schema:
`{"id","text","image_path","lang","gold_label","gold_categories","source","split","perturbation","meta"}`.
Binary `gold_label` ∈ {safe, unsafe} is the headline metric; category slugs are defined in
`data/TAXONOMY.md`.

## NVIDIA Nemotron Content Safety (derived files — what the final model trained/evaluated on)

Source: `nvidia/Nemotron-3.5-Content-Safety-Dataset`
(https://huggingface.co/datasets/nvidia/Nemotron-3.5-Content-Safety-Dataset), **license CC-BY-4.0**
(attribution: NVIDIA Corporation). Derived by `scripts/prep_nvidia.py` (seed 17); the
`filtered/` files additionally had eval-colliding rows removed by `scripts/check_leakage.py`
(exact sha256 + Jaccard ≥ 0.8 near-dup; report in `filtered/leakage_report.json`,
method in `data/LEAKAGE.md`). Topic-following (`on-topic`/`off-topic`) rows and image rows
were excluded upstream; provenance + sha256 of raw inputs in `nvidia/manifest.json`.

| file | rows | how made | used for |
|---|---:|---|---|
| `nvidia/filtered/train_sft.jsonl` | 19,830 | text-safety rows from the train split (88,688 → dedup + leakage filter), eval schema → chat-SFT JSON target `{"label","categories"}` | the file uploaded for the final LoRA (`--max-examples 8000` seeded cap applied by `scripts/train_sft.py` at upload) |
| `nvidia/test.jsonl` | 3,340 | full text-safety test split, untouched | held-out headline eval set (`nvidia_test`); never trained or calibrated on |
| `nvidia/filtered/dev_calib.jsonl` | 2,479 | held-out-from-train calibration rows, leakage-filtered | threshold/escalation calibration (`scripts/calibrate_gate.py`, `systems.yaml`) |
| `nvidia/filtered/dev_sft.jsonl` | 2,479 | `dev_calib` converted to chat-SFT format | `--eval-file` for the training API run |
| `nvidia/filtered/leakage_report.json` | — | output of `scripts/check_leakage.py` | provenance |
| `nvidia/manifest.json` | — | output of `scripts/prep_nvidia.py` | provenance, license, raw-file sha256 |

Not shipped (regenerate with `scripts/prep_nvidia.py`): `train_kway.jsonl` (k-way
alternative target, unused by the final model), `validation.jsonl` (not used by the final
runs — the calibration dev set is `dev_calib`), `image_rows.jsonl` and `raw/` (parquet
dumps + image bundle; image rows were never evaluated).

## Public eval benchmarks (`data/public/`)

One JSONL per benchmark, built by `scripts/prep_public.py` (`--fetch` re-downloads).
Licenses verified per-source; see `data/BENCHMARKS.md` for the full table, license quotes,
and published reference scores.

| file | rows | source | license |
|---|---:|---|---|
| `openai_moderation.jsonl` | 1,680 | `openai/moderation-api-release` `samples-1680.jsonl.gz` | MIT |
| `xstest.jsonl` | 450 | `paul-rottger/xstest` `xstest_prompts.csv` | CC-BY-4.0 |
| `hatemojicheck.jsonl` | 3,930 | `HannahKirk/Hatemoji` GitHub `HatemojiCheck/test.csv` | CC-BY-4.0 |
| `aegis2_test.jsonl` | 1,964 | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` `test.json` | CC-BY-4.0 |
| `wildguard_test.ids.txt` | 1,725 ids | `walledai/WildGuardTest` (ungated mirror of gated `allenai/wildguardmix` prompt-harm test) | ODC-BY — **ids only** |

**WildGuardTest ships as row ids only.** The upstream set is distributed behind an
access-control form (AI2 Responsible Use), so message text is not republished here.
`wildguard_test.ids.txt` lists the 1,725 eval-row ids; regenerate the full JSONL locally
with `python3 scripts/prep_public.py --fetch`.

**VLGuard is not shipped at all** — it is gated on Hugging Face (HF_TOKEN + terms) and was
not used by the runs reported in `results/`. Same for SIUO (vision eval, gated) and all of
`data/vision/` — not used by the final text results.

## Frozen eval (`data/frozen/`)

| file | rows | how made |
|---|---:|---|
| `frozen/disguised_v1.jsonl` | 2,222 | 300 base rows (150 unsafe / 150 safe, seed 20261102) sampled from the obfuscation-robustness slice across the six eval sources above, each perturbed with up to 8 deterministic disguises (`none`, `base64_wrap`, `leetspeak`, `homoglyph`, `zero_width`, `mixed_case`, `char_spacing`, `emoji_sub`). See `data/ROBUSTNESS.md` and `frozen/README.md`. |

**Redaction:** the 320 rows in `disguised_v1.jsonl` derived from WildGuardTest
(`source == "wildguard_test"`) have `text` and `meta.orig_text` replaced by
`[redacted:<row-id>]`. `meta.source_row_id` and `meta.orig_sha256` are kept so the rows
can be re-materialised and verified after a local `prep_public.py`/`prep_robustness.py`
rebuild. The parent `data/robustness/obfuscation_test.jsonl` (8,251 rows) is not shipped —
it is superseded by the frozen `disguised_v1` subset used for scoring; it also contains
WildGuard text. Rebuild locally with `scripts/prep_robustness.py`.

## Docs

| file | what |
|---|---|
| `TAXONOMY.md` | binary label + 26 category slugs (`nvidia_label` mapping), locked |
| `BENCHMARKS.md` | per-benchmark sources, license quotes, gold-label mappings, published reference scores |
| `ROBUSTNESS.md` | disguise/perturbation methodology for `disguised_v1` |
| `LEAKAGE.md` | train↔eval leakage check method and results |
| `frozen/README.md` | frozen eval set v1 manifest + redaction note |
