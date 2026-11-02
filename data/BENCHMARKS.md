# Public eval benchmarks

One JSONL per benchmark in `data/public/`, eval-row schema:
`{"id","text","image_path","lang","gold_label","gold_categories","source","split"}`.
Binary `gold_label` ∈ {safe, unsafe} is the headline metric; `gold_categories` uses the
slugs in `data/TAXONOMY.md`. Rebuild with `python3 scripts/prep_public.py`
(`--fetch` re-downloads). Raw sources live in `data/public/raw/` (gitignored).
Current footprint: ~8 MB of ≤400 MB cap.

## Benchmark table

| benchmark | rows | unsafe | source / how obtained | license |
|---|---:|---:|---|---|
| `openai_moderation.jsonl` | 1,680 | 522 | `openai/moderation-api-release` → `data/samples-1680.jsonl.gz`, direct download | MIT |
| `wildguard_test.ids.txt` (ids only; full JSONL not shipped) | 1,725 ids | 754 unsafe in full set | `walledai/WildGuardTest` parquet (ungated mirror of the gated `allenai/wildguardmix` test split). Regenerate JSONL locally with `scripts/prep_public.py --fetch`. | ODC-BY — **ids only** |
| `xstest.jsonl` | 450 | 200 | `paul-rottger/xstest` → `xstest_prompts.csv`, direct download | CC-BY-4.0 |
| `hatemojicheck.jsonl` | 3,930 | 2,654 | `HannahKirk/Hatemoji` GitHub `HatemojiCheck/test.csv` (HF copy `HannahRoseKirk/HatemojiCheck` is gated; same file) | CC-BY-4.0 |
| `aegis2_test.jsonl` | 1,964 | 1,059 | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` → `test.json`, ungated | CC-BY-4.0 |
| `vlguard.jsonl` | 1,558 | 1,000 | `ys-zong/VLGuard` GitHub `data/test.json` metadata; **images gated** on HF (`test.zip`, ~125 MB, needs HF_TOKEN + terms) | MIT* |

\* VLGuard license was added by the author on 2024-08-09 per
https://huggingface.co/datasets/ys-zong/VLGuard/discussions/2 — MIT, plus
"users should also follow the licenses of the original image sources".

### License quotes (from each source's own LICENSE/card)

- **OpenAI Moderation** — `LICENSE` in repo: "MIT License … Permission is hereby granted,
  free of charge, to any person obtaining a copy of this software and associated
  documentation files …" (Copyright 2022 OpenAI). Repo is archived, file unchanged since 2022.
- **WildGuardTest** — dataset card YAML of `allenai/wildguardmix`: `license: odc-by`.
  Gating is `auto` with an AI2 Responsible Use form; the `walledai` mirror carries the
  identical 1,725-row prompt-harm parquet and is ungated.
- **XSTest** — repo `LICENSE` file header: "Attribution 4.0 International" (CC-BY-4.0).
- **HatemojiCheck** — repo README: "It is licensed under CC-BY-4.0." (`LICENSE` file:
  Creative Commons Attribution 4.0 International.)
- **Aegis 2.0** — dataset card: "**License:** CC-BY-4.0"; card YAML `license: cc-by-4.0`.
- **VLGuard** — MIT per HF discussion above (no LICENSE file in repo; mark partial).

## Gold-label → binary/category mapping (per `data/TAXONOMY.md`)

| benchmark | raw gold field | → `gold_label` | → `gold_categories` |
|---|---|---|---|
| openai_moderation | per-category flags S,H,V,HR,SH,S3,H2,V2 (missing = unknown) | any flag==1 ⇒ `unsafe` | S→`sexual`, S3→`sexual_minor`, H→`hate_identity`, H2→`hate_identity`+`threat`, HR→`harassment`, V→`violence`, V2→`violence`, SH→`self_harm` |
| wildguard_test | `label` ∈ {harmful, unharmful} (prompt-harm subset; `adversarial` flag dropped, not in schema) | harmful ⇒ `unsafe` | `[]` (binary-only benchmark) |
| xstest | `label` ∈ {safe, unsafe}; `type`/`focus`/`note` in raw csv | as given | `[]` — primary metric is over-block rate on the 250 safe rows |
| hatemojicheck | `label_gold` 1/0 | 1 ⇒ `unsafe` (+`hate_identity`), 0 ⇒ `safe` | `hate_identity` when hateful; `target`/`functionality`/`set` live in raw csv for the demojize-vs-VLM A/B |
| aegis2_test | `prompt_label` ∈ {safe, unsafe} + `violated_categories` csv | as given (`Needs Caution` ⇒ safe, kept as category) | 1:1 through TAXONOMY `nvidia_label` map; unknowns → `other` + logged (0 unknowns observed on test) |
| vlguard | `instr-resp` entries: `safe_instruction` ⇒ safe; `unsafe_instruction` ⇒ unsafe; bare `instruction` (only on `safe:false` items, the "unsafes" subset) ⇒ unsafe | as above | `[]` — per-instruction categories not annotated upstream |

Notes:
- `wildguard_test` ships as **ids only** (`wildguard_test.ids.txt`) because the upstream
  set is access-controlled; message text is not republished. The parquet = the 1,725-item
  **prompt-harm** slice of WildGuardTest (the full test set also has response-harm/refusal
  axes not needed for the binary task).
- `vlguard` rows carry `image_path = data/public/raw/vlguard/images/…` which does **not**
  exist until `test.zip` is fetched with HF_TOKEN; text-only evals should skip or
  treat them, vision evals must check existence.
- `aegis2_test` uses `prompt_label` (human-labeled per card `prompt_label_source`);
  `response_label` rows (llm_jury) not emitted — single-column input contract.
- XSTest row `id` preserves the original numeric id (1–450).

## Published scores (macro-F1 unless noted; verified vs papers where possible)

Verified directly against **SIREN** (arXiv:2604.18519, ACL 2026, Table 1 —
checked 2026-11-02; sota2.com rows match it exactly) and **ConsisGuard**
(arXiv:2605.31073, Table — checked 2026-11-02; matches sota2 rows):

| benchmark | model | macro-F1 | source |
|---|---|---:|---|
| OpenAI Moderation | SIREN (Llama3.2-1B) | **92.9** | arXiv:2604.18519 T1 ✓ |
| OpenAI Moderation | SIREN (Llama3.1-8B) | 92.0 | arXiv:2604.18519 T1 ✓ |
| OpenAI Moderation | LlamaGuard3-8B | 85.3 | arXiv:2604.18519 T1 ✓ |
| OpenAI Moderation | ShieldGemma-9B | 81.43 | sota2 (2026.06), unverified |
| OpenAI Moderation | Aegis Guard 2.0-8B | 81.0 | sota2 (2026.05), unverified |
| OpenAI Moderation | **GPT-5.1** | **80.24** | sota2 (2026.06), **aggregator-only, unverified** |
| OpenAI Moderation | Qwen3Guard-4B | 78.3 | arXiv:2604.18519 T1 ✓ |
| OpenAI Moderation | GPT-4o+CoT | 76.2 | arXiv:2605.31073 ✓ |
| OpenAI Moderation | SingGuard-8B | 75.61 | sota2 (2026.06); paper arXiv:2606.22873 exists, table not checked |
| OpenAI Moderation | ConsisGuard-7B | 73.11 | arXiv:2605.31073 ✓ |
| OpenAI Moderation | WildGuard-7B | 72.1 | arXiv:2605.31073 / sota2 ✓ |
| WildGuardTest | **Qwen3Guard-4B** | **90.6** | arXiv:2604.18519 T1 ✓ |
| WildGuardTest | ConsisGuard-7B | 89.96 | arXiv:2605.31073 ✓ |
| WildGuardTest | ConsisGuard-3B | 89.8 | arXiv:2605.31073 ✓ |
| WildGuardTest | GuardReasoner-8B | 89.17 | sota2 (2026.05), unverified |
| WildGuardTest | Qwen3Guard-0.6B | 89.1 | arXiv:2604.18519 T1 ✓ |
| WildGuardTest | WildGuard-7B | 88.9 | sota2 / WildGuard paper arXiv:2406.18495 ✓ |
| WildGuardTest | SIREN (Qwen3-4B) | 88.3 | arXiv:2604.18519 T1 ✓ |
| WildGuardTest | Gemini 1.5+CoT | 82.89 | arXiv:2605.31073 ✓ |
| WildGuardTest | LlamaGuard3-8B | 81.3 | arXiv:2604.18519 T1 ✓ |
| WildGuardTest | **GPT-4o+CoT** | **80.13** | arXiv:2605.31073 ✓ |
| WildGuardTest | GPT-4+CoT | 80.05 | arXiv:2605.31073 ✓ |
| Aegis 2.0 test | SIREN (Qwen3-4B) | 83.4 | arXiv:2604.18519 T1 ✓ |
| Aegis 2.0 test | Qwen3Guard-4B | 82.5 | arXiv:2604.18519 T1 ✓ |
| Aegis 2.0 test | LlamaGuard3-8B | 78.0 | arXiv:2604.18519 T1 ✓ |
| XSTest | GPT-4 (jun-2023) | 6.4% full + 2% partial refusal on safe prompts; refused all but 1 unsafe prompt | arXiv:2308.01263 §4.3–4.4 ✓ (response-level, not macro-F1) |
| HatemojiCheck | R6/R7/R8-T (Kirk et al., DeBERTa) | accuracy 0.87–0.88 | arXiv:2108.05921 Table 3 ✓ |
| HatemojiCheck | Perspective API "Identity Attack" | accuracy 0.69 | arXiv:2108.05921 Table 3 ✓ |
| VLGuard | — | evals are attack-success-rate of VLLM generations, not moderation F1; no frontier/guard scores published | arXiv:2402.02207 |

Frontier-model reference points for the demo pitch (from the verified rows):
OpenAI Moderation — GPT-5.1 **80.24** (unverified) vs best open guard SIREN-1B **92.9**.
WildGuard — GPT-4o+CoT **80.13** vs Qwen3Guard-4B **90.6**..

## Sources

- https://github.com/openai/moderation-api-release (MIT) · paper arXiv:2208.03274
- https://huggingface.co/datasets/allenai/wildguardmix (ODC-BY, gated) · mirror
  https://huggingface.co/datasets/walledai/WildGuardTest · paper arXiv:2406.18495
- https://github.com/paul-rottger/xstest (CC-BY-4.0) · paper arXiv:2308.01263 (NAACL 2024)
- https://github.com/HannahKirk/Hatemoji (CC-BY-4.0) · HF HannahRoseKirk/HatemojiCheck
  (gated auto) · paper arXiv:2108.05921 (NAACL 2022)
- https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (CC-BY-4.0)
- https://github.com/ys-zong/VLGuard · https://huggingface.co/datasets/ys-zong/VLGuard
  (gated) · paper arXiv:2402.02207 (ICML 2024)
- SIREN: arXiv:2604.18519 · ConsisGuard: arXiv:2605.31073 · SingGuard: arXiv:2606.22873 ·
  Qwen3Guard report: arXiv:2510.14276
- Aggregator (used only where marked): https://www.sota2.com/research/sota/
  harmfulness-detection-on-openai-moderation · /-on-wildguard · /-on-aegis-2-0

## Gated / blocked (record, don't work around)

| artifact | status | unblock |
|---|---|---|
| `allenai/wildguardmix` test parquet | gated (auto-accept form + HF account) | use walledai mirror (done — identical 1,725 rows) or HF_TOKEN |
| `HannahRoseKirk/HatemojiCheck` | gated (auto) | use GitHub `HatemojiCheck/test.csv` (done — identical 3,930 rows) or HF_TOKEN |
| `ys-zong/VLGuard` `test.zip` images | gated (share-HF-info) + 125 MB | set `HF_TOKEN`, run `prep_public.py --fetch` |
