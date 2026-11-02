# Train ↔ public-eval leakage check

`scripts/check_leakage.py` (stdlib only, ~15 s) compares every NVIDIA training row against
every eval text we report on: all of `data/public/*.jsonl`, `data/robustness/obfuscation_test.jsonl`
(both the perturbed `text` and the clean `meta.orig_text`), and the NVIDIA `test`/`validation`
splits as a re-check of `prep_nvidia.py`'s own pass.

- **Exact:** sha256 of normalized text (NFKC, casefold, zero-width + punctuation stripped,
  whitespace collapsed; emoji and symbols kept).
- **Near-duplicate:** Jaccard ≥ 0.8 on word 3-gram shingles (character 4-grams for texts under
  5 words, e.g. CJK/Thai). Exact all-pairs search with prefix filtering, then verified Jaccard.

## Result (run 2026-11-02)

| file | rows | flagged | exact | near-dup only | kept |
|---|---:|---:|---:|---:|---:|
| `train_sft.jsonl` | 20,000 | 170 | 115 | 55 | 19,830 |
| `train_kway.jsonl` (row-aligned with SFT) | 20,000 | 170 | 115 | 55 | 19,830 |
| `dev_calib.jsonl` | 2,500 | 21 | 12 | 9 | 2,479 |

Train rows flagged per eval source (a row can hit several sources):

| eval source | exact | near-dup |
|---|---:|---:|
| `public/aegis2_test` | 114 | 48 |
| `robustness/obfuscation_test` (via `orig_text`) | 11 | 0 |
| `public/openai_moderation` | 1 | 0 |
| `nvidia/test` | 0 | 4 |
| `nvidia/validation` | 0 | 3 |
| `public/wildguard_test`, `xstest`, `hatemojicheck`, `vlguard` | 0 | 0 |

Aegis 2.0 is the expected overlap: the NVIDIA dataset includes Aegis-derived sources. The 7
NVIDIA test/validation near-dups were missed by the earlier probabilistic MinHash pass.

## Outputs (originals untouched)

- `data/nvidia/filtered/train_sft.jsonl`, `train_kway.jsonl`, `dev_calib.jsonl`: colliding rows removed.
- `data/nvidia/filtered/leakage_report.json`: counts, per-source table, examples, sha256 of outputs.

**Train on `data/nvidia/filtered/train_sft.jsonl`.** Even after
filtering, the Aegis 2.0 score should carry a footnote: it shares a source with the training data,
and we cannot rule out leakage of paraphrases below Jaccard 0.8.
