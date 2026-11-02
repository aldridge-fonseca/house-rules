# Frozen eval set v1 (locked 2026-11-02)

Eval rows in this repo:

| File | Rows |
|---|---:|
| `data/public/openai_moderation.jsonl` | 1,680 |
| `data/public/xstest.jsonl` | 450 |
| `data/frozen/disguised_v1.jsonl` | 2,222 |
| `data/public/wildguard_test.ids.txt` | 1,725 (ids only — see below) |

`disguised_v1.jsonl`: 300 base messages (150 unsafe / 150 safe, seed 20261102) drawn from
the obfuscation-robustness slice (see `data/ROBUSTNESS.md`), each with every disguise
variant it has (none, base64_wrap, leetspeak, homoglyph, zero_width, mixed_case,
char_spacing, emoji_sub). Same bases across variants, so per-disguise deltas are paired.

**Redaction:** rows derived from WildGuardTest (`source == "wildguard_test"`, 320 rows)
have `text` and `meta.orig_text` replaced by `[redacted:<row-id>]` because the upstream
set is access-controlled. Regenerate them locally via `scripts/prep_public.py --fetch`
+ `scripts/prep_robustness.py`; row ids are stable (`rob-<perturbation>-wildguard-NNNN`)
and `meta.source_row_id` / `meta.orig_sha256` are kept for join-back verification.
