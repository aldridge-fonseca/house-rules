#!/usr/bin/env python3
"""Build eval JSONL files from public safety benchmarks (safety-one/data/public/).

Reads raw downloads from data/public/raw/<benchmark>/ (gitignored) and writes
one JSONL per benchmark in the eval-row schema:

    {"id","text","image_path","lang","gold_label","gold_categories","source","split"}

Label mappings follow data/TAXONOMY.md ("Benchmark mappings" section).

Usage:
    python3 scripts/prep_public.py            # build all eval JSONLs from raw/
    python3 scripts/prep_public.py --fetch   # also (re)download the public raw files

Gated sources (need HF_TOKEN, or accept terms on HF first):
  - VLGuard images (test.zip, ~125 MB) — metadata comes from GitHub and is NOT gated.
    Without images the vlguard rows are emitted with image_path set to the expected
    local location (data/public/raw/vlguard/images/<relpath>); vision evals must
    skip rows whose image_path does not exist.
  - allenai/wildguardmix (the official WildGuardTest parquet) is gated; we use the
    identical walledai/WildGuardTest mirror (ODC-BY, ungated) instead.
  - HannahRoseKirk/HatemojiCheck on HF is gated; we use the identical test.csv in
    the authors' GitHub repo (CC-BY-4.0) instead.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "public"
RAW = DATA / "raw"

# ---------------------------------------------------------------------------
# raw file locations (fetch URLs + local relative path)
# ---------------------------------------------------------------------------
RAW_FILES = {
    "openai_moderation": (
        "https://github.com/openai/moderation-api-release/raw/main/data/samples-1680.jsonl.gz",
        "openai_moderation/samples-1680.jsonl.gz",
    ),
    # Official allenai/wildguardmix is gated; walledai mirror is identical (ODC-BY).
    "wildguard": (
        "https://huggingface.co/datasets/walledai/WildGuardTest/resolve/main/data/train-00000-of-00001.parquet",
        "wildguard/wildguard_test.parquet",
    ),
    "xstest": (
        "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
        "xstest/xstest_prompts.csv",
    ),
    # HF copy is gated; the authors' GitHub repo ships the same file (CC-BY-4.0).
    "hatemoji": (
        "https://raw.githubusercontent.com/HannahKirk/Hatemoji/main/HatemojiCheck/test.csv",
        "hatemoji/test.csv",
    ),
    "aegis": (
        "https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/resolve/main/test.json",
        "aegis/test.json",
    ),
    "vlguard": (
        "https://raw.githubusercontent.com/ys-zong/VLGuard/main/data/test.json",
        "vlguard/test.json",
    ),
}

# Optional gated image archive for VLGuard (requires HF_TOKEN + accepting terms).
VLGUARD_ZIP_URL = "https://huggingface.co/datasets/ys-zong/VLGuard/resolve/main/test.zip"


def fetch(url: str, dest: Path, token: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "safety-one-prep/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def fetch_all() -> None:
    for name, (url, rel) in RAW_FILES.items():
        dest = RAW / rel
        print(f"[fetch] {name}: {url}")
        fetch(url, dest)
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            fetch(VLGUARD_ZIP_URL, RAW / "vlguard" / "test.zip", token=token)
            print("[fetch] vlguard test.zip OK (gated, HF_TOKEN used)")
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] vlguard test.zip FAILED even with HF_TOKEN: {e}")
    else:
        print("[fetch] vlguard images skipped (gated; set HF_TOKEN to fetch test.zip)")


def row(rid, text, gold_label, gold_categories, source, image_path=None, lang="en"):
    assert gold_label in ("safe", "unsafe")
    assert set(gold_categories) <= KNOWN_CATEGORIES, gold_categories
    return {
        "id": rid,
        "text": text,
        "image_path": image_path,
        "lang": lang,
        "gold_label": gold_label,
        "gold_categories": gold_categories,
        "source": source,
        "split": "test",
    }


# Taxonomy slugs from data/TAXONOMY.md (26 categories).
KNOWN_CATEGORIES = {
    "criminal_planning", "violence", "hate_identity", "harassment",
    "needs_caution", "controlled_substances", "profanity", "sexual", "pii",
    "weapons", "self_harm", "fraud_deception", "illegal_activity",
    "sexual_minor", "other", "immoral_unethical", "misinformation", "threat",
    "unauthorized_advice", "manipulation", "malware", "high_risk_gov",
    "copyright", "economic_harm", "discrimination", "environmental_harm",
}

# OpenAI Moderation letter codes -> taxonomy slugs (TAXONOMY.md mapping table).
OMOD_MAP = {
    "S": ["sexual"],
    "S3": ["sexual_minor"],
    "H": ["hate_identity"],
    "H2": ["hate_identity", "threat"],
    "HR": ["harassment"],
    "V": ["violence"],
    "V2": ["violence"],
    "SH": ["self_harm"],
}

# NVIDIA violated_categories literal strings -> slugs (TAXONOMY.md nvidia_label col).
AEGIS_MAP = {
    "Criminal Planning/Confessions": "criminal_planning",
    "Violence": "violence",
    "Hate/Identity Hate": "hate_identity",
    "Harassment": "harassment",
    "Needs Caution": "needs_caution",
    "Controlled/Regulated Substances": "controlled_substances",
    "Profanity": "profanity",
    "Sexual": "sexual",
    "PII/Privacy": "pii",
    "Guns and Illegal Weapons": "weapons",
    "Suicide and Self Harm": "self_harm",
    "Fraud/Deception": "fraud_deception",
    "Illegal Activity": "illegal_activity",
    "Sexual (minor)": "sexual_minor",
    "Other": "other",
    "Immoral/Unethical": "immoral_unethical",
    "Political/Misinformation/Conspiracy": "misinformation",
    "Threat": "threat",
    "Unauthorized Advice": "unauthorized_advice",
    "Manipulation": "manipulation",
    "Malware": "malware",
    "High Risk Gov Decision Making": "high_risk_gov",
    "Copyright/Trademark/Plagiarism": "copyright",
    "Economic Harm": "economic_harm",
    "Discrimination": "discrimination",
    "Environmental Harm": "environmental_harm",
    # Extra variants observed in Aegis 2.0 test.json beyond the Nemotron-3.5 set:
    "Sexual (minor) - Exploitation": "sexual_minor",
    "Sexual - Non-consensual": "sexual",
    "Harassment - Non-sexual": "harassment",
    "Sexual Harassment": "harassment",
    "Threat - Self-harm": "self_harm",
    "Exploitation": "other",
    "Needs caution": "needs_caution",
}


def openai_moderation() -> list[dict]:
    out = []
    with gzip.open(RAW / "openai_moderation/samples-1680.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            cats = sorted({c for code, cs in OMOD_MAP.items() if r.get(code) == 1 for c in cs})
            out.append(row(f"omod-{i:04d}", r["prompt"], "unsafe" if cats else "safe",
                           cats, "openai_moderation"))
    return out


def wildguard() -> list[dict]:
    import pyarrow.parquet as pq  # noqa: PLC0415
    t = pq.read_table(RAW / "wildguard/wildguard_test.parquet")
    out = []
    for i, r in enumerate(t.to_pylist()):
        out.append(row(f"wildguard-{i:04d}", r["prompt"],
                       "unsafe" if r["label"] == "harmful" else "safe",
                       [], "wildguard_test"))
    return out


def xstest() -> list[dict]:
    out = []
    for r in csv.DictReader(open(RAW / "xstest/xstest_prompts.csv")):
        out.append(row(f"xstest-{r['id']}", r["prompt"], r["label"], [], "xstest"))
    return out


def hatemoji() -> list[dict]:
    out = []
    for r in csv.DictReader(open(RAW / "hatemoji/test.csv")):
        hate = r["label_gold"].strip() == "1"
        out.append(row(f"hatemoji-{r['case_id']}", r["text"],
                       "unsafe" if hate else "safe",
                       ["hate_identity"] if hate else [], "hatemojicheck"))
    return out


def aegis() -> list[dict]:
    out, unknown = [], set()
    for i, r in enumerate(json.load(open(RAW / "aegis/test.json"))):
        label = (r.get("prompt_label") or "safe").strip().lower()
        gold = "unsafe" if label == "unsafe" else "safe"
        cats = []
        for c in (r.get("violated_categories") or "").split(","):
            c = c.strip()
            if not c:
                continue
            slug = AEGIS_MAP.get(c)
            if slug is None:
                unknown.add(c)
                slug = "other"  # TAXONOMY.md: unknown -> other, logged not dropped
            if slug not in cats:
                cats.append(slug)
        out.append(row(f"aegis2-{i:04d}", r["prompt"], gold, cats, "aegis2_test"))
    if unknown:
        print(f"  [aegis] unknown category strings mapped to 'other': {sorted(unknown)}")
    return out


def vlguard() -> list[dict]:
    """Emit one row per instruction (safe + unsafe) so the binary task works.
    image_path points at the expected local path inside the gated test.zip."""
    out = []
    for r in json.load(open(RAW / "vlguard/test.json")):
        img = f"data/public/raw/vlguard/images/{r['image']}"
        for ir in r["instr-resp"]:
            # 'instruction' (no safe_/unsafe_ prefix) appears only on the 442
            # safe:false items = VLGuard's "unsafes" subset (unsafe image+instr).
            if "safe_instruction" in ir:
                instr, label = ir["safe_instruction"], "safe"
            elif "unsafe_instruction" in ir:
                instr, label = ir["unsafe_instruction"], "unsafe"
            else:
                instr, label = ir["instruction"], "unsafe"
            out.append(row(f"vlguard-{r['id']}-{label}",
                           instr, label, [], "vlguard", image_path=img))
    return out


BUILDERS = {
    "openai_moderation": openai_moderation,
    "wildguard_test": wildguard,
    "xstest": xstest,
    "hatemojicheck": hatemoji,
    "aegis2_test": aegis,
    "vlguard": vlguard,
}


def main() -> None:
    if "--fetch" in sys.argv:
        fetch_all()
    counts = {}
    for name, build in BUILDERS.items():
        rows = build()
        out = DATA / f"{name}.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_unsafe = sum(1 for r in rows if r["gold_label"] == "unsafe")
        n_img = sum(1 for r in rows if r["image_path"])
        counts[name] = (len(rows), n_unsafe, n_img)
        print(f"[write] {out.name}: {len(rows)} rows ({n_unsafe} unsafe, {n_img} with image)")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
