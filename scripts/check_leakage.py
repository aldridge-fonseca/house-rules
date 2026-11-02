#!/usr/bin/env python3
"""Train <-> eval leakage check for the NVIDIA training files (stdlib only).

Compares every NVIDIA train row (and the dev/calibration split) against every
eval set we report on: data/public/*.jsonl, data/robustness/*.jsonl (both the
perturbed `text` and the clean `meta.orig_text`), plus the NVIDIA test and
validation splits as a re-check of prep_nvidia.py's own leakage pass.

Two collision types:
  exact     sha256 of normalized text (NFKC, casefold, zero-width strip,
            punctuation removed, whitespace collapsed; emoji/symbols kept)
  near_dup  Jaccard >= --threshold (default 0.8) on word-3gram shingles
            (char-4grams for texts with < 5 words, e.g. CJK/Thai). Exact
            all-pairs via prefix filtering, then verified Jaccard.

Originals are never rewritten. Filtered copies go to data/nvidia/filtered/:
  train_sft.jsonl, train_kway.jsonl  (rows aligned with the originals; a row
                                      is dropped from both if its text collides)
  dev_calib.jsonl
  leakage_report.json                 counts per eval source, examples

  python3 scripts/check_leakage.py            # check + write filtered files
  python3 scripts/check_leakage.py --no-write # report only
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NV = os.path.join(ROOT, "data", "nvidia")
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_WORD = re.compile(r"\w+", re.UNICODE)


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(_ZW).casefold()
    t = "".join(" " if unicodedata.category(c).startswith("P") else c for c in t)
    return " ".join(t.split())


def shingles(n: str) -> frozenset:
    words = _WORD.findall(n)
    if len(words) >= 5:
        return frozenset(hash(("w",) + tuple(words[i:i + 3])) for i in range(len(words) - 2))
    s = n.replace(" ", "")
    if len(s) < 4:
        return frozenset([hash(("c", s))]) if s else frozenset()
    return frozenset(hash(("c", s[i:i + 4])) for i in range(len(s) - 3))


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sft_text(row: dict) -> str:
    return next((m["content"] for m in row.get("messages", []) if m.get("role") == "user"), "")


def load_eval() -> list[tuple[str, str, str]]:
    """[(source_tag, row_id, text)] for every eval text."""
    out = []
    files = sorted(glob.glob(os.path.join(ROOT, "data", "public", "*.jsonl")))
    files += sorted(glob.glob(os.path.join(ROOT, "data", "robustness", "*.jsonl")))
    files += [os.path.join(NV, "test.jsonl"), os.path.join(NV, "validation.jsonl")]
    for path in files:
        tag = os.path.relpath(path, os.path.join(ROOT, "data"))
        for r in read_jsonl(path):
            out.append((tag, r.get("id", ""), r.get("text") or ""))
            orig = (r.get("meta") or {}).get("orig_text")
            if orig and orig != r.get("text"):
                out.append((tag + "#orig_text", r.get("id", ""), orig))
    return out


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


def prefix(sh: frozenset, order: dict, t: float) -> list:
    toks = sorted(sh, key=lambda x: (order.get(x, 0), x))
    k = len(toks) - math.ceil(t * len(toks)) + 1
    return toks[:max(k, 1)]


def check(train_texts: list[str], eval_items, t: float):
    """Return {train_idx: [(kind, source, eval_id, jaccard), ...]}."""
    ev_norm = [norm(x[2]) for x in eval_items]
    exact_idx = collections.defaultdict(list)
    for j, n in enumerate(ev_norm):
        if n:
            exact_idx[hashlib.sha256(n.encode()).hexdigest()].append(j)
    ev_sh = [shingles(n) for n in ev_norm]
    tr_norm = [norm(x) for x in train_texts]
    tr_sh = [shingles(n) for n in tr_norm]
    # Global token order: rarest first (frequency over both sides).
    freq = collections.Counter()
    for s in ev_sh:
        freq.update(s)
    for s in tr_sh:
        freq.update(s)
    order = dict(freq)
    pidx = collections.defaultdict(list)
    for j, s in enumerate(ev_sh):
        for tok in prefix(s, order, t):
            pidx[tok].append(j)

    hits: dict[int, list] = {}
    for i, (n, s) in enumerate(zip(tr_norm, tr_sh)):
        found = []
        seen_exact = set(exact_idx.get(hashlib.sha256(n.encode()).hexdigest(), [])) if n else set()
        for j in seen_exact:
            found.append(("exact", eval_items[j][0], eval_items[j][1], 1.0))
        if s:
            cands = set()
            for tok in prefix(s, order, t):
                cands.update(pidx.get(tok, ()))
            for j in cands - seen_exact:
                # Size filter before the set intersection.
                a, b = len(s), len(ev_sh[j])
                if min(a, b) < t * max(a, b):
                    continue
                jac = jaccard(s, ev_sh[j])
                if jac >= t:
                    found.append(("near_dup", eval_items[j][0], eval_items[j][1], round(jac, 4)))
        if found:
            hits[i] = found
    return hits


def summarize(name: str, texts: list[str], hits: dict, n_examples: int = 5) -> dict:
    by_src = collections.defaultdict(lambda: {"exact": set(), "near_dup": set()})
    for i, fs in hits.items():
        for kind, src, _, _ in fs:
            by_src[src.split("#")[0]][kind].add(i)
    exact_rows = {i for i, fs in hits.items() if any(k == "exact" for k, *_ in fs)}
    examples = []
    for i in sorted(hits)[:n_examples]:
        kind, src, eid, jac = hits[i][0]
        examples.append({"row": i, "text": texts[i][:160], "kind": kind,
                         "eval_source": src, "eval_id": eid, "jaccard": jac})
    return {
        "file": name,
        "rows": len(texts),
        "rows_flagged": len(hits),
        "rows_exact": len(exact_rows),
        "rows_near_dup_only": len(hits) - len(exact_rows),
        "rows_kept": len(texts) - len(hits),
        "by_eval_source": {k: {"exact_train_rows": len(v["exact"]),
                               "near_dup_train_rows": len(v["near_dup"] - v["exact"])}
                           for k, v in sorted(by_src.items())},
        "examples": examples,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out-dir", default=os.path.join(NV, "filtered"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    eval_items = load_eval()
    sft = list(read_jsonl(os.path.join(NV, "train_sft.jsonl")))
    kway = list(read_jsonl(os.path.join(NV, "train_kway.jsonl")))
    dev = list(read_jsonl(os.path.join(NV, "dev_calib.jsonl")))
    sft_t = [sft_text(r) for r in sft]
    kway_t = [r.get("state", "") for r in kway]
    aligned = len(sft_t) == len(kway_t) and all(a == b for a, b in zip(sft_t, kway_t))
    dev_t = [r.get("text") or "" for r in dev]

    train_hits = check(sft_t, eval_items, args.threshold)
    kway_hits = train_hits if aligned else check(kway_t, eval_items, args.threshold)
    dev_hits = check(dev_t, eval_items, args.threshold)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/check_leakage.py",
        "method": ("exact: sha256 of NFKC+casefold+zero-width/punctuation-stripped text; "
                   f"near_dup: Jaccard>={args.threshold} on word-3gram shingles (char-4gram if <5 words), "
                   "prefix-filter all-pairs, verified"),
        "eval_sets": dict(collections.Counter(s.split("#")[0] for s, _, _ in eval_items)),
        "eval_texts_compared": len(eval_items),
        "train_sft_kway_aligned": aligned,
        "train_sft": summarize("data/nvidia/train_sft.jsonl", sft_t, train_hits),
        "train_kway": summarize("data/nvidia/train_kway.jsonl", kway_t, kway_hits),
        "dev_calib": summarize("data/nvidia/dev_calib.jsonl", dev_t, dev_hits),
        "seconds": None,
    }
    if not args.no_write:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, rows, hits in (("train_sft.jsonl", sft, train_hits),
                                 ("train_kway.jsonl", kway, kway_hits),
                                 ("dev_calib.jsonl", dev, dev_hits)):
            path = os.path.join(args.out_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                for i, r in enumerate(rows):
                    if i not in hits:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(path, "rb") as f:
                report.setdefault("outputs", {})[os.path.relpath(path, ROOT)] = {
                    "rows": len(rows) - len(hits), "sha256": hashlib.sha256(f.read()).hexdigest()}
    report["seconds"] = round(time.time() - t0, 1)
    if not args.no_write:
        with open(os.path.join(args.out_dir, "leakage_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    for key in ("train_sft", "train_kway", "dev_calib"):
        s = report[key]
        print(f"{s['file']}: {s['rows']} rows, flagged {s['rows_flagged']} "
              f"(exact {s['rows_exact']}, near-dup only {s['rows_near_dup_only']}), kept {s['rows_kept']}")
        for src, c in s["by_eval_source"].items():
            print(f"    {src}: exact {c['exact_train_rows']}, near-dup {c['near_dup_train_rows']}")
    print(f"done in {report['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
