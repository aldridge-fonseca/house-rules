#!/usr/bin/env python3
"""results/nvidia-v1 OURS_SUMMARY tables from receipts ($0, no model calls).

    .venv/bin/python scripts/nvidia_summary.py   # writes ours_summary.json (md printed to stdout)

Dedicated-deployment calls carry $0 in receipts; GPU cost is allocated per
run as rate x measured run wall-clock / model calls (results/nvidia-v1/
ours_timing.json, one entry per system x dataset run, runs on a deployment
were sequential). Receipts reused from results/frozen-v1 use that run's
allocation (frozen-v1/nano_summary.json gpu_per_call_usd). Per-token costs
(Ultra escalation / zero-shot) are the measured values in the receipts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.eval.metrics import evaluate, paired_comparison  # noqa: E402
from safety_one.eval.runner import _hash  # noqa: E402

DATA = {"nvidia_test": "data/nvidia/test.jsonl", "disguised_v1": "data/frozen/disguised_v1.jsonl"}
HEADLINE = ("final", "nano-untuned", "ultra-untuned")
INTERNAL = ("specialist-only", "final-rawtext")


def load_rows():
    rows = {}
    for ds, rel in DATA.items():
        for line in (ROOT / rel).open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                rows[(str(row["id"]), _hash(row))] = (ds, row)
    return rows


def load_receipts(path: Path, rows, by_id):
    out = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                k = (str(rec["id"]), str(rec["input_hash"]))
                if k in rows:
                    out[k] = rec
                elif (k2 := by_id.get(str(rec["id"]))) and \
                        rec.get("text") == f"[redacted:{rec['id']}]" and \
                        rows[k2][1].get("text") == rec["text"]:
                    out[k2] = rec  # redacted WildGuard row: id + placeholder match
    return out


def model_calls(rec) -> int:
    route = str((rec.get("prediction") or {}).get("route", ""))
    return 0 if route.startswith("rule") or rec.get("error") else 1


def stats(recs, gpu_per_call: float):
    m = evaluate(recs, groups=False)
    n = len(recs)
    pred = [r.get("prediction") or {} for r in recs]
    tok = sum(float(p.get("cost_usd") or 0) for p in pred)
    gpu = gpu_per_call * sum(model_calls(r) for r in recs)
    pct = lambda x: None if x is None else round(100 * x, 1)  # noqa: E731
    return {"n": n, "errors": sum(1 for r in recs if r.get("error")),
            "accuracy": pct(m["accuracy"]), "macro_f1": pct(m["macro_f1"]),
            "unsafe_recall": pct(m["unsafe_recall"]), "over_block": pct(m["over_block_rate"]),
            "p50_ms": m["latency_p50_ms"], "p95_ms": m["latency_p95_ms"],
            "escalated_pct": pct(sum("escalation" in str(p.get("route", "")) for p in pred) / n) if n else None,
            "token_usd": round(tok, 4), "gpu_alloc_usd": round(gpu, 4),
            "usd_per_1m_msgs": round((tok + gpu) / n * 1e6, 1) if n else None}


def slices(ds, recs_rows):
    yield ds, [r for r, _ in recs_rows]
    if ds == "nvidia_test":
        yield "nvidia_test_non_english", [r for r, row in recs_rows if row.get("lang") != "en"]
    else:
        for p in sorted({row.get("perturbation") for _, row in recs_rows}):
            yield f"disguised_v1:{p}", [r for r, row in recs_rows if row.get("perturbation") == p]


def fmt(v, nd=1):
    return "–" if v is None else f"{v:,.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(ROOT / "results/nvidia-v1"))
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    rows = load_rows()
    by_id = {k[0]: k for k in rows}
    timing = json.loads((run / "ours_timing.json").read_text(encoding="utf-8"))
    out = {"label": "LIVE", "timing": timing, "systems": {}}
    md = ["| system | slice | n | err | acc | macro-F1 | unsafe recall | over-block | p50 ms | p95 ms "
          "| esc % | $/1M msgs |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in (*HEADLINE, *INTERNAL):
        rec = load_receipts(run / f"{name}.jsonl", rows, by_id)
        for ds in DATA:
            pairs = [(r, rows[k][1]) for k, r in rec.items() if rows[k][0] == ds]
            if not pairs:
                continue
            per = float((timing.get(name, {}).get(ds) or {}).get("gpu_per_call_usd") or 0.0)
            for sl, rs in slices(ds, pairs):
                s = stats(rs, per)
                out["systems"].setdefault(name, {})[sl] = s
                md.append(f"| {name} | {sl} | {s['n']:,} | {s['errors']} | {fmt(s['accuracy'])} "
                          f"| {fmt(s['macro_f1'])} | {fmt(s['unsafe_recall'])} | {fmt(s['over_block'])} "
                          f"| {fmt(s['p50_ms'], 0)} | {fmt(s['p95_ms'], 0)} | {fmt(s['escalated_pct'])} "
                          f"| {fmt(s['usd_per_1m_msgs'], 0)} |")
    md += ["", "Paired (same rows) macro-F1 difference `final` − other, bootstrap 95 % CI "
           "(1,000 resamples) + exact McNemar on correctness:", "",
           "| other | dataset | n | Δ macro-F1 | 95 % CI | McNemar p |", "|---|---|---:|---:|---|---:|"]
    final = load_receipts(run / "final.jsonl", rows, by_id)
    for other in ("ultra-untuned", "nano-untuned", "specialist-only", "final-rawtext"):
        rec = load_receipts(run / f"{other}.jsonl", rows, by_id)
        for ds in DATA:
            a = [r for k, r in rec.items() if rows[k][0] == ds and k in final]
            if not a:
                continue
            b = [final[k] for k, r in rec.items() if rows[k][0] == ds and k in final]
            c = paired_comparison(a, b, metric="macro_f1", n_boot=1000)
            lo, hi = c["ci95"]
            out.setdefault("paired_vs_final", {}).setdefault(other, {})[ds] = c
            md.append(f"| {other} | {ds} | {c['n_paired']:,} | {100 * c['difference']:+.1f} "
                      f"| [{100 * lo:+.1f}, {100 * hi:+.1f}] | {c['mcnemar_p']:.2g} |")
    (run / "ours_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
