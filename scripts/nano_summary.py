#!/usr/bin/env python3
"""Build results/frozen-v1/NANO_SUMMARY tables from existing receipts ($0, no model calls).

    .venv/bin/python scripts/nano_summary.py --wall-s 300 > /tmp/nano_tables.md

Specialist calls run on a dedicated 1x B200 deployment (billed per GPU-second,
no per-token price), so receipts carry $0 for them; this script allocates
GPU cost = rate x measured ladder wall-clock / specialist calls and adds the
measured per-token cost already in the receipts (Ultra escalation / zero-shot).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.eval.metrics import evaluate  # noqa: E402
from safety_one.eval.runner import _hash  # noqa: E402

DATA = {
    "openai_moderation": "data/public/openai_moderation.jsonl",
    "wildguard_test": "data/public/wildguard_test.jsonl",
    "xstest": "data/public/xstest.jsonl",
    "disguised_v1": "data/frozen/disguised_v1.jsonl",
}
LADDER = ("specialist-only", "+rules", "+router", "full")
FRONTIER = ("frontier-router-a", "frontier-router-b")


def load_rows():
    keys, order = {}, []
    for name, rel in DATA.items():
        for line in (ROOT / rel).open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                k = (str(row["id"]), _hash(row))
                keys[k] = name
                order.append(k)
    return keys, order


def load_receipts(path: Path, keys):
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        k = (str(rec["id"]), str(rec["input_hash"]))
        if k in keys:
            out[k] = rec  # last write wins (resumed runs append the retry)
    return out


def is_refusal(rec) -> bool:
    return "ContentFilter" in (rec.get("error") or "")


def specialist_calls(rec) -> int:
    route = str((rec.get("prediction") or {}).get("route", ""))
    return 0 if route.startswith("rule") or rec.get("error") else 1


def fmt(x, scale=1.0, nd=1):
    return "–" if x is None else f"{x * scale:,.{nd}f}"


def row_stats(recs, gpu_per_call: float, refusal_as_unsafe: bool = False):
    rows = []
    for r in recs:
        if refusal_as_unsafe and is_refusal(r):
            r = {**r, "prediction": {**(r.get("prediction") or {}), "label": "unsafe"}}
        rows.append(r)
    m = evaluate(rows, groups=False)
    n = len(rows)
    errs = sum(1 for r in recs if r.get("error") and not is_refusal(r))
    refusals = sum(1 for r in recs if is_refusal(r))
    esc = sum(1 for r in recs if "escalation" in str((r.get("prediction") or {}).get("route", "")))
    token_cost = sum(float((r.get("prediction") or {}).get("cost_usd") or 0) for r in recs)
    gpu_cost = gpu_per_call * sum(specialist_calls(r) for r in recs)
    per_m = (token_cost + gpu_cost) / n * 1e6 if n else None
    return {"n": n, "errors": errs, "refusals": refusals, "macro_f1": m["macro_f1"],
            "recall": m["unsafe_recall"], "overblock": m["over_block_rate"],
            "p50": m["latency_p50_ms"], "p95": m["latency_p95_ms"], "esc": esc / n if n else None,
            "token_cost": token_cost, "gpu_cost": gpu_cost, "per_m": per_m}


HEADER = ("| system | dataset | n | err | macro-F1 | unsafe recall | over-block | p50 ms | p95 ms "
          "| esc % | $ tokens | $ GPU alloc | $/1M msgs |\n"
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def line(system, dataset, s):
    return (f"| {system} | {dataset} | {s['n']:,} | {s['errors'] + s['refusals']} | {fmt(s['macro_f1'], 100)} "
            f"| {fmt(s['recall'], 100)} | {fmt(s['overblock'], 100)} | {fmt(s['p50'], nd=0)} "
            f"| {fmt(s['p95'], nd=0)} | {fmt(s['esc'], 100)} | {s['token_cost']:.4f} | {s['gpu_cost']:.4f} "
            f"| {fmt(s['per_m'], nd=0)} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(ROOT / "results/frozen-v1"))
    ap.add_argument("--wall-s", type=float, required=True,
                    help="measured wall-clock seconds of the ladder run on the deployment")
    ap.add_argument("--gpu-rate-usd-h", type=float, default=13.0, help="1x B200 on-demand $/h")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--ladder", nargs="+", default=list(LADDER),
                    help="specialist-backed systems whose calls share --wall-s (default: r4 ladder)")
    args = ap.parse_args(argv)
    ladder = tuple(args.ladder)
    run_dir = Path(args.run_dir)
    keys, order = load_rows()
    rec = {name: load_receipts(run_dir / f"{name}.jsonl", keys)
           for name in (*ladder, "nvidia-ultra-zeroshot", *FRONTIER)}
    calls = sum(specialist_calls(r) for name in ladder for r in rec[name].values())
    gpu_per_call = args.gpu_rate_usd_h / 3600 * args.wall_s / calls if calls else 0.0
    out = [f"Specialist calls in ladder: {calls:,} over {args.wall_s:.0f} s wall-clock "
           f"=> {calls / args.wall_s:.1f} req/s on 1x B200; allocated GPU cost "
           f"${gpu_per_call * 1e6:,.0f} per 1M specialist calls (${args.gpu_rate_usd_h}/h).", "",
           "### Full frozen files (LIVE)", "", HEADER]
    js = {"gpu_per_call_usd": gpu_per_call, "specialist_calls": calls, "wall_s": args.wall_s,
          "full": {}, "matched": {}}
    for name in (*ladder, "nvidia-ultra-zeroshot"):
        if not rec[name]:
            continue
        per = 0.0 if name == "nvidia-ultra-zeroshot" else gpu_per_call
        groups = {"ALL": [rec[name][k] for k in order if k in rec[name]]}
        for ds in DATA:
            groups[ds] = [rec[name][k] for k in order if k in rec[name] and keys[k] == ds]
        for ds, rs in groups.items():
            if rs:
                s = row_stats(rs, per)
                js["full"].setdefault(name, {})[ds] = s
                out.append(line(name, ds, s))
    # Matched rows: exactly the rows the frontier arms ran (disguised_v1 + 300/public seeded).
    matched = [k for k in order if all(k in rec[f] for f in FRONTIER)]
    if matched:
        out += ["", f"### Matched rows vs frontier ({len(matched):,} rows both frontier arms ran; "
                "frontier refusals counted as unsafe)", "", HEADER]
        for name in (*ladder, "nvidia-ultra-zeroshot", *FRONTIER):
            have = [k for k in matched if k in rec[name]]
            if not have:
                continue
            per = gpu_per_call if name in ladder else 0.0
            for ds in ("ALL", *DATA):
                rs = [rec[name][k] for k in have if ds == "ALL" or keys[k] == ds]
                if rs:
                    s = row_stats(rs, per, refusal_as_unsafe=name in FRONTIER)
                    js["matched"].setdefault(name, {})[ds] = s
                    out.append(line(name, ds, s))
    print("\n".join(out))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(js, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
