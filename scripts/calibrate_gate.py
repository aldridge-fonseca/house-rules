#!/usr/bin/env python3
"""Choose the decision threshold + gray-zone band on DEV receipts only ($0, no model calls).

    .venv/bin/python scripts/calibrate_gate.py --dev-dir /tmp/s1-dev/dev-calib \
        --json-out results/dev-calib/calibration.json

Inputs (run_eval receipts on data/nvidia/filtered/dev_calib.jsonl, never test):
- specialist-only.jsonl: tuned Nano alone -> P(unsafe) per row.
- calib-escalate-all.jsonl: `full` wiring with band [0, 1] (scripts.nano_ladder),
  so every model-routed row carries raw.pre_escalation (specialist label/probs)
  and the NVIDIA Nemotron 3 Ultra verdict; rule-routed rows carry the rule verdict.

Search (maximise dev macro-F1):
- specialist-cal: t, label = unsafe iff P(unsafe) >= t.
- full-cal: (t, lo, hi), pre-gate label by t; rows with lo <= P(unsafe) <= hi take
  the Ultra label; subject to escalated rows <= --max-escalation of all rows.
Ties: lower escalation, then the band closest to t.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def metrics(gold, pred):
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = len(gold) - tp - fp - fn
    safe_n = tn + fp
    return {"n": len(gold), "macro_f1": (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2,
            "unsafe_recall": tp / (tp + fn) if tp + fn else None,
            "over_block": fp / safe_n if safe_n else None,
            "accuracy": (tp + tn) / len(gold) if gold else None}


def load(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


T_GRID = sorted({round(0.05 * i, 3) for i in range(1, 20)}
                | {0.025 * i for i in range(1, 40)}
                | {0.96, 0.97, 0.98, 0.99, 0.995, 0.999})
B_GRID = sorted({0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.95, 0.97, 0.98, 0.99,
                 0.995, 0.999, 1.0} | {round(0.05 * i, 3) for i in range(2, 19)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--max-escalation", type=float, default=0.20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    d = Path(args.dev_dir)
    spec = load(d / "specialist-only.jsonl")
    esc = load(d / "calib-escalate-all.jsonl")
    ids = [i for i in spec if i in esc and not spec[i].get("error") and not esc[i].get("error")]
    dropped = len(set(spec) | set(esc)) - len(ids)

    gold = [spec[i]["gold_label"] == "unsafe" for i in ids]
    sp = [float(spec[i]["prediction"]["probs"].get("unsafe", 0.0)) for i in ids]
    sp_gen = [spec[i]["prediction"]["label"] == "unsafe" for i in ids]
    rows = []  # (is_rule, p_pre, gen_pre, ultra_or_rule_label)
    for i in ids:
        pred = esc[i]["prediction"]
        pre = (pred.get("raw") or {}).get("pre_escalation")
        if pre is None:  # rule-routed (or no escalation recorded): final label stands
            rows.append((True, None, None, pred["label"] == "unsafe"))
        else:
            rows.append((False, float(pre["probs"].get("unsafe", 0.0)), pre["label"] == "unsafe",
                         pred["label"] == "unsafe"))
    n = len(ids)

    out = {"dev_file": "data/nvidia/filtered/dev_calib.jsonl", "rows": n, "dropped_error_rows": dropped,
           "max_escalation": args.max_escalation}
    out["specialist_generated_label"] = metrics(gold, sp_gen)
    best_t = round(max(T_GRID, key=lambda t: (metrics(gold, [p >= t for p in sp])["macro_f1"],
                                              -abs(t - 0.5))), 3)
    out["specialist_cal"] = {"unsafe_threshold": best_t, **metrics(gold, [p >= best_t for p in sp])}
    out["specialist_sweep"] = {f"{t:g}": round(metrics(gold, [p >= t for p in sp])["macro_f1"], 4)
                               for t in T_GRID}

    full_gen = [r[3] if r[0] else (r[3] if (r[2] != (r[1] >= 0.5) or 0.35 <= r[1] <= 0.65) else r[2])
                for r in rows]
    out["full_current_band_no_category_overrides"] = metrics(gold, full_gen)
    out["ultra_all_rows"] = metrics(gold, [r[3] for r in rows])

    # Model-routed rows sorted by P(unsafe): a band is a contiguous slice, so
    # tp/fp/fn for any (t, lo, hi) come from prefix sums in O(1).
    rule = [(g, r[3]) for g, r in zip(gold, rows) if r[0]]
    rtp = sum(g and u for g, u in rule)
    rfp = sum((not g) and u for g, u in rule)
    rfn = sum(g and (not u) for g, u in rule)
    model = sorted(((r[1], g, r[3]) for g, r in zip(gold, rows) if not r[0]), key=lambda x: x[0])
    P = [x[0] for x in model]

    def prefix(flags):
        acc = [0]
        for f in flags:
            acc.append(acc[-1] + f)
        return acc

    u_tp = prefix(g and u for _, g, u in model)
    u_fp = prefix((not g) and u for _, g, u in model)
    u_fn = prefix(g and (not u) for _, g, u in model)
    from bisect import bisect_left, bisect_right
    bands = [(lo, hi, bisect_left(P, lo), bisect_right(P, hi))
             for lo in B_GRID for hi in B_GRID if hi >= lo]
    prefixes = {}
    for t in T_GRID:
        prefixes[t] = (prefix(g and p >= t for p, g, _ in model),
                       prefix((not g) and p >= t for p, g, _ in model),
                       prefix(g and p < t for p, g, _ in model))

    def search(cap):
        best = None
        for t in T_GRID:
            s_tp, s_fp, s_fn = prefixes[t]
            for lo, hi, a, b in bands:
                k = b - a
                if k > cap * n:
                    continue
                tp = rtp + (u_tp[b] - u_tp[a]) + s_tp[-1] - (s_tp[b] - s_tp[a])
                fp = rfp + (u_fp[b] - u_fp[a]) + s_fp[-1] - (s_fp[b] - s_fp[a])
                fn = rfn + (u_fn[b] - u_fn[a]) + s_fn[-1] - (s_fn[b] - s_fn[a])
                tn = n - tp - fp - fn
                mf1 = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
                key = (round(mf1, 6), -k, -abs((lo + hi) / 2 - t))
                if best is None or key > best[0]:
                    best = (key, t, lo, hi, k)
        return best

    def forced(k_min):
        """Best (t, band) escalating at least k_min rows (diagnostic: what escalation costs)."""
        best = None
        for t in T_GRID:
            s_tp, s_fp, s_fn = prefixes[t]
            for lo, hi, a, b in bands:
                k = b - a
                if k < k_min or k > args.max_escalation * n:
                    continue
                tp = rtp + (u_tp[b] - u_tp[a]) + s_tp[-1] - (s_tp[b] - s_tp[a])
                fp = rfp + (u_fp[b] - u_fp[a]) + s_fp[-1] - (s_fp[b] - s_fp[a])
                fn = rfn + (u_fn[b] - u_fn[a]) + s_fn[-1] - (s_fn[b] - s_fn[a])
                tn = n - tp - fp - fn
                mf1 = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
                if best is None or mf1 > best[0]:
                    best = (mf1, t, lo, hi, k)
        return best

    out["cap_sweep"] = {}
    for cap in (0.0, 0.05, 0.10, args.max_escalation):
        key, t, lo, hi, k = search(cap)
        out["cap_sweep"][f"{cap:g}"] = {"macro_f1": key[0], "unsafe_threshold": round(t, 3),
                                        "gray_zone": [lo, hi], "escalation_rate": k / n}
    out["forced_min_escalation"] = {}
    for frac in (0.05, 0.10, 0.15):
        f = forced(int(frac * n))
        if f:
            out["forced_min_escalation"][f"{frac:g}"] = {
                "macro_f1": f[0], "unsafe_threshold": round(f[1], 3), "gray_zone": [f[2], f[3]],
                "escalation_rate": f[4] / n}
    _, t, lo, hi, k = search(args.max_escalation)
    t = round(t, 3)
    pred = [r[3] if r[0] else (r[3] if lo <= r[1] <= hi else r[1] >= t) for r in rows]
    out["full_cal"] = {"unsafe_threshold": t, "gray_zone": [lo, hi], "escalated": k,
                       "escalation_rate": k / n, **metrics(gold, pred)}
    print(json.dumps({k: v for k, v in out.items() if k != "specialist_sweep"}, indent=1))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
