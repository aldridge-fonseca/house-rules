#!/usr/bin/env python3
"""Paired uncertainty stats: our final system vs the frontier baselines (refusals = wrong).

Reads receipts — no model calls, $0:

    results/nvidia-v1/final.jsonl                          ours
    results/nvidia-v1-frontier/frontier-router-a.jsonl     GPT-6.1 Sol
    results/nvidia-v1-frontier/frontier-router-b.jsonl     Claude Opus 5.5

Writes ``results/stats/vs_frontier.json`` + ``vs_frontier.md``. One command,
idempotent — re-run to refresh:

    .venv/bin/python scripts/stats_vs_frontier.py

Metrics follow ``safety_one.eval.metrics.evaluate``; refusals/errors are scored
as wrong (pred = flipped gold), matching scripts/refusal_incorrect.py. The
95 % CI is a paired bootstrap (percentile, seeded); significance is the exact
two-sided McNemar on per-row correctness from ``metrics.paired_comparison``.
Cost conventions match nvidia_summary.py (ours: receipt tokens + GPU
allocation per model call) and frontier_summary.py (frontier refusals billed
via the spend ledger). Frontier latency is over verdict rows only — refusal
receipts carry latency_ms 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.eval.metrics import _f1, _percentile, evaluate, paired_comparison  # noqa: E402

OURS = ROOT / "results/nvidia-v1/final.jsonl"
TIMING = ROOT / "results/nvidia-v1/ours_timing.json"
SPEND = ROOT / "results/spend.json"
FRONTIER = {"gpt": ("results/nvidia-v1-frontier/frontier-router-a.jsonl", "GPT-6.1 Sol"),
            "opus": ("results/nvidia-v1-frontier/frontier-router-b.jsonl", "Claude Opus 5.5")}
OUT_DIR = ROOT / "results/stats"


def _rel(p) -> str:
    """Repo-relative path for receipts (never leak the local checkout path)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name
N_BOOT, SEED = 10_000, 0
DIFF_METRICS = ("macro_f1", "unsafe_recall", "over_block_rate")
_FLIP = {"safe": "unsafe", "unsafe": "safe"}


def load_receipts(path: Path) -> dict[str, dict]:
    """id -> last receipt. Tolerates a partially-written tail line."""
    last = {}
    if not path.exists():
        return last
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        last[str(r["id"])] = r
    return last


def is_refusal(r: dict) -> bool:
    return bool(r.get("error")) or str((r.get("prediction") or {}).get("label", "")).lower() \
        not in ("safe", "unsafe")


def scored_row(r: dict) -> dict:
    """Receipt with prediction.label forced to the scored label (refusal = wrong)."""
    if not is_refusal(r):
        return r
    p = dict(r.get("prediction") or {})
    p["label"] = _FLIP.get(str(r.get("gold_label", "")).lower(), "unsafe")
    return {**r, "prediction": p}


def _code(r: dict) -> int:
    """2x2 confusion cell: gold*2 + pred, safe=0 unsafe=1 -> SS SU US UU."""
    g = str(r.get("gold_label", "")).lower() == "unsafe"
    p = str((r.get("prediction") or {}).get("label", "")).lower() == "unsafe"
    return 2 * g + p


def metric_from_counts(c: np.ndarray, metric: str) -> float:
    """Same definitions as safety_one.eval.metrics.evaluate, from a 2x2 count."""
    ss, su, us, uu = (int(x) for x in c)
    _, _, safe_f1 = _f1(ss, us, su)
    _, unsafe_recall, unsafe_f1 = _f1(uu, su, us)
    if metric == "macro_f1":
        return (safe_f1 + unsafe_f1) / 2
    if metric == "unsafe_recall":
        return unsafe_recall
    if metric == "over_block_rate":
        return su / (ss + su) if ss + su else 0.0
    raise ValueError(metric)


def paired_boot_ci(ours: np.ndarray, base: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Percentile 95% CI of ours-baseline for each DIFF_METRIC; same resampled ids."""
    rng = np.random.default_rng(seed)
    n = len(ours)
    diffs = np.empty((n_boot, len(DIFF_METRICS)))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ca, cb = np.bincount(ours[idx], minlength=4), np.bincount(base[idx], minlength=4)
        for j, m in enumerate(DIFF_METRICS):
            diffs[i, j] = metric_from_counts(ca, m) - metric_from_counts(cb, m)
    return {m: [float(np.percentile(diffs[:, j], 2.5)), float(np.percentile(diffs[:, j], 97.5))]
            for j, m in enumerate(DIFF_METRICS)}


def ledger_costs(ledger: dict) -> dict[tuple[str, str], float]:
    out = {}
    for e in ledger.get("entries", []):
        md = e.get("metadata") or {}
        if e.get("provider") == "frontier-router" and md.get("row_id"):
            out[(md.get("model"), md["row_id"])] = float(e.get("cost_usd") or 0.0)
    return out


def ours_row_cost(r: dict, gpu: dict[str, float]) -> float:
    p = r.get("prediction") or {}
    cost = float(p.get("cost_usd") or 0.0)
    route = str(p.get("route") or "")
    if not r.get("error") and not route.startswith("rule"):
        ds = "disguised_v1" if str(r["id"]).startswith("rob-") else "nvidia_test"
        cost += gpu.get(ds, 0.0)
    return cost


def _gpu_rates(timing: dict) -> dict[str, float]:
    per = timing.get("final") or {}
    return {ds: float((per.get(ds) or {}).get("gpu_per_call_usd") or 0.0)
            for ds in ("nvidia_test", "disguised_v1")}


def _side_stats(rows: list[dict], cost_of) -> dict:
    m = evaluate(rows, groups=False)
    lat = [float((r.get("prediction") or {}).get("latency_ms") or 0) for r in rows if not r.get("error")]
    usd = sum(cost_of(r) for r in rows)
    n = len(rows)
    return {"n": n, "refusals": sum(is_refusal(r) for r in rows),
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "unsafe_recall": m["unsafe_recall"], "over_block_rate": m["over_block_rate"],
            "latency_p50_ms": _percentile(lat, 50), "latency_p95_ms": _percentile(lat, 95),
            "usd_total": usd, "usd_per_1m_msgs": usd / n * 1e6 if n else None}


def compare(ours: dict[str, dict], base: dict[str, dict], *, gpu: dict[str, float],
            ledger: dict[tuple[str, str], float] | None = None, base_model: str | None = None,
            n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    ids = sorted(set(ours) & set(base))
    if not ids:
        return {"n_paired": 0}
    o = [scored_row(ours[i]) for i in ids]
    b = [scored_row(base[i]) for i in ids]
    ledger = ledger or {}

    def base_cost(r):
        usd = float((r.get("prediction") or {}).get("cost_usd") or 0.0)
        if is_refusal(r) and base_model:
            usd = ledger.get((base_model, r["id"]), usd)
        return usd

    m_o = _side_stats(o, lambda r: ours_row_cost(r, gpu))
    m_b = _side_stats(b, base_cost)
    mc = paired_comparison(b, o, metric="macro_f1", n_boot=1, seed=seed)
    ci = paired_boot_ci(np.array([_code(r) for r in o]), np.array([_code(r) for r in b]),
                        n_boot=n_boot, seed=seed)

    def ratio(k):
        a, c = m_o[k], m_b[k]
        return a / c if a is not None and c else None

    return {"n_paired": len(ids),
            "ours": m_o, "baseline": m_b,
            "differences": {m: {"diff": m_o[m] - m_b[m], "ci95": ci[m]} for m in DIFF_METRICS},
            "mcnemar": {"ours_wins": mc["discordant_system_wins_losses"][0],
                        "baseline_wins": mc["discordant_system_wins_losses"][1],
                        "p_exact_two_sided": mc["mcnemar_p"]},
            "ratios": {"latency_p50_ours_over_baseline": ratio("latency_p50_ms"),
                       "latency_p95_ours_over_baseline": ratio("latency_p95_ms"),
                       "usd_per_1m_ours_over_baseline": ratio("usd_per_1m_msgs")}}


def build(ours_path: Path, frontier_paths: dict[str, Path], timing_path: Path,
          spend_path: Path, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    ours = load_receipts(ours_path)
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    ledger = ledger_costs(json.loads(spend_path.read_text())) if spend_path.exists() else {}
    doc = {"generated_by": "scripts/stats_vs_frontier.py", "seed": seed, "n_boot": n_boot,
           "scoring": "refusals/errors scored as wrong (pred = flipped gold); inner join on row id",
           "inputs": {"ours": _rel(ours_path),
                      **{k: _rel(p[0]) for k, p in frontier_paths.items()}},
           "ours_rows": len(ours), "comparisons": {}}
    for key, (path, label) in frontier_paths.items():
        base = load_receipts(path)
        verdict_models = sorted({str((r.get("prediction") or {}).get("model"))
                                 for r in base.values() if not r.get("error")})
        prefixes = {i.split("-", 1)[0] for i in base}
        expected = sum(1 for i in ours if i.split("-", 1)[0] in prefixes) if prefixes else 0
        entry = {"baseline_label": label, "baseline_rows": len(base),
                 "expected_rows": expected, "incomplete": len(base) < expected,
                 **compare(ours, base, gpu=_gpu_rates(timing), ledger=ledger,
                           base_model=verdict_models[0] if len(verdict_models) == 1 else None,
                           n_boot=n_boot, seed=seed)}
        doc["comparisons"][f"ours_vs_{key}"] = entry
    return doc


def _pp(v):
    return "–" if v is None else f"{100 * v:+.2f}"


def _p(p, d):
    if p is None:
        return "–"
    return "<1e-300" if p == 0 and d else f"{p:.3g}"


def render(doc: dict) -> str:
    L = ["# Ours vs frontier — paired stats (refusals = wrong)", "",
         "Generated by `scripts/stats_vs_frontier.py`. Inner join on row id; every "
         "refusal/error scored as wrong. Bootstrap: "
         f"{doc['n_boot']:,} paired resamples, seed {doc['seed']} (percentile 95% CI of "
         "ours − baseline). McNemar: exact two-sided binomial on discordant rows. "
         f"Our receipt file: {doc['ours_rows']:,} rows.", "",
         "| comparison | paired rows | Δ macro-F1, pp [95% CI] | Δ unsafe recall, pp [95% CI] "
         "| Δ over-block, pp [95% CI] | McNemar p (ours/base wins) | p50 lat × | $/1M × |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, c in doc["comparisons"].items():
        if not c.get("n_paired"):
            L.append(f"| {c['baseline_label']} | 0 | – | – | – | – | – | – |")
            continue
        d, mc, r = c["differences"], c["mcnemar"], c["ratios"]
        cell = lambda m: f"{_pp(d[m]['diff'])} [{_pp(d[m]['ci95'][0])}, {_pp(d[m]['ci95'][1])}]"  # noqa: E731
        x = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
        L.append(f"| ours − {c['baseline_label']} | {c['n_paired']:,} | {cell('macro_f1')} "
                 f"| {cell('unsafe_recall')} | {cell('over_block_rate')} "
                 f"| {_p(mc['p_exact_two_sided'], mc['ours_wins'] + mc['baseline_wins'])} "
                 f"({mc['ours_wins']}/{mc['baseline_wins']}) "
                 f"| {x(r['latency_p50_ours_over_baseline'])} "
                 f"| {x(r['usd_per_1m_ours_over_baseline'])} |")
    L += ["", "| system | rows used | refusals (=wrong) | macro-F1 | unsafe recall | over-block "
          "| p50 / p95 ms | $/1M msgs |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in doc["comparisons"].values():
        if not c.get("n_paired"):
            continue
        for name, s in (("ours", c["ours"]), (c["baseline_label"], c["baseline"])):
            ms = lambda v: "–" if v is None else f"{v:.0f}"  # noqa: E731
            L.append(f"| {name} | {s['n']:,} | {s['refusals']:,} | {100 * s['macro_f1']:.2f} "
                     f"| {100 * s['unsafe_recall']:.2f} | {100 * (s['over_block_rate'] or 0):.2f} "
                     f"| {ms(s['latency_p50_ms'])} / {ms(s['latency_p95_ms'])} "
                     f"| {ms(s['usd_per_1m_msgs'])} |")
    L += ["", "- Positive Δ = ours better for macro-F1 / unsafe recall, worse for over-block. "
          "McNemar wins/losses = discordant rows each side got right.",
          "- Ours $/1M = receipt per-token cost + GPU allocation per model call "
          "(`ours_timing.json`). Frontier $/1M = receipt cost + spend-ledger cost for refusal "
          "rows. Frontier latency over verdict rows only.", ""]
    for key, c in doc["comparisons"].items():
        if c.get("incomplete") and c.get("n_paired"):
            L.append(f"**Note:** `{key}` baseline is still being written "
                     f"({c['baseline_rows']:,} of {c['expected_rows']:,} rows so far) — re-run "
                     "this script to refresh.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=OURS)
    ap.add_argument("--frontier-a", type=Path, default=ROOT / FRONTIER["gpt"][0])
    ap.add_argument("--frontier-b", type=Path, default=ROOT / FRONTIER["opus"][0])
    ap.add_argument("--timing", type=Path, default=TIMING)
    ap.add_argument("--spend", type=Path, default=SPEND)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    doc = build(args.ours, {"gpt": (args.frontier_a, FRONTIER["gpt"][1]),
                            "opus": (args.frontier_b, FRONTIER["opus"][1])},
                args.timing, args.spend, n_boot=args.n_boot, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "vs_frontier.json").write_text(json.dumps(doc, indent=1) + "\n",
                                                 encoding="utf-8")
    (args.out_dir / "vs_frontier.md").write_text(render(doc), encoding="utf-8")
    for key, c in doc["comparisons"].items():
        if not c.get("n_paired"):
            print(f"{key}: no paired rows")
            continue
        d, mc = c["differences"], c["mcnemar"]
        print(f"{key}: n={c['n_paired']} ΔF1 {100 * d['macro_f1']['diff']:+.2f}pp "
              f"[{100 * d['macro_f1']['ci95'][0]:+.2f},{100 * d['macro_f1']['ci95'][1]:+.2f}] "
              f"McNemar p={mc['p_exact_two_sided']:.3g}")
    print(f"wrote {args.out_dir}/vs_frontier.json + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
