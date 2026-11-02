"""Pure-Python metrics and paired uncertainty estimates."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Sequence


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


GROUP_KEYS = ("perturbation", "route", "source")


def group_key(row: dict[str, Any], key: str) -> str:
    """Grouping value for a receipt row. `perturbation` absent -> "none"
    (standard eval rows carry no perturbation); `route` is the full trail,
    e.g. "specialist>escalation"."""
    if key == "perturbation":
        return str(row.get("perturbation") or "none")
    if key == "route":
        return str((row.get("prediction") or {}).get("route") or "unknown").lower()
    return str(row.get(key) or "unknown")


def evaluate_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Compact headline metrics per group (see GROUP_KEYS)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r, key)].append(r)
    out = {}
    for name in sorted(groups):
        m = evaluate(groups[name], groups=False)
        out[name] = {k: m[k] for k in ("n", "accuracy", "unsafe_f1", "unsafe_recall", "macro_f1",
                                        "over_block_rate", "safe_rows", "latency_p50_ms",
                                        "latency_p95_ms", "cost_per_1k_usd")}
    return out


def evaluate(rows: list[dict[str, Any]], *, groups: bool = True) -> dict[str, Any]:
    """Summarize receipt rows; each must have gold_label and prediction.label.

    With groups=True the result also carries by_perturbation / by_route /
    by_source breakdowns (skipped inside the bootstrap for speed)."""
    gold = [str(r.get("gold_label", "")).lower() for r in rows]
    pred = [str((r.get("prediction") or {}).get("label", "")).lower() for r in rows]
    labels = ("safe", "unsafe")
    f1s: dict[str, float] = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        _, _, f1s[label] = _f1(tp, fp, fn)
    utp = sum(g == "unsafe" and p == "unsafe" for g, p in zip(gold, pred))
    ufp = sum(g != "unsafe" and p == "unsafe" for g, p in zip(gold, pred))
    ufn = sum(g == "unsafe" and p != "unsafe" for g, p in zip(gold, pred))
    precision, recall, unsafe_f1 = _f1(utp, ufp, ufn)
    correct = [g == p for g, p in zip(gold, pred)]
    n = len(rows)
    safe_n = sum(g == "safe" for g in gold)
    overblocked = sum(g == "safe" and p == "unsafe" for g, p in zip(gold, pred))
    overblock_sources: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r, g, p in zip(rows, gold, pred):
        src = str(r.get("source", "unknown"))
        if g == "safe":
            overblock_sources[src][1] += 1
            overblock_sources[src][0] += p == "unsafe"
    cat_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        truth = set(r.get("gold_categories") or [])
        guess = set((r.get("prediction") or {}).get("categories") or [])
        for category in truth | guess:
            c = cat_counts[category]
            c[0] += category in truth and category in guess
            c[1] += category not in truth and category in guess
            c[2] += category in truth and category not in guess
    categories = {k: {"precision": _f1(*v)[0], "recall": _f1(*v)[1], "f1": _f1(*v)[2],
                      "support": v[0] + v[2]} for k, v in sorted(cat_counts.items())}
    routes: dict[str, list[bool]] = defaultdict(list)
    for r, ok in zip(rows, correct):
        route = str((r.get("prediction") or {}).get("route", "unknown")).lower()
        found = {part for part in route.split(">") if part in ("rule", "specialist", "vision", "multilingual", "escalation")}
        if not found:
            found = {route}
        for part in found:
            routes[part].append(ok)
    route_names = ("rule", "specialist", "vision", "multilingual", "escalation")
    route_metrics = {k: {"share": len(routes[k]) / n if n else 0.0,
                         "accuracy": sum(routes[k]) / len(routes[k]) if routes[k] else None, "n": len(routes[k])}
                     for k in route_names}
    latencies = [float((r.get("prediction") or {}).get("latency_ms", 0) or 0) for r in rows]
    costs = [float((r.get("prediction") or {}).get("cost_usd", 0) or 0) for r in rows]
    cost_total = sum(costs)
    grouped = {f"by_{k}": evaluate_by(rows, k) for k in GROUP_KEYS} if groups else {}
    return {**grouped, "n": n, "accuracy": sum(correct) / n if n else None,
            "unsafe_precision": precision, "unsafe_recall": recall, "unsafe_f1": unsafe_f1,
            "safe_f1": f1s["safe"], "macro_f1": sum(f1s.values()) / 2,
            "per_category": categories, "over_block_rate": overblocked / safe_n if safe_n else None,
            "over_block_by_source": {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(overblock_sources.items())},
            "safe_rows": safe_n, "routes": route_metrics,
            "latency_p50_ms": _percentile(latencies, 50), "latency_p95_ms": _percentile(latencies, 95),
            "cost_total_usd": cost_total, "cost_per_1k_usd": cost_total * 1000 / n if n else None}


def _stat(rows: list[dict[str, Any]], metric: str) -> float:
    return float(evaluate(rows, groups=False)[metric] or 0.0)


def paired_comparison(base_rows: list[dict[str, Any]], system_rows: list[dict[str, Any]], *,
                      metric: str = "unsafe_f1", n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Join by id, paired bootstrap CI for aggregate-metric difference, exact McNemar."""
    base = {r["id"]: r for r in base_rows}
    system = {r["id"]: r for r in system_rows}
    ids = sorted(base.keys() & system.keys())
    if not ids:
        return {"n_paired": 0, "metric": metric, "difference": None, "ci95": [None, None], "mcnemar_p": None}
    a, b = [system[i] for i in ids], [base[i] for i in ids]
    difference = _stat(a, metric) - _stat(b, metric)
    rng, diffs = random.Random(seed), []
    for _ in range(max(1, n_boot)):
        indices = [rng.randrange(len(ids)) for _ in ids]
        diffs.append(_stat([a[i] for i in indices], metric) - _stat([b[i] for i in indices], metric))
    lo, hi = _percentile(diffs, 2.5), _percentile(diffs, 97.5)
    # Discordant correctness pairs; two-sided exact binomial tail.
    wins = sum((system[i]["prediction"].get("label") == system[i]["gold_label"]) and
               (base[i]["prediction"].get("label") != base[i]["gold_label"]) for i in ids)
    losses = sum((system[i]["prediction"].get("label") != system[i]["gold_label"]) and
                 (base[i]["prediction"].get("label") == base[i]["gold_label"]) for i in ids)
    d = wins + losses
    p = 1.0 if d == 0 else min(1.0, 2 * sum(math.comb(d, k) for k in range(min(wins, losses) + 1)) / (2 ** d))
    return {"n_paired": len(ids), "metric": metric, "difference": difference,
            "ci95": [lo, hi], "mcnemar_p": p, "discordant_system_wins_losses": [wins, losses]}
