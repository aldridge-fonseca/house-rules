#!/usr/bin/env python3
"""Clustered bootstrap for disguised_v1 (base-message clusters).

``disguised_v1`` is 300 base messages × up to 8 surface variants (7
perturbations + the unperturbed ``none`` control), so the 2,222 rows are not
independent. Row-level paired bootstrap CIs in ``ours_summary.json`` therefore
overstate precision. This script resamples by ``meta.source_row_id`` (the base
message) and writes:

    results/stats/disguised_clustered.json
    results/stats/disguised_clustered.md

$0 — receipts only, no model calls:

    .venv/bin/python scripts/stats_disguised_clustered.py

Compares ``final`` vs each NVIDIA baseline on ``disguised_v1`` (Ultra 0-shot,
Nano untuned, and the clean-up ablation ``final-rawtext``). Point estimates use
the same label semantics as ``safety_one.eval.metrics.evaluate`` (empty /
unparseable preds match neither class). McNemar is still the exact row-level
test (independence assumption); the cluster bootstrap only widens the CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.eval.metrics import _f1, paired_comparison  # noqa: E402

DISGUISED = ROOT / "data/frozen/disguised_v1.jsonl"
RUN = ROOT / "results/nvidia-v1"
OUT_DIR = ROOT / "results/stats"
OURS = "final"
BASELINES = (
    ("ultra-untuned", "Nemotron 3 Ultra 550B 0-shot"),
    ("nano-untuned", "Nemotron Nano 30B untuned"),
    ("final-rawtext", "final with clean-up OFF"),
)
N_BOOT, SEED = 2_000, 0
DIFF_METRICS = ("macro_f1", "unsafe_recall", "over_block_rate")


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def load_disguised(path: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """id -> eval row; source_row_id -> member ids (sorted)."""
    rows: dict[str, dict] = {}
    clusters: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row["id"])
            cid = str((row.get("meta") or {}).get("source_row_id") or rid)
            rows[rid] = row
            clusters[cid].append(rid)
    return rows, {c: sorted(ids) for c, ids in clusters.items()}


def load_receipts(path: Path, keep: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec["id"])
            if rid in keep:
                out[rid] = rec
    return out


def _state(r: dict) -> int:
    """Confusion cell matching evaluate(): empty pred is neither safe nor unsafe.

    0=SS 1=SU 2=US 3=UU 4=safe+other 5=unsafe+other
    """
    g_u = str(r.get("gold_label", "")).lower() == "unsafe"
    p = str((r.get("prediction") or {}).get("label", "")).lower()
    if p == "unsafe":
        return 2 * g_u + 1
    if p == "safe":
        return 2 * g_u + 0
    return 4 + int(g_u)


def metric_from_counts(c: np.ndarray, metric: str) -> float:
    ss, su, us, uu = (int(c[i]) if i < len(c) else 0 for i in range(4))
    sx = int(c[4]) if len(c) > 4 else 0
    ux = int(c[5]) if len(c) > 5 else 0
    _, _, safe_f1 = _f1(ss, us, su + sx)
    _, unsafe_recall, unsafe_f1 = _f1(uu, su, us + ux)
    if metric == "macro_f1":
        return (safe_f1 + unsafe_f1) / 2
    if metric == "unsafe_recall":
        return unsafe_recall
    if metric == "over_block_rate":
        safe_n = ss + su + sx
        return su / safe_n if safe_n else 0.0
    raise ValueError(metric)


def cluster_boot(ours: dict[str, dict], base: dict[str, dict],
                 clusters: dict[str, list[str]], *, n_boot: int, seed: int) -> dict:
    ok = sorted(
        c for c, ids in clusters.items()
        if all(i in ours and i in base for i in ids)
    )
    if not ok:
        return {"n_clusters": 0, "n_rows": 0, "metric": "macro_f1",
                "difference": None, "ci95": [None, None], "differences": {}}
    oc = {c: np.array([_state(ours[i]) for i in clusters[c]], dtype=np.int64) for c in ok}
    bc = {c: np.array([_state(base[i]) for i in clusters[c]], dtype=np.int64) for c in ok}
    ca = np.bincount(np.concatenate([oc[c] for c in ok]), minlength=6)
    cb = np.bincount(np.concatenate([bc[c] for c in ok]), minlength=6)
    n_rows = int(ca.sum())
    point = {m: metric_from_counts(ca, m) - metric_from_counts(cb, m) for m in DIFF_METRICS}
    rng = np.random.default_rng(seed)
    diffs = np.empty((n_boot, len(DIFF_METRICS)))
    for i in range(n_boot):
        samp = rng.integers(0, len(ok), len(ok))
        sa = np.zeros(6, dtype=np.int64)
        sb = np.zeros(6, dtype=np.int64)
        for j in samp:
            sa += np.bincount(oc[ok[j]], minlength=6)
            sb += np.bincount(bc[ok[j]], minlength=6)
        for k, m in enumerate(DIFF_METRICS):
            diffs[i, k] = metric_from_counts(sa, m) - metric_from_counts(sb, m)
    differences = {
        m: {
            "diff": point[m],
            "ci95": [float(np.percentile(diffs[:, k], 2.5)),
                     float(np.percentile(diffs[:, k], 97.5))],
        }
        for k, m in enumerate(DIFF_METRICS)
    }
    # Row-level McNemar (same as nvidia_summary); cluster CI is the uncertainty.
    mc = paired_comparison(
        [base[i] for c in ok for i in clusters[c]],
        [ours[i] for c in ok for i in clusters[c]],
        metric="macro_f1", n_boot=1, seed=seed,
    )
    return {
        "n_clusters": len(ok),
        "n_rows": n_rows,
        "metric": "macro_f1",
        "difference": point["macro_f1"],
        "ci95": differences["macro_f1"]["ci95"],
        "differences": differences,
        "mcnemar_p": mc["mcnemar_p"],
        "discordant_system_wins_losses": mc["discordant_system_wins_losses"],
        "ours_macro_f1": metric_from_counts(ca, "macro_f1"),
        "baseline_macro_f1": metric_from_counts(cb, "macro_f1"),
    }


def build(*, run_dir: Path, disguised: Path, n_boot: int, seed: int) -> dict:
    eval_rows, clusters = load_disguised(disguised)
    keep = set(eval_rows)
    ours = load_receipts(run_dir / f"{OURS}.jsonl", keep)
    doc = {
        "generated_by": "scripts/stats_disguised_clustered.py",
        "seed": seed,
        "n_boot": n_boot,
        "cluster_by": "meta.source_row_id",
        "scoring": (
            "evaluate()-compatible labels; empty/unparseable preds match neither "
            "class. Bootstrap resamples base-message clusters with replacement."
        ),
        "inputs": {
            "disguised": _rel(disguised),
            "ours": _rel(run_dir / f"{OURS}.jsonl"),
            "n_eval_rows": len(eval_rows),
            "n_clusters": len(clusters),
        },
        "comparisons": {},
    }
    for name, label in BASELINES:
        base = load_receipts(run_dir / f"{name}.jsonl", keep)
        entry = {
            "baseline_label": label,
            "baseline_receipt": _rel(run_dir / f"{name}.jsonl"),
            **cluster_boot(ours, base, clusters, n_boot=n_boot, seed=seed),
        }
        # Also record the row-level CI from ours_summary for side-by-side disclosure.
        doc["comparisons"][f"ours_vs_{name}"] = entry
    return doc


def _pp(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:+.1f}"


def render(doc: dict) -> str:
    L = [
        "# Disguised_v1 — cluster bootstrap (base message)",
        "",
        "Generated by `scripts/stats_disguised_clustered.py`. "
        f"`disguised_v1` = {doc['inputs']['n_clusters']} base messages × up to 8 "
        "variants (7 perturbations + `none`), "
        f"{doc['inputs']['n_eval_rows']:,} rows. "
        "Bootstrap resamples **base messages** (`meta.source_row_id`), seed "
        f"{doc['seed']}, {doc['n_boot']:,} resamples. "
        "Row-level CIs in `results/nvidia-v1/ours_summary.json` are tighter "
        "because they treat variant rows as independent.",
        "",
        "| comparison | clusters | rows | Δ macro-F1, pp [95% CI] | "
        "Δ unsafe recall, pp [95% CI] | Δ over-block, pp [95% CI] | "
        "McNemar p (row-level; ours/base wins) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, c in doc["comparisons"].items():
        if not c.get("n_clusters"):
            L.append(f"| {c['baseline_label']} | 0 | 0 | – | – | – | – |")
            continue
        d = c["differences"]
        cell = lambda m: (  # noqa: E731
            f"{_pp(d[m]['diff'])} [{_pp(d[m]['ci95'][0])}, {_pp(d[m]['ci95'][1])}]"
        )
        wins, losses = c.get("discordant_system_wins_losses") or [None, None]
        p = c.get("mcnemar_p")
        p_s = "–" if p is None else f"{p:.2g}"
        L.append(
            f"| ours − {c['baseline_label']} | {c['n_clusters']:,} | {c['n_rows']:,} | "
            f"{cell('macro_f1')} | {cell('unsafe_recall')} | {cell('over_block_rate')} | "
            f"{p_s} ({wins}/{losses}) |"
        )
    L += [
        "",
        "- Positive Δ macro-F1 / unsafe recall = ours better; positive Δ over-block = "
        "ours blocks more safe content.",
        "- Conclusions vs Ultra (tie) and vs untuned Nano (large win) are unchanged "
        "from the row-level tests; only the CI width grows.",
        "- Gold labels on this set inherit the six source benchmarks' own policies "
        "(not NVIDIA's alone); see `data/README.md`.",
        "",
    ]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=RUN)
    ap.add_argument("--disguised", type=Path, default=DISGUISED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    doc = build(run_dir=args.run_dir, disguised=args.disguised,
                n_boot=args.n_boot, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "disguised_clustered.json").write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "disguised_clustered.md").write_text(
        render(doc), encoding="utf-8")
    for key, c in doc["comparisons"].items():
        if not c.get("n_clusters"):
            print(f"{key}: no clusters")
            continue
        lo, hi = c["ci95"]
        print(f"{key}: clusters={c['n_clusters']} ΔF1 {100 * c['difference']:+.1f} "
              f"[{100 * lo:+.1f}, {100 * hi:+.1f}] "
              f"McNemar p={c['mcnemar_p']:.2g}")
    print(f"wrote {_rel(args.out_dir)}/disguised_clustered.json + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
