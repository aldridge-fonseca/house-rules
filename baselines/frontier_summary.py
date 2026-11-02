#!/usr/bin/env python3
"""Summarize the LIVE frontier-router baseline receipts.

Reads ``results/<run>/frontier-router-{a,b}.jsonl`` (last record per id +
input_hash wins), the plan file (row id -> dataset) and the spend ledger
(``results/spend.json``; the only place a content-filter refusal's billed cost
survives, because refusal receipts are error rows). Writes
``results/<run>/frontier_summary.json`` and ``FRONTIER_SUMMARY.md``.

Quality metrics are reported two ways:
  * ``refusals_excluded``  — provider refusals (content_filter / HTTP 400 policy
    blocks) and other errors are dropped from the denominator;
  * ``refusals_as_unsafe`` — a refusal counts as a block (label ``unsafe``);
    other errors are still dropped (reported separately as other-error rate).
No network, no spend.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.eval.metrics import _percentile, evaluate  # noqa: E402

ARMS = ("frontier-router-a", "frontier-router-b")
DATASETS = {"data/frozen/disguised_v1.jsonl": "disguised_v1",
            "data/public/openai_moderation.jsonl": "openai_moderation_sample300",
            "data/public/wildguard_test.jsonl": "wildguard_test_sample300",
            "data/public/xstest.jsonl": "xstest_sample300"}


def load_receipts(path: Path) -> list[dict]:
    last = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # blank or a line still being appended
                continue
            last[(str(r["id"]), str(r["input_hash"]))] = r
    return list(last.values())


def is_refusal(r: dict) -> bool:
    return "ContentFilter" in (r.get("error") or "")


def ledger_last_cost(ledger: dict) -> dict[tuple[str, str], float]:
    """(model, row_id) -> cost of the most recent settled call for that row."""
    out = {}
    for e in ledger.get("entries", []):
        md = e.get("metadata") or {}
        if e.get("provider") == "frontier-router" and md.get("row_id"):
            out[(md.get("model"), md["row_id"])] = float(e.get("cost_usd") or 0.0)
    return out


def ledger_model_totals(ledger: dict) -> dict[str, dict]:
    tot = defaultdict(lambda: {"calls": 0, "usd": 0.0})
    for e in ledger.get("entries", []):
        if e.get("provider") == "frontier-router":
            t = tot[(e.get("metadata") or {}).get("model") or "unknown"]
            t["calls"] += 1
            t["usd"] += float(e.get("cost_usd") or 0.0)
    return dict(tot)


def ledger_open_reservations(ledger: dict) -> dict[str, dict]:
    tot = defaultdict(lambda: {"count": 0, "usd": 0.0})
    for v in (ledger.get("reservations") or {}).values():
        if v.get("provider") == "frontier-router":
            t = tot[(v.get("metadata") or {}).get("model") or "unknown"]
            t["count"] += 1
            t["usd"] += float(v.get("amount_usd") or 0.0)
    return dict(tot)


def _usage(r: dict) -> tuple[int, int, int]:
    raw = (r.get("prediction") or {}).get("raw") or {}
    usage = (raw.get("response") or {}).get("usage") or {}
    reasoning = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    return int(raw.get("prompt_tokens") or 0), int(raw.get("completion_tokens") or 0), reasoning


def _quality(rows: list[dict]) -> dict:
    if not rows:
        return {"n_scored": 0}
    m = evaluate(rows, groups=False)
    xs = [r for r in rows if r.get("_dataset") == "xstest_sample300" and r.get("gold_label") == "safe"]
    xs_block = sum((r["prediction"]["label"] == "unsafe") for r in xs)
    pct = lambda v: None if v is None else round(100 * v, 2)  # noqa: E731
    return {"n_scored": m["n"], "accuracy": pct(m["accuracy"]), "macro_f1": pct(m["macro_f1"]),
            "unsafe_recall": pct(m["unsafe_recall"]), "unsafe_precision": pct(m["unsafe_precision"]),
            "over_block_rate": pct(m["over_block_rate"]),
            "xstest_over_block_rate": pct(xs_block / len(xs)) if xs else None,
            "xstest_safe_rows": len(xs)}


def summarize(rows: list[dict], costs: dict, model: str) -> dict:
    n = len(rows)
    ok = [r for r in rows if not r.get("error")]
    refused = [r for r in rows if is_refusal(r)]
    other = [r for r in rows if r.get("error") and not is_refusal(r)]
    as_unsafe = ok + [{**r, "prediction": {**r["prediction"], "label": "unsafe"}} for r in refused]
    lat = [float(r["prediction"].get("latency_ms") or 0) for r in ok]
    toks = [_usage(r) for r in ok]
    # Final-outcome cost: receipt cost for verdict rows; ledger cost of the
    # (billed) refusing call for refusal rows — refusal receipts store $0.
    usd_ok = sum(float(r["prediction"].get("cost_usd") or 0) for r in ok)
    usd_ref = sum(costs.get((model, r["id"]), 0.0) for r in refused)
    ref_missing = sum((model, r["id"]) not in costs for r in refused)
    gold_safe = [r for r in rows if r.get("gold_label") == "safe"]
    gold_unsafe = [r for r in rows if r.get("gold_label") == "unsafe"]
    rate = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
    return {
        "n": n, "verdicts": len(ok), "refusals": len(refused), "other_errors": len(other),
        "refusal_rate": rate(len(refused), n), "other_error_rate": rate(len(other), n),
        "refusal_rate_on_gold_safe": rate(sum(is_refusal(r) for r in gold_safe), len(gold_safe)),
        "refusal_rate_on_gold_unsafe": rate(sum(is_refusal(r) for r in gold_unsafe), len(gold_unsafe)),
        "refusals_excluded": _quality(ok),
        "refusals_as_unsafe": _quality(as_unsafe),
        "latency_p50_ms": round(_percentile(lat, 50), 1) if lat else None,
        "latency_p95_ms": round(_percentile(lat, 95), 1) if lat else None,
        "input_tokens": sum(t[0] for t in toks), "output_tokens": sum(t[1] for t in toks),
        "reasoning_tokens": sum(t[2] for t in toks),
        "usd_verdict_rows": round(usd_ok, 6), "usd_refusal_rows_ledger": round(usd_ref, 6),
        "usd_total": round(usd_ok + usd_ref, 6),
        "usd_per_1m_messages": round((usd_ok + usd_ref) / n * 1_000_000, 2) if n else None,
        "refusal_rows_without_ledger_cost": ref_missing,
    }


def build(run_dir: Path, ledger_path: Path) -> dict:
    plan = json.loads((run_dir / "frontier-router.plan.json").read_text())
    dataset_of = {i: DATASETS[f["path"]] for f in plan["files"] for i in f["ids"]}
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    costs = ledger_last_cost(ledger)
    out = {"label": "LIVE", "run": run_dir.name, "planned_rows_per_arm": plan["total_rows"],
           "ledger_by_model": ledger_model_totals(ledger),
           "ledger_open_reservations": ledger_open_reservations(ledger), "arms": {}}
    for arm in ARMS:
        path = run_dir / f"{arm}.jsonl"
        if not path.exists():
            continue
        rows = [r for r in load_receipts(path) if r["id"] in dataset_of]
        for r in rows:
            r["_dataset"] = dataset_of[r["id"]]
        models = sorted({r["prediction"]["model"] for r in rows if not r.get("error")})
        model = models[0] if len(models) == 1 else ",".join(models)
        groups = {"ALL": rows}
        for name in DATASETS.values():
            groups[name] = [r for r in rows if r["_dataset"] == name]
        groups["public_samples_combined"] = [r for r in rows if r["_dataset"] != "disguised_v1"]
        perts = sorted({r.get("perturbation") or "none" for r in groups["disguised_v1"]})
        for p in perts:
            groups[f"disguised_v1/{p}"] = [r for r in groups["disguised_v1"]
                                           if (r.get("perturbation") or "none") == p]
        out["arms"][arm] = {"model": model, "complete": len(rows) == plan["total_rows"],
                            "groups": {k: summarize(v, costs, model) for k, v in groups.items()}}
    return out


def _f(v, nd=1):
    return "–" if v is None else (f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}")


def render(doc: dict) -> str:
    L = ["# Frontier baseline summary — LIVE", "",
         "All numbers below are **LIVE** (measured on our own paid calls through the token "
         "router). Same shared moderation prompt as every other arm; "
         "`response_format=json_object`. Rows depend on the receipt set under `--run` "
         "(submission frontier receipts: `results/nvidia-v1-frontier/`, nvidia_test only). "
         "Generated by `baselines/frontier_summary.py`; machine-readable twin: "
         "`frontier_summary.json`.", "",
         "**Refusal** = the frontier provider's own safety layer declined to return a verdict "
         "(`finish_reason=content_filter` with empty content, or HTTP 400 policy block such as "
         "`cyber_policy`). It is a final outcome, not retried. Metrics are given two ways: "
         "**excl.** = refusals dropped; **as-unsafe** = refusal counted as a block. Other "
         "(transient) errors are dropped in both views and shown as their own rate.", "",
         "## Headline (all 3,122 rows per arm)", "",
         "| arm | model | refusal % | macro-F1 excl. / as-unsafe | unsafe recall excl. / as-unsafe "
         "| XSTest over-block excl. / as-unsafe | p50 / p95 ms | $ total | $ per 1M msgs |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, a in doc["arms"].items():
        g = a["groups"]["ALL"]
        e, u = g["refusals_excluded"], g["refusals_as_unsafe"]
        L.append(f"| {arm} | `{a['model']}` | {_f(g['refusal_rate'])} | {_f(e.get('macro_f1'))} / "
                 f"{_f(u.get('macro_f1'))} | {_f(e.get('unsafe_recall'))} / {_f(u.get('unsafe_recall'))} "
                 f"| {_f(e.get('xstest_over_block_rate'))} / {_f(u.get('xstest_over_block_rate'))} "
                 f"| {_f(g['latency_p50_ms'], 0)} / {_f(g['latency_p95_ms'], 0)} | {g['usd_total']:.2f} "
                 f"| {_f(g['usd_per_1m_messages'], 0)} |")
    L.append("")
    for arm, a in doc["arms"].items():
        g = a["groups"]["ALL"]
        L += [f"## {arm} — `{a['model']}`" + ("" if a["complete"] else " (INCOMPLETE)"), "",
              f"- rows {g['n']:,}; verdicts {g['verdicts']:,}; refusals {g['refusals']:,} "
              f"({_f(g['refusal_rate'], 2)}%; {_f(g['refusal_rate_on_gold_safe'], 2)}% of gold-safe, "
              f"{_f(g['refusal_rate_on_gold_unsafe'], 2)}% of gold-unsafe); other errors "
              f"{g['other_errors']} ({_f(g['other_error_rate'], 2)}%)",
              f"- tokens (verdict rows): input {g['input_tokens']:,}, output {g['output_tokens']:,}, "
              f"reasoning {g['reasoning_tokens']:,}",
              f"- cost: ${g['usd_total']:.4f} for the final outcomes (${g['usd_verdict_rows']:.4f} "
              f"verdict rows + ${g['usd_refusal_rows_ledger']:.4f} refusal rows from the ledger) → "
              f"**${_f(g['usd_per_1m_messages'], 0)} per 1M messages**", ""]
        L += ["| group | n | refusal % | other-err % | acc excl. | macro-F1 excl. | unsafe recall excl. "
              "| acc as-unsafe | macro-F1 as-unsafe | unsafe recall as-unsafe | XSTest over-block excl. / as-unsafe "
              "| p50 ms | p95 ms | in tok | out tok | reas tok | $ total | $/1M msgs |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, s in a["groups"].items():
            e, u = s["refusals_excluded"], s["refusals_as_unsafe"]
            xs = ("–" if e.get("xstest_over_block_rate") is None else
                  f"{_f(e['xstest_over_block_rate'])} / {_f(u['xstest_over_block_rate'])}")
            L.append(f"| {name} | {s['n']:,} | {_f(s['refusal_rate'])} | {_f(s['other_error_rate'])} "
                     f"| {_f(e.get('accuracy'))} | {_f(e.get('macro_f1'))} | {_f(e.get('unsafe_recall'))} "
                     f"| {_f(u.get('accuracy'))} | {_f(u.get('macro_f1'))} | {_f(u.get('unsafe_recall'))} "
                     f"| {xs} | {_f(s['latency_p50_ms'], 0)} | {_f(s['latency_p95_ms'], 0)} "
                     f"| {s['input_tokens']:,} | {s['output_tokens']:,} | {s['reasoning_tokens']:,} "
                     f"| {s['usd_total']:.4f} | {_f(s['usd_per_1m_messages'], 0)} |")
        L.append("")
    L += ["## Spend ledger (LIVE, `results/spend.json`, provider `frontier-router`)", "",
          "Every billed call, including superseded attempts (re-runs of malformed-JSON rows, "
          "earlier aborted runs, smokes) — so it is ≥ the final-outcome cost above.", "",
          "| model | billed calls | $ |", "|---|---:|---:|"]
    for model, t in sorted(doc["ledger_by_model"].items()):
        L.append(f"| `{model}` | {t['calls']:,} | {t['usd']:.4f} |")
    res = doc.get("ledger_open_reservations") or {}
    if res:
        L += ["", "Open (never-settled) reservations left by processes killed mid-call — conservative "
              "upper bounds that still count toward the frontier cap; actual billing for those "
              "in-flight calls is unknown:", "", "| model | open reservations | $ (upper bound) |",
              "|---|---:|---:|"]
        L += [f"| `{m}` | {t['count']:,} | {t['usd']:.4f} |" for m, t in sorted(res.items())]
    L += ["", "## Caveats", "",
          "- Latency and token totals cover verdict rows only: refusal receipts are error rows and "
          "do not keep the response (their billed cost is recovered from the ledger by row id; "
          "the last settled call for that row/model is used). HTTP 400 policy blocks "
          "(gpt-6.1-sol `cyber_policy`/`bio_policy`) are rejected before generation and are not "
          "billed, so they carry $0; Opus `content_filter` refusals are billed.",
          "- Accuracy / macro-F1 / unsafe recall / over-block are percentages (0-100), binary "
          "safe/unsafe, computed with `safety_one.eval.metrics.evaluate`. XSTest over-block = "
          "share of gold-safe rows in the XSTest 300-row sample labelled unsafe.",
          "- Public-benchmark frontier numbers in `data/BENCHMARKS.md` are PUBLISHED, not this run.",
          ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="nvidia-v1-frontier",
                    help="run id under results/ (frozen-v1 = earlier internal run, receipts not published)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    run_dir = ROOT / args.out / args.run
    doc = build(run_dir, ROOT / args.out / "spend.json")
    (run_dir / "frontier_summary.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (run_dir / "FRONTIER_SUMMARY.md").write_text(render(doc), encoding="utf-8")
    for arm, a in doc["arms"].items():
        g = a["groups"]["ALL"]
        print(f"{arm} {a['model']}: n={g['n']} refusals={g['refusals']} errors={g['other_errors']} "
              f"macroF1 excl={g['refusals_excluded'].get('macro_f1')} "
              f"as-unsafe={g['refusals_as_unsafe'].get('macro_f1')} ${g['usd_total']:.4f}")
    print(f"wrote {run_dir / 'FRONTIER_SUMMARY.md'} and frontier_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
