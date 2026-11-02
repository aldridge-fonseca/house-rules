"""Tests for scripts/stats_vs_frontier.py (paired stats vs frontier baselines)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import scripts.stats_vs_frontier as svf
from safety_one.eval.metrics import evaluate


def rec(rid, gold, pred, *, error=None, route="frontier-router", lat=100.0, cost=0.001):
    return {"id": rid, "gold_label": gold, "error": error,
            "prediction": {"label": pred, "route": route, "latency_ms": lat, "cost_usd": cost,
                           "model": "test-model"}}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    ours = [rec("r1", "safe", "safe"), rec("r2", "unsafe", "unsafe"),
            rec("r3", "safe", "safe"), rec("r4", "unsafe", "unsafe"),
            rec("r5", "safe", "safe")]
    base = [rec("r1", "safe", "unsafe"), rec("r2", "unsafe", "unsafe"),
            rec("r3", "safe", "", error="FrontierContentFilter: blocked", lat=0.0, cost=0.0),
            rec("r4", "unsafe", "unsafe")]
    op = tmp_path / "ours.jsonl"
    bp = tmp_path / "frontier-router-a.jsonl"
    write_jsonl(op, ours)
    write_jsonl(bp, base)
    tp = tmp_path / "timing.json"
    tp.write_text(json.dumps({"final": {"nvidia_test": {"gpu_per_call_usd": 1e-4},
                                        "disguised_v1": {"gpu_per_call_usd": 2e-4}}}))
    sp = tmp_path / "spend.json"
    sp.write_text(json.dumps({"entries": [{"provider": "frontier-router", "cost_usd": 0.5,
                                           "metadata": {"model": "test-model", "row_id": "r3"}}]}))
    return op, bp, tp, sp


def test_load_receipts_dedupes_and_skips_bad_tail(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(rec("a", "safe", "safe")) + "\n" +
                 json.dumps(rec("a", "safe", "unsafe")) + "\n" +
                 '{"id": "truncat', encoding="utf-8")
    rows = svf.load_receipts(p)
    assert len(rows) == 1 and rows["a"]["prediction"]["label"] == "unsafe"


def test_refusal_scored_wrong():
    r = rec("x", "safe", "", error="FrontierContentFilter: blocked")
    assert svf.is_refusal(r)
    s = svf.scored_row(r)
    assert s["prediction"]["label"] == "unsafe"          # flipped gold -> wrong
    assert svf.scored_row(rec("y", "unsafe", "", error="e"))["prediction"]["label"] == "safe"


def test_metric_from_counts_matches_evaluate(fixture):
    op, bp, _, _ = fixture
    ours, base = svf.load_receipts(op), svf.load_receipts(bp)
    ids = sorted(set(ours) & set(base))
    for rows in ([svf.scored_row(ours[i]) for i in ids], [svf.scored_row(base[i]) for i in ids]):
        m = evaluate(rows, groups=False)
        c = np.bincount(np.array([svf._code(r) for r in rows]), minlength=4)
        for metric in svf.DIFF_METRICS:
            assert svf.metric_from_counts(c, metric) == pytest.approx(m[metric] or 0.0)


def test_compare_values(fixture):
    op, bp, tp, sp = fixture
    ours, base = svf.load_receipts(op), svf.load_receipts(bp)
    ledger = svf.ledger_costs(json.loads(sp.read_text()))
    c = svf.compare(ours, base, gpu={"nvidia_test": 1e-4, "disguised_v1": 0.0},
                    ledger=ledger, base_model="test-model", n_boot=200, seed=0)
    assert c["n_paired"] == 4                          # inner join: r5 dropped
    assert c["baseline"]["refusals"] == 1 and c["ours"]["refusals"] == 0
    d = c["differences"]
    assert d["macro_f1"]["diff"] == pytest.approx(1 - (0 + 2 / 3) / 2)
    assert d["unsafe_recall"]["diff"] == pytest.approx(0.0)
    assert d["over_block_rate"]["diff"] == pytest.approx(-1.0)
    mc = c["mcnemar"]
    assert (mc["ours_wins"], mc["baseline_wins"]) == (2, 0)
    assert mc["p_exact_two_sided"] == pytest.approx(0.5)  # exact binomial, d=2
    # frontier refusal row cost recovered from ledger; ours = receipt + gpu alloc
    assert c["baseline"]["usd_total"] == pytest.approx(0.001 * 3 + 0.5)
    assert c["ours"]["usd_total"] == pytest.approx(4 * (0.001 + 1e-4))


def test_bootstrap_seeded_and_brackets(fixture):
    op, bp, _, _ = fixture
    ours, base = svf.load_receipts(op), svf.load_receipts(bp)
    kw = dict(gpu={}, ledger={}, base_model="m", n_boot=500, seed=7)
    c1 = svf.compare(ours, base, **kw)
    c2 = svf.compare(ours, base, **kw)
    assert c1["differences"] == c2["differences"]       # seeded -> deterministic
    for m, d in c1["differences"].items():
        lo, hi = d["ci95"]
        assert lo <= hi and -1 <= lo and hi <= 1
    lo, hi = c1["differences"]["macro_f1"]["ci95"]
    assert lo <= d_point(c1) <= hi + 1e-9 or hi >= lo   # wide enough on n=4


def d_point(c):
    return c["differences"]["macro_f1"]["diff"]


def test_build_and_render(tmp_path, fixture):
    op, bp, tp, sp = fixture
    missing = tmp_path / "frontier-router-b.jsonl"
    doc = svf.build(op, {"gpt": (bp, "GPT"), "opus": (missing, "Opus")}, tp, sp,
                    n_boot=50, seed=0)
    assert doc["ours_rows"] == 5 and doc["seed"] == 0
    assert doc["comparisons"]["ours_vs_gpt"]["n_paired"] == 4
    assert doc["comparisons"]["ours_vs_opus"]["n_paired"] == 0
    md = svf.render(doc)
    assert "seed 0" in md and "McNemar" in md and "ours − GPT" in md
