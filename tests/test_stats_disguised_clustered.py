"""Tests for scripts/stats_disguised_clustered.py (base-message cluster bootstrap)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.stats_disguised_clustered as sdc
from safety_one.eval.metrics import evaluate


def rec(rid, gold, pred, *, error=None):
    return {"id": rid, "gold_label": gold, "error": error,
            "prediction": {"label": pred, "route": "specialist"}}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_state_and_metric_match_evaluate():
    rows = [
        rec("a", "safe", "safe"),
        rec("b", "safe", "unsafe"),
        rec("c", "unsafe", "unsafe"),
        rec("d", "unsafe", "safe"),
        rec("e", "safe", "", error="unparseable"),
        rec("f", "unsafe", "", error="unparseable"),
    ]
    m = evaluate(rows, groups=False)
    c = np.bincount([sdc._state(r) for r in rows], minlength=6)
    assert sdc.metric_from_counts(c, "macro_f1") == pytest.approx(m["macro_f1"] or 0.0)
    assert sdc.metric_from_counts(c, "unsafe_recall") == pytest.approx(m["unsafe_recall"] or 0.0)
    assert sdc.metric_from_counts(c, "over_block_rate") == pytest.approx(m["over_block_rate"] or 0.0)


def test_cluster_boot_seeded_and_wider_than_point(tmp_path):
    # Two bases × two variants; ours perfect on base A, baseline perfect on B.
    disguised = [
        {"id": "rob-none-a", "gold_label": "unsafe", "perturbation": "none",
         "meta": {"source_row_id": "base-a"}},
        {"id": "rob-leet-a", "gold_label": "unsafe", "perturbation": "leetspeak",
         "meta": {"source_row_id": "base-a"}},
        {"id": "rob-none-b", "gold_label": "safe", "perturbation": "none",
         "meta": {"source_row_id": "base-b"}},
        {"id": "rob-leet-b", "gold_label": "safe", "perturbation": "leetspeak",
         "meta": {"source_row_id": "base-b"}},
    ]
    ours = [rec("rob-none-a", "unsafe", "unsafe"), rec("rob-leet-a", "unsafe", "unsafe"),
            rec("rob-none-b", "safe", "unsafe"), rec("rob-leet-b", "safe", "unsafe")]
    base = [rec("rob-none-a", "unsafe", "safe"), rec("rob-leet-a", "unsafe", "safe"),
            rec("rob-none-b", "safe", "safe"), rec("rob-leet-b", "safe", "safe")]
    dpath = tmp_path / "disguised.jsonl"
    write_jsonl(dpath, disguised)
    run = tmp_path / "run"
    run.mkdir()
    write_jsonl(run / "final.jsonl", ours)
    write_jsonl(run / "ultra-untuned.jsonl", base)
    write_jsonl(run / "nano-untuned.jsonl", base)
    write_jsonl(run / "final-rawtext.jsonl", base)
    out = tmp_path / "stats"
    doc = sdc.build(run_dir=run, disguised=dpath, n_boot=200, seed=0)
    c = doc["comparisons"]["ours_vs_ultra-untuned"]
    assert c["n_clusters"] == 2 and c["n_rows"] == 4
    c2 = sdc.build(run_dir=run, disguised=dpath, n_boot=200, seed=0)
    assert c2["comparisons"]["ours_vs_ultra-untuned"]["ci95"] == c["ci95"]
    lo, hi = c["ci95"]
    assert lo <= c["difference"] <= hi
    # Paths in the doc must be repo-relative or basename — never absolute.
    assert not str(doc["inputs"]["disguised"]).startswith("/")
    sdc.main(["--run-dir", str(run), "--disguised", str(dpath),
              "--out-dir", str(out), "--n-boot", "50", "--seed", "0"])
    assert (out / "disguised_clustered.json").is_file()
    assert (out / "disguised_clustered.md").is_file()
    md = (out / "disguised_clustered.md").read_text(encoding="utf-8")
    assert "cluster bootstrap" in md.lower()
    assert "/home/" not in md


def test_randomization_preserves_cluster_dependence():
    # Two independent base messages: safe and unsafe. Ours is always right,
    # baseline always wrong. Of four possible cluster swaps, exactly two
    # retain |delta F1| = 1, so the exact two-sided p is 1/2.
    point = {"macro_f1": 1.0, "unsafe_recall": 1.0, "over_block_rate": -1.0}
    results = []
    for variants in (1, 8):
        ours = {"safe": np.zeros(variants, dtype=int),
                "unsafe": np.full(variants, 3, dtype=int)}
        base = {"safe": np.ones(variants, dtype=int),
                "unsafe": np.full(variants, 2, dtype=int)}
        results.append(sdc.cluster_rand_p(
            ours, base, ["safe", "unsafe"], point, n_perm=10_000, seed=19))
    assert results[0]["macro_f1"] == pytest.approx(0.5, abs=0.02)
    # Duplicating correlated variants adds no independent evidence.
    assert results[0] == results[1]


def test_shipped_receipts_match_review_ballpark():
    """Live receipts: Ultra ~tie with wide CI; Nano large win; no abs paths."""
    run = sdc.RUN
    if not (run / "final.jsonl").is_file():
        pytest.skip("nvidia-v1 receipts not present")
    doc = sdc.build(run_dir=run, disguised=sdc.DISGUISED, n_boot=sdc.N_BOOT, seed=sdc.SEED)
    ultra = doc["comparisons"]["ours_vs_ultra-untuned"]
    nano = doc["comparisons"]["ours_vs_nano-untuned"]
    assert ultra["n_clusters"] == 300 and ultra["n_rows"] == 2222
    assert abs(100 * ultra["difference"]) < 0.15
    ulo, uhi = ultra["ci95"]
    assert 100 * ulo == pytest.approx(-2.9, abs=0.15)
    assert 100 * uhi == pytest.approx(2.8, abs=0.15)
    assert 100 * nano["difference"] == pytest.approx(18.7, abs=0.15)
    nlo, nhi = nano["ci95"]
    assert nlo < nano["difference"] < nhi
    assert 100 * nlo == pytest.approx(14.1, abs=0.3)
    assert 100 * nhi == pytest.approx(23.4, abs=0.3)
    assert not str(doc["inputs"]["ours"]).startswith("/")
