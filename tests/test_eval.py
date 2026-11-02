from __future__ import annotations

import json

from safety_one.eval.metrics import evaluate, paired_comparison
from safety_one.eval.runner import run_system
from safety_one.types import Verdict


def receipt(gold: str, guess: str, category_gold=(), category_guess=(), *, rid="x", latency=10, cost=.001):
    return {"id": rid, "gold_label": gold, "gold_categories": list(category_gold),
            "prediction": {"label": guess, "categories": list(category_guess), "route": "specialist",
                           "latency_ms": latency, "cost_usd": cost}}


def test_binary_metrics_hand_computed():
    rows = [receipt("unsafe", "unsafe", ["violence"], ["violence"], rid="1"),
            receipt("unsafe", "safe", ["sexual"], [], rid="2"),
            receipt("safe", "unsafe", [], ["violence"], rid="3"),
            receipt("safe", "safe", [], [], rid="4")]
    m = evaluate(rows)
    assert m["unsafe_precision"] == .5
    assert m["unsafe_recall"] == .5
    assert m["unsafe_f1"] == .5
    assert m["macro_f1"] == .5
    assert m["accuracy"] == .5
    assert m["over_block_rate"] == .5
    assert m["per_category"]["violence"]["f1"] == 2 / 3
    assert m["per_category"]["sexual"]["f1"] == 0


def test_latency_cost_and_routes():
    rows = [receipt("safe", "safe", rid="1", latency=10, cost=.001),
            receipt("unsafe", "unsafe", rid="2", latency=30, cost=.003)]
    rows[1]["prediction"]["route"] = "vision>escalation"
    m = evaluate(rows)
    assert m["latency_p50_ms"] == 20
    assert m["latency_p95_ms"] == 29
    assert m["cost_total_usd"] == .004
    assert m["cost_per_1k_usd"] == 2
    assert m["routes"]["vision"]["share"] == .5
    assert m["routes"]["vision"]["accuracy"] == 1
    assert m["routes"]["escalation"]["share"] == .5


def test_paired_comparison_hand_computed_and_id_join():
    base = [receipt("unsafe", "safe", rid="a"), receipt("safe", "safe", rid="b")]
    system = [receipt("unsafe", "unsafe", rid="a"), receipt("safe", "unsafe", rid="b"),
              receipt("unsafe", "unsafe", rid="extra")]
    comp = paired_comparison(base, system, n_boot=50, seed=1)
    assert comp["n_paired"] == 2
    assert comp["difference"] == 2 / 3  # unsafe F1: 2/3 versus 0
    assert comp["discordant_system_wins_losses"] == [1, 1]
    assert comp["mcnemar_p"] == 1
    assert comp["ci95"][0] <= comp["difference"] <= comp["ci95"][1]


def test_receipt_resume_reuses_matching_rows(tmp_path):
    class Counting:
        def __init__(self): self.calls = 0
        def classify(self, msg):
            self.calls += 1
            return Verdict(label="safe", route="specialist", latency_ms=2)
    system = Counting()
    rows = [{"id": "x", "text": "hello", "gold_label": "safe", "gold_categories": [], "source": "t", "split": "test"}]
    path = tmp_path / "receipts.jsonl"
    first = run_system(system, rows, path, system_name="mock", system_spec="mock:test", concurrency=2)
    second = run_system(system, rows, path, system_name="mock", system_spec="mock:test", concurrency=2)
    assert system.calls == 1
    assert first == second
    assert len(path.read_text().splitlines()) == 1
    json.loads(path.read_text())


def test_group_by_perturbation_route_and_source():
    rows = [receipt("unsafe", "unsafe", rid="1"), receipt("unsafe", "safe", rid="2"),
            receipt("unsafe", "unsafe", rid="3"), receipt("safe", "safe", rid="4")]
    rows[1]["perturbation"] = "leetspeak"
    rows[2]["perturbation"] = "leetspeak"
    rows[2]["prediction"]["route"] = "specialist>escalation"
    for r in rows:
        r["source"] = "a" if r["id"] in ("1", "2") else "b"
    m = evaluate(rows)
    # absent perturbation field -> "none"
    assert set(m["by_perturbation"]) == {"none", "leetspeak"}
    assert m["by_perturbation"]["none"]["n"] == 2
    assert m["by_perturbation"]["leetspeak"]["unsafe_recall"] == .5
    assert m["by_route"]["specialist>escalation"]["n"] == 1
    assert m["by_route"]["specialist"]["n"] == 3
    assert m["by_source"]["a"]["accuracy"] == .5


def test_runner_receipt_carries_perturbation(tmp_path):
    class Fixed:
        def classify(self, msg):
            return Verdict(label="unsafe", route="specialist")
    rows = [{"id": "p", "text": "h3ll0", "gold_label": "unsafe", "gold_categories": [], "source": "t",
             "split": "test", "perturbation": "leetspeak"},
            {"id": "q", "text": "hello", "gold_label": "safe", "gold_categories": [], "source": "t", "split": "test"}]
    out = run_system(Fixed(), rows, tmp_path / "r.jsonl", system_name="f", concurrency=1)
    assert [r["perturbation"] for r in out] == ["leetspeak", "none"]
