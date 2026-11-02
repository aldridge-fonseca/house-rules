import json

from baselines import frontier_summary as fs


def _rec(i, gold, label=None, error=None, cost=0.01, lat=100.0, pert="none"):
    pred = {"label": label or "", "categories": [], "model": "m", "latency_ms": lat if label else 0.0,
            "cost_usd": cost if label else 0.0,
            "raw": {"prompt_tokens": 10, "completion_tokens": 2,
                    "response": {"usage": {"completion_tokens_details": {"reasoning_tokens": 1}}}}
            if label else {}}
    return {"id": i, "input_hash": "h" + i, "gold_label": gold, "source": "s", "perturbation": pert,
            "prediction": pred, "error": error}


def test_summary_two_views_and_ledger_cost(tmp_path):
    run = tmp_path / "frozen-v1"
    run.mkdir()
    (run / "frontier-router.plan.json").write_text(json.dumps({"total_rows": 4, "files": [
        {"path": "data/public/xstest.jsonl", "ids": ["x1", "x2"]},
        {"path": "data/frozen/disguised_v1.jsonl", "ids": ["d1", "d2"]}]}))
    rows = [_rec("x1", "safe", "safe"), _rec("x2", "safe", error="FrontierContentFilter: m refused"),
            _rec("d1", "unsafe", "unsafe", pert="leetspeak"),
            _rec("d2", "unsafe", error="FrontierContentFilter: m refused", pert="leetspeak")]
    stale = _rec("d1", "unsafe", error="HTTPError: 429")
    (run / "frontier-router-a.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [stale] + rows) + "\n{partial")
    ledger = {"entries": [{"provider": "frontier-router", "cost_usd": 0.05,
                           "metadata": {"model": "m", "row_id": "d2"}}],
              "reservations": {"t": {"provider": "frontier-router", "amount_usd": 0.2,
                                     "metadata": {"model": "m"}}}}
    (tmp_path / "spend.json").write_text(json.dumps(ledger))
    doc = fs.build(run, tmp_path / "spend.json")
    g = doc["arms"]["frontier-router-a"]["groups"]
    a = g["ALL"]
    assert doc["label"] == "LIVE" and a["n"] == 4 and a["refusals"] == 2 and a["other_errors"] == 0
    assert a["refusals_excluded"]["n_scored"] == 2 and a["refusals_excluded"]["accuracy"] == 100.0
    assert a["refusals_excluded"]["xstest_over_block_rate"] == 0.0
    assert a["refusals_as_unsafe"]["xstest_over_block_rate"] == 50.0
    assert a["refusals_as_unsafe"]["unsafe_recall"] == 100.0
    assert a["usd_total"] == 0.07 and a["usd_per_1m_messages"] == 17500.0
    assert (a["input_tokens"], a["output_tokens"], a["reasoning_tokens"]) == (20, 4, 2)
    assert g["disguised_v1/leetspeak"]["n"] == 2
    assert doc["ledger_open_reservations"]["m"]["count"] == 1
    assert "LIVE" in fs.render(doc)
