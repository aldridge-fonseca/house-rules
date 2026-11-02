import json

import pytest

from safety_one.budget import BudgetExceeded, BudgetTracker
from scripts import watchdog


def test_reservation_is_persisted_and_settled(tmp_path):
    path = tmp_path / "spend.json"
    tracker = BudgetTracker(path, global_cap=2, provider_caps={"openai": 1})
    token = tracker.reserve("openai", 0.4, metadata={"purpose": "test"})
    assert json.loads(path.read_text())["reservations"][token]["amount_usd"] == 0.4
    tracker.settle(token, 0.2)
    totals = BudgetTracker(path, global_cap=2, provider_caps={"openai": 1}).totals()
    assert totals["global"] == pytest.approx(0.2)
    assert totals["providers"]["openai"] == pytest.approx(0.2)
    assert not totals["reservations"]


def test_provider_and_global_caps_block_before_call(tmp_path):
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=0.5,
                            provider_caps={"openai": 0.4})
    tracker.reserve("openai", 0.35)
    with pytest.raises(BudgetExceeded, match="openai"):
        tracker.reserve("openai", 0.1)
    tracker.cancel(next(iter(tracker.totals()["reservations"])))
    tracker.reserve("anthropic", 0.35)
    with pytest.raises(BudgetExceeded, match="global"):
        tracker.reserve("gemini", 0.2)


def test_frontier_cap_is_shared_between_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("SAFETY_ONE_BUDGET_FRONTIER_USD", "0.5")
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=5)
    tracker.reserve("openai", 0.35)
    with pytest.raises(BudgetExceeded, match="frontier group"):
        tracker.reserve("gemini", 0.2)


class Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self): return self
    def __exit__(self, *_): pass
    def read(self): return self.body


def test_watchdog_paginates_with_mock_http():
    calls = []

    def opener(request, timeout=0):
        calls.append(request.full_url)
        if "pageToken" in request.full_url:
            return Response({"deployments": [{"id": "d2"}]})
        return Response({"deployments": [{"id": "d1"}], "nextPageToken": "next"})

    found = watchdog.list_deployments("acct", "not-a-real-key", opener=opener)
    assert [row["id"] for row in found] == ["d1", "d2"]
    assert len(calls) == 2


def test_watchdog_cleanup_selection_uses_age_or_budget():
    rows = [{"name": "accounts/a/deployments/old", "createTime": "2026-01-01T00:00:00Z"},
            {"name": "accounts/a/deployments/new", "createTime": "2026-11-02T12:30:00Z"}]
    # now = 2026-11-02T13:00:00Z; "new" is 30 min old, below the 60 min cutoff.
    selected = watchdog.select_for_cleanup(rows, now=1793624400, older_than_minutes=60,
                                           over_budget=False)
    assert [row["name"] for row, _ in selected] == [rows[0]["name"]]
    # Boundary: "at least N minutes old" is inclusive (exactly 60 min qualifies).
    edge = [{"name": "accounts/a/deployments/edge", "createTime": "2026-11-02T12:00:00Z"}]
    assert len(watchdog.select_for_cleanup(edge, now=1793624400, older_than_minutes=60,
                                           over_budget=False)) == 1
    assert len(watchdog.select_for_cleanup(rows, now=1793624400,
                                            older_than_minutes=60, over_budget=True)) == 2
