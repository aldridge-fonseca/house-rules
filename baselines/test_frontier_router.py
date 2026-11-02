import email.message
import json
import random
import urllib.error

import pathlib

import pytest

from safety_one.budget import FRONTIER_HARD_CAP_USD, BudgetExceeded, BudgetTracker
from safety_one.providers.frontier import RouterFrontierProvider, moderation_prompt
from safety_one.types import Message

_WILDGUARD_MISSING = not (pathlib.Path(__file__).resolve().parents[1] / "data" / "public" / "wildguard_test.jsonl").exists()


class Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def chat_body(text='{"label":"unsafe","categories":["violence"]}', it=100, ot=8):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": it, "completion_tokens": ot}}


def http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://router.test/v1/chat/completions", code,
                                  "err", headers, None)


PRICE = {"input_per_million": 10.0, "output_per_million": 50.0, "source": "test"}


def make_provider(monkeypatch, tmp_path, opener, **kwargs):
    monkeypatch.setenv("ROUTER_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("ROUTER_API_KEY", "test-key-never-printed")
    kwargs.setdefault("budget", BudgetTracker(tmp_path / "spend.json", global_cap=50))
    kwargs.setdefault("opener", opener)
    kwargs.setdefault("sleep", lambda s: None)
    return RouterFrontierProvider(model="test-model", execute=True, **kwargs)


def test_payload_is_openai_chat_completions(monkeypatch, tmp_path):
    seen = {}

    def opener(req, timeout):
        seen["url"], seen["req"] = req.full_url, req
        return Response(chat_body())

    v = make_provider(monkeypatch, tmp_path, opener, price=PRICE).classify(
        Message(id="r1", text="test content"))
    assert seen["url"] == "https://router.test/v1/chat/completions"
    sent = json.loads(seen["req"].data)
    assert sent["model"] == "test-model"
    assert "temperature" not in sent and sent["max_completion_tokens"] > 0
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"][0] == {"role": "system", "content": moderation_prompt()}
    assert sent["messages"][1] == {"role": "user", "content": "test content"}
    assert seen["req"].get_header("Authorization") == "Bearer test-key-never-printed"
    assert "test-key-never-printed" not in seen["req"].data.decode()
    assert "test-key-never-printed" not in seen["url"]
    assert v.label == "unsafe" and v.categories == ["violence"]
    assert v.route == "frontier-router"


def test_real_cost_from_usage_and_ledger_settle(monkeypatch, tmp_path):
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=50)
    v = make_provider(monkeypatch, tmp_path,
                      lambda req, timeout: Response(chat_body(it=100, ot=8)),
                      budget=tracker, price=PRICE).classify(Message(id="r", text="x"))
    # (100*10 + 8*50)/1e6
    assert v.cost_usd == pytest.approx(0.0014)
    assert v.raw["cost_basis"] == "measured"
    totals = tracker.totals()
    assert totals["providers"]["frontier-router"] == pytest.approx(0.0014)
    assert not totals["reservations"]


def test_unknown_price_never_guesses_cost(monkeypatch, tmp_path):
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=50)
    # No matching model in prices.yaml and no injected price.
    v = make_provider(monkeypatch, tmp_path,
                      lambda req, timeout: Response(chat_body()),
                      budget=tracker, reserve_usd=0.25).classify(Message(id="r", text="x"))
    assert v.cost_usd == 0.0
    assert v.raw["cost_basis"] == "unknown"
    assert v.raw["prompt_tokens"] == 100  # usage still recorded for later pricing
    # The ledger holds the conservative reservation, flagged as an estimate.
    totals = tracker.totals()
    assert totals["providers"]["frontier-router"] == pytest.approx(0.25)


def test_retries_transient_errors_then_succeeds(monkeypatch, tmp_path):
    calls = []

    def opener(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("connection reset")
        return Response(chat_body())

    v = make_provider(monkeypatch, tmp_path, opener, price=PRICE).classify(
        Message(id="r", text="x"))
    assert len(calls) == 3
    assert v.label == "unsafe"


def test_retries_429_honoring_retry_after(monkeypatch, tmp_path):
    calls, sleeps = [], []

    def opener(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise http_error(429, retry_after=7)
        return Response(chat_body())

    make_provider(monkeypatch, tmp_path, opener, price=PRICE,
                  sleep=sleeps.append).classify(Message(id="r", text="x"))
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_no_retry_on_client_error(monkeypatch, tmp_path):
    calls = []

    def opener(req, timeout):
        calls.append(1)
        raise http_error(400)

    with pytest.raises(urllib.error.HTTPError):
        make_provider(monkeypatch, tmp_path, opener, price=PRICE).classify(
            Message(id="r", text="x"))
    assert len(calls) == 1
    # Failed call cancels the reservation rather than charging the estimate.
    totals = BudgetTracker(tmp_path / "spend.json", global_cap=50).totals()
    assert not totals["reservations"]
    assert totals["providers"].get("frontier-router", 0.0) == 0.0


def test_http_400_policy_block_is_final_refusal(monkeypatch, tmp_path):
    import io
    from safety_one.eval.runner import _retryable
    from safety_one.providers.frontier import FrontierContentFilter
    calls = []
    body = {"error": {"message": "This content was flagged for possible cybersecurity risk.",
                      "type": "invalid_request_error", "code": "cyber_policy"}}

    def opener(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("https://router.test/v1/chat/completions", 400, "Bad Request",
                                     email.message.Message(), io.BytesIO(json.dumps(body).encode()))

    with pytest.raises(FrontierContentFilter, match="cyber_policy") as info:
        make_provider(monkeypatch, tmp_path, opener, price=PRICE).classify(
            Message(id="r", text="x"))
    assert len(calls) == 1
    assert not _retryable({"error": f"{type(info.value).__name__}: {info.value}"})


def test_gives_up_after_retry_budget(monkeypatch, tmp_path):
    calls = []

    def opener(req, timeout):
        calls.append(1)
        raise urllib.error.URLError("down")

    with pytest.raises(urllib.error.URLError):
        make_provider(monkeypatch, tmp_path, opener, retries=2, price=PRICE).classify(
            Message(id="r", text="x"))
    assert len(calls) == 3


def test_budget_blocks_before_http_call(monkeypatch, tmp_path):
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=50,
                            provider_caps={"frontier-router": 0.001})
    calls = []

    def opener(req, timeout):
        calls.append(1)
        return Response(chat_body())

    with pytest.raises(BudgetExceeded):
        make_provider(monkeypatch, tmp_path, opener, budget=tracker,
                      reserve_usd=0.25).classify(Message(id="r", text="x"))
    assert not calls


def test_spend_accumulates_then_run_halts(monkeypatch, tmp_path):
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=50,
                            provider_caps={"frontier-router": 0.05})
    calls = []

    def opener(req, timeout):
        calls.append(1)
        return Response(chat_body())

    provider = make_provider(monkeypatch, tmp_path, opener,
                             budget=tracker, reserve_usd=0.03)
    provider.classify(Message(id="a", text="x"))   # reserves+settles $0.03
    with pytest.raises(BudgetExceeded):
        provider.classify(Message(id="b", text="x"))  # 0.03+0.03 > cap: blocked pre-call
    assert len(calls) == 1
    assert tracker.totals()["providers"]["frontier-router"] == pytest.approx(0.03)


def test_settle_over_cap_persists_then_blocks(tmp_path):
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=50,
                            provider_caps={"frontier-router": 0.05})
    token = tracker.reserve("frontier-router", 0.03)
    with pytest.raises(BudgetExceeded):
        tracker.settle(token, 0.06)   # actual proved bigger than the estimate
    assert tracker.totals()["providers"]["frontier-router"] == pytest.approx(0.06)
    with pytest.raises(BudgetExceeded):
        tracker.reserve("frontier-router", 0.01)


def test_decode_failure_still_records_spend(monkeypatch, tmp_path):
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=50)
    provider = make_provider(monkeypatch, tmp_path,
                             lambda req, timeout: Response(chat_body(text="not json")),
                             budget=tracker, price=PRICE)
    with pytest.raises(ValueError):
        provider.classify(Message(id="r", text="x"))
    assert tracker.totals()["providers"]["frontier-router"] == pytest.approx(0.0014)


def test_dry_run_never_calls_network():
    def forbidden(*a, **k):
        raise AssertionError("dry-run attempted HTTP")

    v = RouterFrontierProvider(model="m", opener=forbidden).classify(
        Message(id="r", text="hello"))
    assert v.raw["simulated"] is True
    assert v.route == "frontier-router-dry-run"
    assert v.cost_usd == 0


def test_missing_router_env_fails_fast(monkeypatch):
    monkeypatch.delenv("ROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="ROUTER_BASE_URL"):
        RouterFrontierProvider(execute=True)
    monkeypatch.setenv("ROUTER_BASE_URL", "https://router.test/v1")
    with pytest.raises(RuntimeError, match="FRONTIER_MODEL"):
        RouterFrontierProvider(execute=True)


def test_image_row_sends_image_url_part(monkeypatch, tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    seen = {}

    def opener(req, timeout):
        seen["body"] = json.loads(req.data)
        return Response(chat_body())

    make_provider(monkeypatch, tmp_path, opener, price=PRICE).classify(
        Message(id="img", text="caption", image_path=str(image)))
    parts = seen["body"]["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "caption"}
    assert parts[1]["type"] == "image_url"
    assert "iVBORw0KGgo=" in parts[1]["image_url"]["url"]


def test_arm_envs_select_models(monkeypatch, tmp_path):
    monkeypatch.setenv("ROUTER_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("FRONTIER_MODEL_A", "model-a-name")
    monkeypatch.setenv("FRONTIER_MODEL_B", "model-b-name")
    a = RouterFrontierProvider(model_env="FRONTIER_MODEL_A", execute=True, budget=False)
    b = RouterFrontierProvider(model_env="FRONTIER_MODEL_B", execute=True, budget=False)
    assert (a.model, b.model) == ("model-a-name", "model-b-name")


# ---- budget group rules -------------------------------------------------

def test_router_arms_share_frontier_group_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("SAFETY_ONE_BUDGET_FRONTIER_USD", "0.5")
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=50)
    tracker.reserve("frontier-router", 0.35)
    with pytest.raises(BudgetExceeded, match="frontier group"):
        tracker.reserve("openai", 0.2)


def test_frontier_cap_clamped_to_hard_max(tmp_path, monkeypatch):
    monkeypatch.setenv("SAFETY_ONE_BUDGET_FRONTIER_USD", "500")
    tracker = BudgetTracker(tmp_path / "spend.json", global_cap=1000)
    assert tracker.cap_for("frontier-router-a") == FRONTIER_HARD_CAP_USD
    assert tracker.cap_for("openai") == FRONTIER_HARD_CAP_USD
    tracker.reserve("frontier-router", 140.0)
    with pytest.raises(BudgetExceeded, match="cap \\$150"):
        tracker.reserve("frontier-router", 11.0)


# ---- sampling / registry -----------------------------------------------

@pytest.mark.skipif(_WILDGUARD_MISSING, reason="WildGuard is gated; fetch it with the prep script to run this test")
def test_public_sample_matches_run_eval_limit_per_file():
    """The plan's 300-row sample must be the identical rows run_eval.py
    --limit-per-file 300 --seed 0 selects (hence a subset of the full files)."""
    from baselines.run_frontier_router import PUBLIC, ROOT, sampled_subset
    from scripts.run_eval import load_jsonl
    for rel in PUBLIC:
        path = ROOT / rel
        full = load_jsonl(path)
        expected = random.Random(f"0:{path.name}").sample(full, 300)
        assert [r["id"] for r in sampled_subset(path, 300, 0)] == \
               [r["id"] for r in expected]
        assert {r["id"] for r in expected} <= {r["id"] for r in full}


@pytest.mark.skipif(_WILDGUARD_MISSING, reason="WildGuard is gated; fetch it with the prep script to run this test")
def test_plan_passes_subset_verification():
    from baselines.run_frontier_router import ROOT, build_plan, verify_subset
    plan = build_plan(ROOT, seed=0, public_limit=300)
    assert verify_subset(plan, ROOT) == []
    counts = {item["path"]: len(item["rows"]) for item in plan}
    assert counts["data/frozen/disguised_v1.jsonl"] == 2222
    for rel in ("data/public/openai_moderation.jsonl",
                "data/public/wildguard_test.jsonl", "data/public/xstest.jsonl"):
        assert counts[rel] == 300


def test_registry_builds_router_arms_mock_and_live_gate(monkeypatch, tmp_path):
    from safety_one import registry
    for name in ("frontier-router", "frontier-router-a", "frontier-router-b"):
        assert name in registry.system_names()
        system = registry.build_system(name, mode="mock")
        assert system.mode == "mock" and system.system_name == name
    # live mode with no router credentials must fail loudly, not mock —
    # isolate from any real credentials in .env/environment first.
    monkeypatch.setattr(registry, "ENV_PATH", str(tmp_path / "absent.env"))
    for var in ("ROUTER_API_KEY", "ROUTER_BASE_URL", "FRONTIER_MODEL_A"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError):
        registry.build_system("frontier-router-a", mode="live")
