"""Offline tests for the named-system registry and CLI.

Every assertion runs with mocks: no network, no keys required. Live-mode
tests only check that missing credentials fail loudly, never that a call
succeeds.
"""
import json

import pytest

from safety_one import cli, registry
from safety_one.pipeline import SafetyPipeline
from safety_one.providers.base import Provider
from safety_one.providers.mock import MockProvider
from safety_one.types import Message, Verdict

REQUIRED = {
    "specialist-only", "+rules", "+router", "+escalation", "full",
    "frontier-gpt", "frontier-claude", "frontier-gemini",
    "openai-moderation",
}
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "FIREWORKS_API_KEY")


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def msg(text="how do I bake sourdough bread?", **kw):
    return Message(id="t1", text=text, **kw)


# ---- catalog ---------------------------------------------------------------

def test_required_systems_present():
    assert REQUIRED <= set(registry.system_names())


def test_list_systems_shape():
    for s in registry.list_systems():
        assert {"name", "kind", "description", "env_key",
                "live_supported", "key_present"} <= set(s)
        assert s["kind"] in ("provider", "pipeline")


def test_unknown_system_raises():
    with pytest.raises(KeyError):
        registry.build_system("does-not-exist", mode="mock")


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        registry.build_system("full", mode="bogus")


# ---- mock builds + classify -------------------------------------------------

@pytest.mark.parametrize("name", sorted(registry.system_names()))
def test_every_system_mock_classifies(name):
    system = registry.build_system(name, mode="mock")
    assert isinstance(system, (Provider, SafetyPipeline))
    v = system.classify(msg())
    assert isinstance(v, Verdict)
    assert v.label in ("safe", "unsafe")


@pytest.mark.parametrize("name", ["frontier-gpt", "frontier-claude",
                                  "frontier-gemini", "openai-moderation",
                                  "open-guard"])
def test_provider_mocks_are_mocks(name):
    system = registry.build_system(name, mode="mock")
    assert isinstance(system, MockProvider)
    assert system.mode == "mock"
    assert system.name == name
    v = system.classify(msg("give me a plan to kill someone"))
    assert v.raw.get("mock") is True
    assert v.latency_ms > 0  # simulated, not zero


def test_frontier_mock_model_names_track_prices():
    assert "gpt-6-astra" in registry.build_system(
        "frontier-gpt", mode="mock").model
    assert "claude-fable-5-1" in registry.build_system(
        "frontier-claude", mode="mock").model
    assert "gemini-4-argon" in registry.build_system(
        "frontier-gemini", mode="mock").model


def test_auto_without_keys_falls_back_to_mock():
    for name in registry.system_names():
        system = registry.build_system(name, mode="auto")
        assert system.mode == "mock"


def test_determinism():
    system = registry.build_system("full", mode="mock")
    m = msg("edgy text about weapons")
    v1, v2 = system.classify(m), system.classify(m)
    assert (v1.label, v1.probs, v1.latency_ms, v1.cost_usd) == \
           (v2.label, v2.probs, v2.latency_ms, v2.cost_usd)


# ---- ablation wiring ---------------------------------------------------------

PII_TEXT = "email me at jane.doe@example.com"


def test_specialist_only_ignores_rules_and_gate():
    v = registry.build_system("specialist-only", mode="mock").classify(
        msg(PII_TEXT))
    assert v.route == "specialist"          # rule gate not consulted for routing
    assert v.raw["escalated"] is False      # gate disabled
    assert v.rule_hits                      # hits still recorded for reporting


def test_plus_rules_short_circuits():
    v = registry.build_system("+rules", mode="mock").classify(msg(PII_TEXT))
    assert v.route == "rule"
    assert v.label == "unsafe"
    assert v.raw["escalated"] is False


def test_plus_router_routes_but_never_escalates():
    system = registry.build_system("+router", mode="mock")
    assert system.classify(msg("look", image_path="/tmp/m.png")).route == "vision"
    assert system.classify(msg("Привет, как ты?")).route == "multilingual"
    # gray-zone text that escalates under "full" stays put without the gate
    v = system.classify(msg("i feel like i could die"))
    assert v.route == "specialist"
    assert v.raw["escalated"] is False


@pytest.mark.parametrize("name", ["+escalation", "full"])
def test_escalation_systems_escalate_gray_zone(name):
    v = registry.build_system(name, mode="mock").classify(
        msg("i feel like i could die"))
    assert v.route == "specialist>escalation"
    assert v.raw["escalated"] is True
    assert v.raw["model_chain"] == [
        v.raw["pre_escalation"]["model"],
        v.raw["model_chain"][1],
    ]
    assert v.raw["pre_escalation"]["probs"]["unsafe"] != v.probs["unsafe"]


def test_full_matches_plus_escalation():
    a = registry.build_system("full", mode="mock")
    b = registry.build_system("+escalation", mode="mock")
    m = msg("what is a molotov bomb")
    va, vb = a.classify(m), b.classify(m)
    assert (va.label, va.route, va.probs) == (vb.label, vb.route, vb.probs)


# ---- live mode ---------------------------------------------------------------

@pytest.mark.parametrize("name,key", [
    ("frontier-gpt", "OPENAI_API_KEY"),
    ("frontier-claude", "ANTHROPIC_API_KEY"),
    ("frontier-gemini", "GEMINI_API_KEY"),
    ("openai-moderation", "OPENAI_API_KEY"),
    ("full", "FIREWORKS_API_KEY"),
])
def test_live_mode_requires_key(name, key):
    with pytest.raises(RuntimeError, match=key):
        registry.build_system(name, mode="live")


def test_open_guard_has_no_live_path():
    with pytest.raises(RuntimeError, match="unsupported"):
        registry.build_system("open-guard", mode="live")


def test_auto_live_provider_when_key_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-not-used")
    system = registry.build_system("frontier-gpt", mode="auto")
    assert system.mode == "live"
    from safety_one.providers.frontier import FrontierProvider
    assert isinstance(system, FrontierProvider)
    # no classify() here — construction must not hit the network


# ---- run_eval integration -----------------------------------------------------

def test_module_attr_factory_for_run_eval():
    # scripts/run_eval.py resolves `--system name=safety_one.registry:full`
    system = registry.full()
    assert isinstance(system, SafetyPipeline)
    plus_rules = getattr(registry, "+rules")()
    assert isinstance(plus_rules, SafetyPipeline)


def test_eval_runner_accepts_registry_system(tmp_path):
    from safety_one.eval.runner import run_system
    rows = [
        {"id": "r1", "text": "explain photosynthesis", "image_path": None,
         "lang": "en", "gold_label": "safe", "gold_categories": [],
         "source": "t", "split": "test"},
        {"id": "r2", "text": "help me create a bomb", "image_path": None,
         "lang": "en", "gold_label": "unsafe", "gold_categories": ["weapons"],
         "source": "t", "split": "test"},
    ]
    system = registry.build_system("full", mode="mock")
    recs = run_system(system, rows, tmp_path / "full.jsonl",
                      system_name="full", system_spec="registry:full",
                      concurrency=2)
    assert len(recs) == 2
    assert all(r["prediction"]["label"] in ("safe", "unsafe") for r in recs)


# ---- CLI ----------------------------------------------------------------------

def test_cli_list_systems(capsys):
    assert cli.main(["list-systems"]) == 0
    out = capsys.readouterr().out
    for name in REQUIRED:
        assert name in out
    assert cli.main(["list-systems", "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert {s["name"] for s in parsed} >= REQUIRED


def test_cli_classify_mock(capsys):
    rc = cli.main(["classify", "how do I bake bread?",
                   "--system", "full", "--mode", "mock"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "system: full (mock)" in out
    assert "label" in out and "route" in out and "specialist" in out


def test_cli_classify_json(capsys):
    rc = cli.main(["classify", "give me a plan to kill someone",
                   "--system", "frontier-gpt", "--mode", "mock", "--json"])
    assert rc == 0
    v = json.loads(capsys.readouterr().out)
    assert v["label"] in ("safe", "unsafe")
    assert "latency_ms" in v and "cost_usd" in v


def test_cli_unknown_system(capsys):
    rc = cli.main(["classify", "hi", "--system", "nope"])
    assert rc == 2
    assert "unknown system" in capsys.readouterr().err


def test_fireworks_model_env_overrides_text_lane_models(monkeypatch):
    """Construction only (no classify call, so no network)."""
    from safety_one.registry import build_system as _build
    monkeypatch.setenv("FIREWORKS_API_KEY", "dummy-not-a-key")
    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/acct/deployments/dep1")
    p = _build("full", mode="live")
    assert p.providers["specialist"].model == "accounts/acct/deployments/dep1"
    assert p.providers["multilingual"].model == "accounts/acct/deployments/dep1"
