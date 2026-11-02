"""Mocked-HTTP tests for safety_one.providers.fireworks.

No network: every provider gets an injected ``opener`` returning canned
responses. The API key is a dummy env value; payloads are inspected, never sent.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from safety_one.providers.fireworks import (
    FireworksProvider, _backoff, _parse_label, _retry_after, _system_entry)
from safety_one.types import Message

KEY_ENV = "FW_TEST_KEY"
MODEL = "accounts/fireworks/models/test-model"


def _response(payload: dict, status: int = 200, headers=None) -> io.BytesIO:
    body = io.BytesIO(json.dumps(payload).encode("utf-8"))
    body.status = status  # type: ignore[attr-defined]
    body.headers = headers or {}  # type: ignore[attr-defined]
    return body


def _chat(label: str = "safe", *, usage=None, logprobs=None, content=None):
    choice = {"message": {"content": content if content is not None else json.dumps({"label": label})}}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"choices": [choice],
            "usage": usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 10}}


class FakeOpener:
    """Records requests and plays back canned responses/exceptions in order."""

    def __init__(self, *items):
        self.items = list(items)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if not self.items:
            raise AssertionError("unexpected extra request")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def payloads(self) -> list[dict]:
        return [json.loads(r.data.decode("utf-8")) for r in self.requests]


def _provider(opener, **kwargs) -> FireworksProvider:
    return FireworksProvider(MODEL, api_key_env=KEY_ENV, opener=opener,
                             base_url="https://example.test", **kwargs)


def _msg(text="hello") -> Message:
    return Message(id="m1", text=text, image_path=None, lang="en", meta={})


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "dummy-not-a-key")
    monkeypatch.delenv("FIREWORKS_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("FIREWORKS_PRICE_INPUT_PER_MILLION_USD", raising=False)
    monkeypatch.delenv("FIREWORKS_PRICE_OUTPUT_PER_MILLION_USD", raising=False)
    monkeypatch.delenv("FIREWORKS_COST_USD_PER_REQUEST", raising=False)


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------

def test_payload_sends_reasoning_effort_none_and_logprobs():
    opener = FakeOpener(_response(_chat("safe")))
    v = _provider(opener).classify(_msg())
    sent = opener.payloads()[0]
    assert sent["reasoning_effort"] == "none"
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 256
    assert sent["logprobs"] is True and sent["top_logprobs"] == 5
    assert v.label == "safe" and v.model == MODEL and v.route == "specialist"


def test_reasoning_effort_none_falls_back_and_logprobs_off():
    opener = FakeOpener(_response(_chat("unsafe")))
    v = _provider(opener, reasoning_effort=None,
                  logprobs=False).classify(_msg())
    # reasoning_effort=None means "fall back to config/default" — here the
    # unknown model has no systems.yaml entry, so the default "none" applies.
    assert opener.payloads()[0]["reasoning_effort"] == "none"
    assert "logprobs" not in opener.payloads()[0]
    assert v.label == "unsafe"


def test_reasoning_effort_explicit_value_and_disabled():
    for effort in ("low", "high"):
        opener = FakeOpener(_response(_chat("safe")))
        _provider(opener, reasoning_effort=effort).classify(_msg())
        assert opener.payloads()[0]["reasoning_effort"] == effort
    opener = FakeOpener(_response(_chat("safe")))
    _provider(opener, reasoning_effort="").classify(_msg())
    assert "reasoning_effort" not in opener.payloads()[0]


# --------------------------------------------------------------------------
# Cost from usage x price
# --------------------------------------------------------------------------

def test_cost_from_usage_and_explicit_prices():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    opener = FakeOpener(_response(_chat("safe", usage=usage)))
    v = _provider(opener, price_input_per_million=0.60,
                  price_output_per_million=2.40).classify(_msg())
    assert v.cost_usd == pytest.approx(0.60 + 1.20)
    assert v.raw["usage"]["prompt_tokens"] == 1_000_000


def test_cost_from_systems_yaml_price_block():
    """The nvidia-lightning-zeroshot entry carries price: — the provider must
    pick it up by model match with no constructor args."""
    model = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
    usage = {"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000}
    opener = FakeOpener(_response(_chat("safe", usage=usage)))
    v = FireworksProvider(model, api_key_env=KEY_ENV, opener=opener,
                          base_url="https://example.test").classify(_msg())
    assert v.cost_usd == pytest.approx(2 * 0.05 + 1 * 0.20)


def test_cost_env_override(monkeypatch):
    monkeypatch.setenv("FIREWORKS_PRICE_INPUT_PER_MILLION_USD", "1.0")
    monkeypatch.setenv("FIREWORKS_PRICE_OUTPUT_PER_MILLION_USD", "2.0")
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    opener = FakeOpener(_response(_chat("safe", usage=usage)))
    v = _provider(opener).classify(_msg())
    assert v.cost_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6)


def test_no_price_falls_back_to_allocated_cost(monkeypatch):
    monkeypatch.setenv("FIREWORKS_COST_USD_PER_REQUEST", "0.001")
    opener = FakeOpener(_response(_chat("safe")))
    v = _provider(opener).classify(_msg())
    assert v.cost_usd == pytest.approx(0.001)


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------

def _http_error(code: int, retry_after: str | None = None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("https://example.test", code, "err", headers, None)


def test_retries_on_429_then_succeeds():
    opener = FakeOpener(_http_error(429), _http_error(500), _response(_chat("unsafe")))
    sleeps = []
    v = _provider(opener, sleep=sleeps.append, backoff_s=0.01).classify(_msg())
    assert v.label == "unsafe" and len(opener.requests) == 3 and len(sleeps) == 2


def test_retry_exhaustion_raises_runtime_error():
    opener = FakeOpener(*[_http_error(503)] * 8)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        _provider(opener, sleep=lambda s: None, max_retries=5).classify(_msg())
    assert len(opener.requests) == 6


def test_non_retryable_400_raises_immediately():
    opener = FakeOpener(_http_error(400), _response(_chat("safe")))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        _provider(opener, sleep=lambda s: None).classify(_msg())
    assert len(opener.requests) == 1


def test_retry_after_header_extends_sleep():
    assert _backoff(1.0, 0, 5.0) >= 5.0
    assert _retry_after({"Retry-After": "2.5"}) == 2.5
    assert _retry_after(None) is None


def test_urlerror_retried():
    opener = FakeOpener(urllib.error.URLError("conn reset"), _response(_chat("safe")))
    v = _provider(opener, sleep=lambda s: None).classify(_msg())
    assert v.label == "safe" and len(opener.requests) == 2


# --------------------------------------------------------------------------
# Label parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("content,expected", [
    ('{"label":"safe"}', "safe"),
    ('{"label": "unsafe"}', "unsafe"),
    ('<think>let me reason</think>{"label":"unsafe"}', "unsafe"),
    ('<think>still thinking...{"label":"safe"}', "safe"),       # unclosed think
    ('```json\n{"label":"unsafe"}\n```', "unsafe"),
    ('The answer is {"label":"safe"} as shown.', "safe"),
    ('{"explanation":"x","label":"unsafe"}', "unsafe"),
    ('"safe"', "safe"),
    ('unsafe', "unsafe"),
])
def test_parse_label_variants(content, expected):
    assert _parse_label(content) == expected


@pytest.mark.parametrize("content", ["", "{}", '{"label":"maybe"}', "no verdict here", "null"])
def test_parse_label_rejects(content):
    with pytest.raises(ValueError):
        _parse_label(content)


def test_classify_parses_think_and_fenced_response():
    content = '<think>reasoning</think>\n```json\n{"label":"unsafe"}\n```'
    opener = FakeOpener(_response(_chat(content=content)))
    assert _provider(opener).classify(_msg()).label == "unsafe"


# --------------------------------------------------------------------------
# Logprob probabilities
# --------------------------------------------------------------------------

def test_label_probabilities_from_top_logprobs():
    logprobs = [{"token": "safe", "logprob": -0.01, "top_logprobs": [
        {"token": "safe", "logprob": -0.01},
        {"token": "unsafe", "logprob": -3.0}]}]
    opener = FakeOpener(_response(_chat("safe", logprobs=logprobs)))
    v = _provider(opener).classify(_msg())
    assert v.raw["probability_source"] == "top_logprobs_label_token"
    assert v.probs["safe"] > 0.9 and v.probs["unsafe"] < 0.1
    assert v.probs["safe"] + v.probs["unsafe"] == pytest.approx(1.0)


def test_label_probabilities_not_inverted_by_lower_ranked_variants():
    # Regression (LIVE 2026-11-02 tuned Nano): later variants "Safe"/"-safe"
    # used to overwrite the real "safe" logprob and report P(unsafe)=1.0.
    logprobs = [{"token": "safe", "logprob": -0.0, "top_logprobs": [
        {"token": "safe", "logprob": -0.0},
        {"token": "unsafe", "logprob": -9.6},
        {"token": "Safe", "logprob": -12.0},
        {"token": "-safe", "logprob": -15.0},
        {"token": " safe", "logprob": -16.0}]}]
    opener = FakeOpener(_response(_chat("safe", logprobs=logprobs)))
    v = _provider(opener).classify(_msg())
    assert v.label == "safe"
    assert v.raw["probability_source"] == "top_logprobs_label_token"
    assert v.probs["unsafe"] < 1e-3 and v.probs["safe"] > 0.999


def test_label_probabilities_ignore_non_label_tokens():
    # "-safe" alone must not stand in for "safe" (exact label tokens only).
    logprobs = [{"token": "unsafe", "logprob": -0.1, "top_logprobs": [
        {"token": "unsafe", "logprob": -0.1},
        {"token": "-safe", "logprob": -2.5}]}]
    opener = FakeOpener(_response(_chat("unsafe", logprobs=logprobs)))
    v = _provider(opener).classify(_msg())
    assert v.raw["probability_source"].startswith("generated_label_one_hot")
    assert v.probs == {"safe": 0.0, "unsafe": 1.0}


def test_label_probabilities_one_hot_fallback():
    opener = FakeOpener(_response(_chat("unsafe", logprobs=[])))
    v = _provider(opener).classify(_msg())
    assert v.raw["probability_source"] == "generated_label_one_hot_fallback_no_two_class_logprobs"
    assert v.probs == {"safe": 0.0, "unsafe": 1.0}


# --------------------------------------------------------------------------
# Config lookup + errors
# --------------------------------------------------------------------------

def test_system_entry_matches_by_model():
    entry = _system_entry("accounts/fireworks/models/nemotron-3-ultra-nvfp4")
    assert entry.get("reasoning_effort") == "none"
    assert entry["price"]["input_per_million_usd"] == 0.60
    assert _system_entry("accounts/x/models/nope") == {}


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=KEY_ENV):
        _provider(FakeOpener()).classify(_msg())


def test_no_choices_raises():
    opener = FakeOpener(_response({"choices": [], "usage": {}}))
    with pytest.raises(ValueError, match="did not contain a choice"):
        _provider(opener).classify(_msg())
