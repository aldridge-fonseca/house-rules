"""vision_live provider tests — HTTP fully mocked, no network."""
from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from safety_one.providers import fireworks as fw
from safety_one.providers import vision_live
from safety_one.types import Message


# ---- canned HTTP layer ----------------------------------------------------

class _Resp:
    def __init__(self, data: dict):
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code: int):
    return urllib.error.HTTPError("https://x", code, "err", hdrs=None, fp=None)


def make_opener(*items):
    """items: response dicts or Exceptions; replays the last item forever."""
    calls = []

    def opener(request, timeout=None):
        calls.append(request)
        item = items[min(len(calls), len(items)) - 1]
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    return calls, opener


def chat_response(label: str, *, prompt_t=11, completion_t=3, content=None):
    return {
        "choices": [{"message": {"content": content or json.dumps({"label": label})},
                     "logprobs": {"content": [{"token": label, "logprob": -0.05,
                                               "top_logprobs": [
                                                   {"token": label, "logprob": -0.05},
                                                   {"token": "safe" if label == "unsafe" else "unsafe",
                                                    "logprob": -3.0}]}]}}],
        "usage": {"prompt_tokens": prompt_t, "completion_tokens": completion_t},
    }


def payloads(calls):
    return [json.loads(c.data) for c in calls]


@pytest.fixture()
def image(tmp_path):
    p = tmp_path / "meme.png"
    p.write_bytes(b"\x89PNG fake")
    return str(p)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("FIREWORKS_VISION_MODEL", "FIREWORKS_MODEL", "FIREWORKS_TEXT_MODEL",
                "FIREWORKS_API_KEY", "FIREWORKS_REASONING_EFFORT",
                "FIREWORKS_VISION_RATE_USD_PER_HOUR", "SAFETY_ONE_NO_DOTENV"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SAFETY_ONE_NO_DOTENV", "1")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")


# ---- MIME / data URI ------------------------------------------------------

def test_image_mime_explicit_map(tmp_path):
    assert vision_live.image_mime(Path("a.png")) == "image/png"
    assert vision_live.image_mime(Path("a.jpg")) == "image/jpeg"
    assert vision_live.image_mime(Path("a.jpeg")) == "image/jpeg"
    assert vision_live.image_mime(Path("a.webp")) == "image/webp"
    with pytest.raises(ValueError):
        vision_live.image_mime(tmp_path / "a.exe")


def test_image_data_uri_roundtrip(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"PNGDATA")
    uri = vision_live.image_data_uri(p)
    assert uri.startswith("data:image/png;base64,")
    import base64
    assert base64.b64decode(uri.split(",", 1)[1]) == b"PNGDATA"
    with pytest.raises(FileNotFoundError):
        vision_live.image_data_uri(tmp_path / "missing.png")


# ---- SuperVLVisionProvider ------------------------------------------------

def _vl(calls, opener, model="accounts/acct/deployments/d1", **kw):
    return vision_live.SuperVLVisionProvider(
        model=model, opener=opener, sleep=lambda s: None, **kw)


def test_supervl_payload_and_verdict(image):
    calls, opener = make_opener(chat_response("unsafe"))
    v = _vl(calls, opener).classify(Message(id="1", text="look", image_path=image))
    body = payloads(calls)[0]
    user = body["messages"][1]["content"]
    assert body["model"] == "accounts/acct/deployments/d1"
    assert body["response_format"]["type"] == "json_schema"
    assert body["reasoning_effort"] == "none"       # LIVE: without it content comes back empty
    assert user[0] == {"type": "text", "text": "look"}
    assert user[1]["type"] == "image_url"
    assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert v.label == "unsafe" and v.route == "vision"
    assert v.model == "accounts/acct/deployments/d1"
    # cost = wall-clock x $52/h upper bound; consistent with latency field
    assert abs(v.cost_usd - v.latency_ms / 1000 * 52 / 3600) < 1e-9
    assert v.raw["pricing_source"] == "https://fireworks.ai/pricing"
    assert v.raw["deployment_gpus"] == 4
    assert v.probs["unsafe"] > v.probs["safe"]      # two-class top-logprobs softmax


def test_supervl_model_from_env(image, monkeypatch):
    monkeypatch.setenv("FIREWORKS_VISION_MODEL", "accounts/acct/deployments/envd")
    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/acct/deployments/TEXT-DEPLOY")
    calls, opener = make_opener(chat_response("safe"))
    v = vision_live.SuperVLVisionProvider(opener=opener, sleep=lambda s: None) \
        .classify(Message(id="1", text="t", image_path=image))
    assert v.model == "accounts/acct/deployments/envd"   # never the text deploy


def test_supervl_requires_model_key_and_image(image, monkeypatch):
    monkeypatch.delenv("FIREWORKS_VISION_MODEL", raising=False)
    p = vision_live.SuperVLVisionProvider(opener=lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="FIREWORKS_VISION_MODEL"):
        p.classify(Message(id="1", text="t", image_path=image))
    p = vision_live.SuperVLVisionProvider(model="m", opener=lambda *a, **k: None)
    monkeypatch.delenv("FIREWORKS_API_KEY")
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        p.classify(Message(id="1", text="t", image_path=image))
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    with pytest.raises(ValueError, match="image_path"):
        p.classify(Message(id="1", text="t"))


def test_supervl_retries_on_429(image):
    calls, opener = make_opener(_http_error(429), _http_error(503), chat_response("safe"))
    v = _vl(calls, opener).classify(Message(id="1", text="t", image_path=image))
    assert v.label == "safe" and len(calls) == 3


def test_supervl_400_drops_response_format(image):
    calls, opener = make_opener(_http_error(400), chat_response("unsafe"))
    v = _vl(calls, opener).classify(Message(id="1", text="t", image_path=image))
    bodies = payloads(calls)
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]
    assert v.label == "unsafe"


def test_supervl_non_retryable_error_surfaces(image):
    calls, opener = make_opener(_http_error(403))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        _vl(calls, opener).classify(Message(id="1", text="t", image_path=image))
    assert len(calls) == 1


def test_supervl_rate_env_override(image, monkeypatch):
    monkeypatch.setenv("FIREWORKS_VISION_RATE_USD_PER_HOUR", "104")
    calls, opener = make_opener(chat_response("safe"))
    v = _vl(calls, opener).classify(Message(id="1", text="t", image_path=image))
    assert v.raw["rate_usd_per_hour"] == 104.0


# ---- NemotronLightningTextProvider ----------------------------------------

def test_lightning_strips_image_and_uses_text(image):
    calls, opener = make_opener(chat_response("safe", prompt_t=100, completion_t=5))
    p = vision_live.NemotronLightningTextProvider(opener=opener, sleep=lambda s: None)
    v = p.classify(Message(id="1", text="just words", image_path=image))
    body = payloads(calls)[0]
    user = body["messages"][1]["content"]
    assert user == "just words"                    # string, no image_url parts
    assert body["reasoning_effort"] == "none"
    assert body["max_tokens"] == 16
    assert body["model"] == vision_live.LIGHTNING_MODEL
    assert v.label == "safe"
    # $0.05/$0.20 per 1M tokens: (100*0.05 + 5*0.20)/1e6
    assert abs(v.cost_usd - (100 * 0.05 + 5 * 0.20) / 1e6) < 1e-12


def test_lightning_lenient_label_parse(image):
    messy = "```json\n{\"label\": \"unsafe\"}\n```"
    calls, opener = make_opener(chat_response("unsafe", content=messy))
    p = vision_live.NemotronLightningTextProvider(opener=opener, sleep=lambda s: None)
    assert p.classify(Message(id="1", text="t", image_path=image)).label == "unsafe"


# ---- registration / systems.yaml ------------------------------------------

def test_install_upgrades_vision_backend():
    vision_live.install()
    provider = __import__("safety_one.registry", fromlist=["x"])._fireworks_provider(
        "fireworks_vision", "accounts/acct/deployments/d1")
    assert isinstance(provider, vision_live.SuperVLVisionProvider)


def test_systems_yaml_vision_role_and_arms():
    from safety_one.registry import load_systems
    cfg = load_systems()
    vision = cfg["pipeline_roles"]["vision"]
    assert vision["backend"] == "fireworks_vision"
    assert vision["model_env"] == "FIREWORKS_VISION_MODEL"
    systems = cfg["systems"]
    assert systems["vision-text-only"]["model"] == vision_live.LIGHTNING_MODEL
    assert systems["vision-routed"]["overrides"]["confidence_gate"]["enabled"] is False


def test_systems_build_offline_mock():
    from safety_one.pipeline import SafetyPipeline
    from safety_one.providers.mock import MockProvider
    from safety_one.registry import build_system
    text = build_system("vision-text-only", mode="mock")
    routed = build_system("vision-routed", mode="mock")
    assert isinstance(text, MockProvider) and text.mode == "mock"
    assert isinstance(routed, SafetyPipeline) and routed.mode == "mock"
    assert isinstance(routed.providers["vision"], MockProvider)


# ---- eval_vision plumbing ---------------------------------------------------

def _load_eval_vision():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_vision", Path(__file__).resolve().parents[1] / "scripts" / "eval_vision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_vision_load_rows_filters_unresolved(tmp_path):
    mod = _load_eval_vision()
    img = tmp_path / "pic.png"
    img.write_bytes(b"png")
    rows = [
        {"id": "a", "image_path": str(img), "gold_label": "unsafe"},   # absolute
        {"id": "b", "image_path": str(tmp_path / "none.png"), "gold_label": "safe"},
        {"id": "c", "image_path": None, "gold_label": "unsafe"},
    ]
    data = tmp_path / "eval.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    kept, skipped = mod.load_rows(data)
    assert [r["id"] for r in kept] == ["a"] and skipped == 2
    assert Path(kept[0]["image_path"]).is_absolute()


def test_eval_vision_wins_losses_counts_unsafe_only():
    mod = _load_eval_vision()
    routed = [
        {"id": "1", "gold_label": "unsafe", "source": "s", "text": "x",
         "prediction": {"label": "unsafe"}},
        {"id": "2", "gold_label": "unsafe", "source": "s", "text": "y",
         "prediction": {"label": "safe"}},
        {"id": "3", "gold_label": "safe", "source": "s", "text": "z",
         "prediction": {"label": "unsafe"}},
    ]
    text = [
        {"id": "1", "gold_label": "unsafe", "prediction": {"label": "safe"}},
        {"id": "2", "gold_label": "unsafe", "prediction": {"label": "unsafe"}},
        {"id": "3", "gold_label": "safe", "prediction": {"label": "safe"}},
    ]
    wins, losses = mod._wins_losses(routed, text)
    assert [w["id"] for w in wins] == ["1"]
    assert [l["id"] for l in losses] == ["2"]


def test_eval_vision_arms_split_then_report_makes_no_calls(tmp_path, monkeypatch):
    mod = _load_eval_vision()
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(mod._registry, "_load_env_file", lambda: None)
    img = tmp_path / "pic.png"
    img.write_bytes(b"png")
    data = tmp_path / "eval.jsonl"
    data.write_text("\n".join(json.dumps({"id": i, "text": "t", "image_path": str(img),
                                          "gold_label": g, "source": "s"})
                              for i, g in (("a", "unsafe"), ("b", "safe"))))
    common = ["--mode", "mock", "--data", str(data), "--out", str(tmp_path), "--run", "r"]
    assert mod.main(common + ["--arms", "routed"]) == 0
    info = json.loads((tmp_path / "r" / "RUN_INFO.json").read_text())
    assert info["modes"]["vision-text-only"] == "missing"
    assert mod.main(common + ["--arms", "text"]) == 0
    monkeypatch.setattr(mod, "run_system", lambda *a, **k: pytest.fail("report must not call"))
    assert mod.main(common + ["--arms", "report"]) == 0
    info = json.loads((tmp_path / "r" / "RUN_INFO.json").read_text())
    assert info["modes"] == {"vision-text-only": "mock", "vision-routed": "mock"}
    assert set(info["wall_seconds"]) == {"vision-text-only", "vision-routed"}


def test_eval_vision_ultra_text_arm_model_and_price():
    mod = _load_eval_vision()
    p = mod._text_arm_system(True, "ultra", 20.0)
    assert p.model == vision_live.ULTRA_MODEL
    assert p.reasoning_effort == "none"
    assert p.price_input_per_million == vision_live.ULTRA_IN_PER_1M
    assert p.price_output_per_million == vision_live.ULTRA_OUT_PER_1M
