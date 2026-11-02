import json

import pytest

from safety_one.providers.frontier import FrontierProvider, OpenAIModerationProvider, moderation_prompt
from safety_one.types import Message


class Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


@pytest.mark.parametrize("provider,env,body,expected", [
    ("openai", "OPENAI_API_KEY", {"output": [{"type": "message", "content": [{"type": "output_text", "text": '```json\n{"label":"unsafe","categories":["violence"]}\n```'}]}], "usage": {"input_tokens": 100, "output_tokens": 8}}, 0.0014),
    ("anthropic", "ANTHROPIC_API_KEY", {"content": [{"type": "text", "text": '{"label":"safe","categories":[]} trailing'}], "usage": {"input_tokens": 100, "output_tokens": 8}}, 0.0014),
    ("gemini", "GEMINI_API_KEY", {"candidates": [{"content": {"parts": [{"text": '{"label":"safe","categories":[]}'}]}}], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5, "thoughtsTokenCount": 3}}, 0.00028),
])
def test_mocked_http_extracts_usage_and_cost(monkeypatch, provider, env, body, expected):
    monkeypatch.setenv(env, "test-key-never-printed")
    seen = {}

    def opener(req, timeout):
        seen["request"] = req
        seen["timeout"] = timeout
        return Response(body)

    result = FrontierProvider(provider, execute=True, opener=opener).classify(Message(id="r1", text="test content"))
    assert result.cost_usd == pytest.approx(expected)
    assert result.model
    sent = json.loads(seen["request"].data)
    assert sent.get("temperature", sent.get("generationConfig", {}).get("temperature")) == 0
    assert "test-key-never-printed" not in seen["request"].data.decode()
    assert "test-key-never-printed" not in seen["request"].full_url
    assert result.label in ("safe", "unsafe")


def test_shared_prompt_has_locked_taxonomy_and_strict_output():
    prompt = moderation_prompt()
    assert "sexual_minor" in prompt and "needs_caution" in prompt
    assert '"label":"safe"|"unsafe"' in prompt


def test_default_dry_run_never_calls_network():
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted HTTP")

    result = FrontierProvider("openai", opener=forbidden).classify(Message(id="r", text="hello"))
    assert result.raw["simulated"] is True
    assert result.cost_usd == 0


@pytest.mark.parametrize("provider,env,body", [
    ("openai", "OPENAI_API_KEY", {"output": [{"type": "message", "content": [{"type": "output_text", "text": '{"label":"safe","categories":[]}'}]}], "usage": {"input_tokens": 4, "output_tokens": 3}}),
    ("anthropic", "ANTHROPIC_API_KEY", {"content": [{"type": "text", "text": '{"label":"safe","categories":[]}'}], "usage": {"input_tokens": 4, "output_tokens": 3}}),
    ("gemini", "GEMINI_API_KEY", {"candidates": [{"content": {"parts": [{"text": '{"label":"safe","categories":[]}'}]}}], "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3}}),
])
def test_images_are_included_in_each_provider_request(tmp_path, monkeypatch, provider, env, body):
    image = tmp_path / "tiny.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv(env, "mock-key")
    seen = {}

    def opener(req, timeout):
        seen["body"] = json.loads(req.data)
        return Response(body)

    FrontierProvider(provider, execute=True, opener=opener).classify(Message(id="img", text="caption", image_path=str(image)))
    encoded = "iVBORw0KGgo="
    sent = json.dumps(seen["body"])
    assert encoded in sent
    assert "taxonomy" in sent.lower()


def test_native_moderation_endpoint_is_free_and_maps_categories(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
    seen = {}

    def opener(req, timeout):
        seen["url"] = req.full_url
        return Response({"results": [{"flagged": True, "categories": {"hate/threatening": True}, "category_scores": {"hate/threatening": 0.9}}]})

    verdict = OpenAIModerationProvider(execute=True, opener=opener).classify(Message(id="m", text="x"))
    assert seen["url"].endswith("/v1/moderations")
    assert verdict.label == "unsafe"
    assert verdict.categories == ["hate_identity", "threat"]
    assert verdict.cost_usd == 0
