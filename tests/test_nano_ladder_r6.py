"""nano_ladder factories (nvidia-v1 run). No network calls."""
from __future__ import annotations

import pytest

import scripts.nano_ladder as ladder


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-not-a-key")
    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/a/deployments/tuned")
    monkeypatch.setenv("NANO_BASE_MODEL", "accounts/a/deployments/base")
    monkeypatch.delenv("NANO_ESCALATION", raising=False)


def test_final_rawtext_is_full_with_raw_provider_text():
    s = getattr(ladder, "final-rawtext")()
    assert s.cfg["router"]["provider_text"] == "raw"
    assert s.cfg["confidence_gate"]["enabled"] is not False
    assert s.providers["specialist"].model == "accounts/a/deployments/tuned"


def test_nano_untuned_uses_base_deployment_without_escalation():
    s = getattr(ladder, "nano-untuned")()
    assert s.cfg["confidence_gate"]["enabled"] is False
    assert s.cfg["router"].get("provider_text", "normalized") == "normalized"
    for role in ("specialist", "multilingual"):
        assert s.providers[role].model == "accounts/a/deployments/base"


def test_nano_untuned_requires_base_env(monkeypatch):
    monkeypatch.delenv("NANO_BASE_MODEL")
    with pytest.raises(RuntimeError, match="NANO_BASE_MODEL"):
        getattr(ladder, "nano-untuned")()
