"""Deterministic mock providers — the eval and demo run fully offline.

Every MockProvider derives its RNG from (seed, msg.id, msg.text) via SHA-256,
so probabilities/latency/cost are stable across runs and machines (unlike
builtin hash(), which is salted). Probabilities are shaped by a small
keyword->category weight table so mocks behave plausibly: unsafe-looking text
scores high, benign text scores low, "sharpness" lets the bigger escalation
model be more decisive.
"""
from __future__ import annotations

import hashlib
import random
import re

from ..types import Message, Verdict
from .base import Provider

# keyword -> (category, P(unsafe) weight)
_KEYWORDS: dict[str, tuple[str, float]] = {
    "kill": ("violence", 0.55),
    "murder": ("violence", 0.55),
    "bomb": ("weapons", 0.60),
    "terror": ("violence", 0.50),
    "attack": ("violence", 0.30),
    "weapon": ("weapons", 0.35),
    "die": ("violence", 0.40),
    "hate": ("hate_identity", 0.35),
    "racist": ("hate_identity", 0.45),
    "slur": ("hate_identity", 0.50),
    "nazi": ("hate_identity", 0.50),
    "nude": ("sexual", 0.50),
    "porn": ("sexual", 0.55),
    "suicide": ("self_harm", 0.55),
    "self-harm": ("self_harm", 0.55),
    "drugs": ("controlled_substances", 0.35),
    "steal": ("criminal_planning", 0.30),
    "hack": ("malware", 0.25),
    "fraud": ("fraud_deception", 0.40),
}
_WORD_RE = re.compile(r"[a-z\-]+")


def _rng(seed: int, msg: Message) -> random.Random:
    h = hashlib.sha256(f"{seed}:{msg.id}:{msg.text}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "little"))


class MockProvider(Provider):
    """One configurable mock; `default_mock_providers` builds the full set."""

    def __init__(
        self,
        name: str,
        model: str,
        seed: int = 0,
        base_latency_ms: float = 50.0,
        per_char_ms: float = 0.02,
        cost_per_call_usd: float = 0.00002,
        cost_per_1k_chars_usd: float = 0.0001,
        image_cost_usd: float = 0.0,
        sharpness: float = 1.0,
        jitter_ms: float = 15.0,
        jitter_prob: float = 0.05,
    ):
        self.name = name
        self.model = model
        self.seed = seed
        self.base_latency_ms = base_latency_ms
        self.per_char_ms = per_char_ms
        self.cost_per_call_usd = cost_per_call_usd
        self.cost_per_1k_chars_usd = cost_per_1k_chars_usd
        self.image_cost_usd = image_cost_usd
        self.sharpness = sharpness
        self.jitter_ms = jitter_ms
        self.jitter_prob = jitter_prob

    def _prob_unsafe(self, msg: Message, rng: random.Random) -> float:
        words = _WORD_RE.findall((msg.text or "").lower())
        w = sum(_KEYWORDS[t][1] for t in words if t in _KEYWORDS)
        p = 0.03 + w + rng.uniform(-self.jitter_prob, self.jitter_prob)
        p = min(0.97, p)
        # sharpness > 1 pushes probabilities away from 0.5 (more decisive)
        p = 0.5 + (p - 0.5) * self.sharpness
        return min(0.99, max(0.01, p))

    def _categories(self, msg: Message) -> list[str]:
        words = _WORD_RE.findall((msg.text or "").lower())
        seen = [_KEYWORDS[t][0] for t in words if t in _KEYWORDS]
        return sorted(set(seen))

    def classify(self, msg: Message) -> Verdict:
        rng = _rng(self.seed, msg)
        p = self._prob_unsafe(msg, rng)
        label = "unsafe" if p >= 0.5 else "safe"
        text_len = len(msg.text or "")
        latency = (
            self.base_latency_ms
            + self.per_char_ms * text_len
            + rng.uniform(0.0, self.jitter_ms)
        )
        cost = (
            self.cost_per_call_usd
            + self.cost_per_1k_chars_usd * text_len / 1000.0
            + (self.image_cost_usd if msg.image_path else 0.0)
        )
        return Verdict(
            label=label,
            categories=self._categories(msg) if label == "unsafe" else [],
            probs={"safe": round(1.0 - p, 4), "unsafe": round(p, 4)},
            route=self.name,
            model=self.model,
            latency_ms=round(latency, 2),
            cost_usd=round(cost, 6),
            raw={"mock": True, "seed": self.seed},
        )


def default_mock_providers(cfg: dict | None = None,
                           seed: int | None = None) -> dict[str, MockProvider]:
    """Build {specialist, vision, multilingual, escalation} from router.yaml."""
    cfg = cfg or {}
    mcfg = cfg.get("mock", {})
    base_seed = mcfg.get("seed", 0) if seed is None else seed
    out: dict[str, MockProvider] = {}
    for i, role in enumerate(("specialist", "vision", "multilingual", "escalation")):
        pcfg = (cfg.get("providers", {}) or {}).get(role, {})
        mc = dict(pcfg.get("mock", {}))
        out[role] = MockProvider(
            name=role,
            model=pcfg.get("model", f"mock-{role}"),
            seed=base_seed + i,
            base_latency_ms=mc.get("base_latency_ms", 50.0),
            per_char_ms=mc.get("per_char_ms", 0.02),
            cost_per_call_usd=mc.get("cost_per_call_usd", 0.00002),
            cost_per_1k_chars_usd=mc.get("cost_per_1k_chars_usd", 0.0001),
            image_cost_usd=mc.get("image_cost_usd", 0.0),
            sharpness=mc.get("sharpness", 1.0),
            jitter_ms=mcfg.get("jitter_ms", 15.0),
            jitter_prob=mcfg.get("jitter_prob", 0.05),
        )
    return out
