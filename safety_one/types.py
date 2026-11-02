"""Shared dataclasses for Safety One.

Field names/types are a shared interface — do not rename or retype these
fields; other modules import them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal["safe", "unsafe"]

# Valid route values. "rule" = decided by the deterministic rule gate,
# "escalation" = sent to the bigger open reasoning model by the confidence gate.
ROUTES = ("rule", "specialist", "vision", "multilingual", "escalation")


@dataclass
class Message:
    """One unit of content to moderate."""

    id: str
    text: str
    image_path: str | None = None
    lang: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    """Moderation result."""

    label: Label
    categories: list[str] = field(default_factory=list)
    probs: dict[str, float] = field(default_factory=dict)
    route: str = ""
    rule_hits: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    """Output of safety_one.router.route()."""

    route: str  # one of ROUTES
    reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
