"""Provider interface for Safety One.

    class Provider: name: str; def classify(self, msg: Message) -> Verdict

All classifier backends (mock, fireworks, frontier, open-guards) implement
this. Keep it dependency-free: every worker imports this module first.
"""
from __future__ import annotations

from ..types import Message, Verdict


class Provider:
    """Base class for classifier providers."""

    name: str = "provider"

    def classify(self, msg: Message) -> Verdict:
        raise NotImplementedError
