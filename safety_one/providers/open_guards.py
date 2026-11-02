"""Availability inventory for requested open guard baselines.

None of the requested Fireworks model pages currently advertise Serverless,
so this provider deliberately refuses to pretend a serverless call is possible.
Published metrics belong in baselines/README.md, not synthetic Verdicts.
"""
from __future__ import annotations

from safety_one.types import Message, Verdict
from .base import Provider


REQUESTED_GUARDS = {
    "llama_guard_3_8b": {"model": "accounts/fireworks/models/llama-guard-3-8b", "serverless": False,
                         "source": "https://fireworks.ai/models/fireworks/llama-guard-3-8b"},
    "qwen3_guard": {"model": None, "serverless": False,
                    "source": "https://arxiv.org/abs/2510.14276"},
    "shieldgemma": {"model": None, "serverless": False,
                    "source": "https://gemma.ai/models/shieldgemma/model-card-v1/"},
    "nemotron_safety_guard": {"model": None, "serverless": False,
                              "source": "https://developer.nvidia.com/blog/?p=107757"},
}


class OpenGuardProvider(Provider):
    """Guard adapter placeholder with explicit non-availability behavior.

    Fireworks lists Llama Guard 3 8B as Ready but explicitly marks Serverless
    unsupported. The other requested families have no matching Fireworks
    serverless model listing. No named model is therefore callable via this
    provider today; use the documented published-score baseline table.
    """
    name = "open-guard"

    def __init__(self, guard: str = "llama_guard_3_8b", *, execute: bool = False):
        if guard not in REQUESTED_GUARDS:
            raise ValueError(f"unknown guard {guard!r}; choose from {sorted(REQUESTED_GUARDS)}")
        self.guard = guard
        self.model = REQUESTED_GUARDS[guard]["model"] or guard
        self.execute = execute

    def classify(self, msg: Message) -> Verdict:
        del msg
        raise RuntimeError(f"{self.guard} is not callable as a Fireworks serverless baseline; consult baselines/README.md for published scores")

