"""Dev-calibrated decision threshold (calibration.unsafe_threshold)."""
from __future__ import annotations

import copy

import pytest

from safety_one.pipeline import SafetyPipeline, load_config
from safety_one.router import apply_decision_threshold, needs_escalation
from safety_one.types import Message, Verdict


class Stub:
    def __init__(self, p, label=None):
        self.p, self.label, self.calls = p, label, 0

    def classify(self, msg):
        self.calls += 1
        label = self.label or ("unsafe" if self.p >= 0.5 else "safe")
        return Verdict(label=label, categories=["violence"] if label == "unsafe" else [],
                       probs={"unsafe": self.p, "safe": 1 - self.p}, model="stub",
                       raw={"usage": {"prompt_tokens": 3, "completion_tokens": 1}})


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["calibration"] = {"unsafe_threshold": 0.8}
    c["confidence_gate"]["gray_zone"] = {"default": [0.6, 0.9], "category_overrides": None}
    return c


def test_threshold_relabels_and_keeps_generated_label(cfg):
    v = apply_decision_threshold(Verdict(label="unsafe", categories=["violence"],
                                         probs={"unsafe": 0.7}), cfg)
    assert v.label == "safe" and v.categories == []
    assert v.raw["generated_label"] == "unsafe" and v.raw["decision_threshold"] == 0.8
    assert apply_decision_threshold(Verdict(label="safe", probs={"unsafe": 0.85}), cfg).label == "unsafe"


def test_null_threshold_is_noop():
    c = load_config()
    v = apply_decision_threshold(Verdict(label="unsafe", probs={"unsafe": 0.1}), c)
    assert v.label == "unsafe" and "decision_threshold" not in v.raw


def test_conflict_check_uses_threshold(cfg):
    cfg["confidence_gate"]["gray_zone"]["default"] = [0.95, 0.99]
    # p=0.7 relabelled safe by t=0.8: consistent, outside band -> no escalation.
    v = apply_decision_threshold(Verdict(label="unsafe", probs={"unsafe": 0.7}), cfg)
    assert needs_escalation(v, cfg) is False


def test_pipeline_band_escalates_and_records_usage(cfg):
    pipe = SafetyPipeline({"specialist": Stub(0.7), "escalation": Stub(0.99)}, cfg)
    v = pipe.classify(Message(id="1", text="borderline"))
    assert v.route == "specialist>escalation" and v.label == "unsafe"
    assert v.raw["pre_escalation"]["label"] == "safe"
    assert v.raw["escalation_usage"] == {"prompt_tokens": 3, "completion_tokens": 1}
    pipe2 = SafetyPipeline({"specialist": Stub(0.3), "escalation": Stub(0.99)}, cfg)
    assert pipe2.classify(Message(id="2", text="hello")).label == "safe"
