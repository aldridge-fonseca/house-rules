"""End-to-end pipeline: normalize -> rules -> route -> provider -> gate.

    SafetyPipeline(providers: dict[str, Provider], cfg).classify(msg) -> Verdict

Stage 0 (normalize) and stage 1 (rules) live in safety_one/normalize.py and
safety_one/rules.py; thin guarded adapters call into them when present and
otherwise fall back to minimal local implementations so the pipeline also
runs offline.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

import yaml

from .router import (
    apply_decision_threshold,
    needs_escalation,
    norm_text,
    route,
    rule_hits,
    rule_verdict,
)
from .types import Message, Verdict

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "router.yaml",
)


def load_config(path: str | None = None) -> dict:
    path = path or _CONFIG_PATH
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------
# Stage 0 adapter: normalize (safety_one/normalize.py)
# --------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)


def _fallback_normalize(msg: Message) -> dict:
    text = unicodedata.normalize("NFKC", msg.text or "").translate(_ZERO_WIDTH)
    return {
        "raw_text": msg.text or "",
        "text": text,
        "lang": msg.lang,
        "meta": {"fallback": True},
    }


def _normalize(msg: Message, cfg: dict) -> Any:
    # normalize(text: str) -> NormalizedText. Always pass the
    # text itself (never the Message: an empty text used to stringify the
    # Message repr into the normalized text).
    try:
        from .normalize import normalize as _norm_fn
    except Exception:
        return _fallback_normalize(msg)
    return _norm_fn(msg.text or "")


# --------------------------------------------------------------------------
# Stage 1 adapter: rule gate (safety_one/rules.py)
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE_RE = re.compile(r"\b\+?1?[ -.]\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def _fallback_rules(msg: Message, norm: Any) -> dict:
    """Minimal PII gate so PII -> 'rule' works before rules.py lands.

    Returns {"hits": [...], "verdict": Verdict|None, "decisive": bool}.
    """
    text = norm_text(norm, msg)
    hits: list[str] = []
    if _EMAIL_RE.search(text):
        hits.append("pii:email")
    if _PHONE_RE.search(text):
        hits.append("pii:phone")
    if _SSN_RE.search(text):
        hits.append("pii:ssn")
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if _luhn_ok(digits):
            hits.append("pii:card")
            break
    verdict = None
    if hits:
        verdict = Verdict(
            label="unsafe",
            categories=["pii"],
            probs={"safe": 0.01, "unsafe": 0.99},
            route="rule",
            rule_hits=hits,
            model="rules",
            raw={"fallback": True},
        )
    return {"hits": hits, "verdict": verdict, "decisive": bool(hits)}


def _rules(msg: Message, norm: Any, cfg: dict) -> Any:
    # check(norm: NormalizedText | str, policy=None) -> RuleResult.
    # Optional per-customer policy: cfg["rules"]["policy"] (builtin name, YAML
    # path/string or spec dict).
    try:
        from .normalize import NormalizedText
        from .rules import check as _check
    except Exception:
        return _fallback_rules(msg, norm)
    policy = (cfg.get("rules") or {}).get("policy")
    arg = norm if isinstance(norm, NormalizedText) else (msg.text or "")
    return _check(arg, policy)


# Rule-gate categories -> data/TAXONOMY.md slugs. The detailed reason stays in
# rule_hits ("jailbreak:<slug>", "policy:<policy>:<cat>:<term>").
_TAXONOMY_SLUGS = {
    "criminal_planning", "violence", "hate_identity", "harassment", "needs_caution",
    "controlled_substances", "profanity", "sexual", "pii", "weapons", "self_harm",
    "fraud_deception", "illegal_activity", "sexual_minor", "other", "immoral_unethical",
    "misinformation", "threat", "unauthorized_advice", "manipulation", "malware",
    "high_risk_gov", "copyright", "economic_harm", "discrimination", "environmental_harm",
}


def _taxonomy_categories(cats: list[str]) -> list[str]:
    out: list[str] = []
    for c in cats:
        c = c.split(":", 1)[1] if c.startswith("policy:") else c
        slug = c if c in _TAXONOMY_SLUGS else "other"
        if slug not in out:
            out.append(slug)
    return out


def _rule_verdict_obj(rule_result: Any) -> Verdict:
    rv = rule_verdict(rule_result)
    if isinstance(rv, Verdict):
        return rv
    v = rv if isinstance(rv, dict) else {}
    raw_cats = list(v.get("categories") or ["other"])
    return Verdict(
        label=v.get("label", "unsafe"),
        categories=_taxonomy_categories(raw_cats),
        probs=dict(v.get("probs", {"safe": 0.01, "unsafe": 0.99})),
        route="rule",
        rule_hits=rule_hits(rule_result),
        model="rules",
        raw={"rule_categories": raw_cats, "rule_confidence": v.get("confidence")},
    )


def _provider_message(msg: Message, norm: Any, cfg: dict) -> Message:
    """Message handed to model providers. Default (router.provider_text:
    normalized) sends the de-obfuscated text so stage 0 reaches the models;
    the raw text stays in meta["raw_text"]. "raw" keeps the original text."""
    if (cfg.get("router") or {}).get("provider_text", "normalized") != "normalized":
        return msg
    text = getattr(norm, "normalized", None)
    if text is None and isinstance(norm, dict):
        text = norm.get("text")
    if not isinstance(text, str) or text == (msg.text or ""):
        return msg
    meta = dict(msg.meta or {})
    meta["raw_text"] = msg.text
    meta["normalize_transforms"] = list(getattr(norm, "transforms_applied", []) or [])
    return Message(id=msg.id, text=text, image_path=msg.image_path, lang=msg.lang, meta=meta)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class SafetyPipeline:
    """providers: {route_name: Provider}; cfg: parsed router.yaml dict."""

    def __init__(self, providers: dict, cfg: dict | None = None):
        self.providers = providers
        self.cfg = cfg if cfg is not None else load_config()

    def classify(self, msg: Message) -> Verdict:
        norm = _normalize(msg, self.cfg)
        rule_result = _rules(msg, norm, self.cfg)
        decision = route(msg, norm, rule_result, self.cfg)

        trail = [decision.route]
        models: list[str] = []
        latency = 0.0
        cost = 0.0
        escalated = False

        if decision.route == "rule":
            verdict = _rule_verdict_obj(rule_result)
            models.append(verdict.model or "rules")
        else:
            provider = self.providers.get(decision.route) or self.providers.get(
                "specialist"
            )
            if provider is None:
                raise KeyError(
                    f"no provider for route {decision.route!r} and no specialist fallback"
                )
            pmsg = _provider_message(msg, norm, self.cfg)
            verdict = apply_decision_threshold(provider.classify(pmsg), self.cfg)
            latency += verdict.latency_ms
            cost += verdict.cost_usd
            models.append(verdict.model)

            if needs_escalation(verdict, self.cfg):
                esc = self.providers.get("escalation")
                if esc is not None:
                    ev = esc.classify(pmsg)
                    escalated = True
                    trail.append("escalation")
                    latency += ev.latency_ms
                    cost += ev.cost_usd
                    models.append(ev.model)
                    verdict.raw["pre_escalation"] = {
                        "label": verdict.label,
                        "probs": dict(verdict.probs),
                        "model": verdict.model,
                    }
                    verdict.raw["escalation_usage"] = (ev.raw or {}).get("usage")
                    verdict.raw["escalation_probability_source"] = (ev.raw or {}).get(
                        "probability_source")
                    verdict.label = ev.label
                    verdict.categories = ev.categories or verdict.categories
                    verdict.probs = ev.probs or verdict.probs

        verdict.route = ">".join(trail)
        verdict.rule_hits = rule_hits(rule_result)
        verdict.model = ">".join(m for m in models if m)
        verdict.latency_ms = round(latency, 2)
        verdict.cost_usd = round(cost, 6)
        verdict.raw.update(
            {
                "route_reasons": decision.reasons,
                "features": decision.features,
                "model_chain": models,
                "escalated": escalated,
            }
        )
        return verdict
