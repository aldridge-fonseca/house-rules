"""Stage 2 (router) and stage 3 (confidence gate) — deterministic, ~0 ms, $0.

route(msg, norm, rule_result, cfg) -> RouteDecision
needs_escalation(verdict, cfg) -> bool

`norm` and `rule_result` are the stage 0/1 outputs and are read duck-typed:

- norm: dict with "text"/"normalized"/"normalized_text"/"lang", an object with
  those attrs, a plain str, or None (falls back to msg.text / msg.lang).
- rule_result: a Verdict, a dict {hits|rule_hits, verdict|label, decisive},
  an object with .hits/.verdict, or a plain list of hit names.

Config comes from config/router.yaml (see that file for keys).
"""
from __future__ import annotations

import re
from typing import Any

from .types import Message, RouteDecision, Verdict

DEFAULT_ROUTE_ORDER = ["rule", "image", "language", "emoji"]

# --------------------------------------------------------------------------
# Features (all deterministic, no dependencies)
# --------------------------------------------------------------------------

# Emoji codepoint blocks: 1F000–1FAFF covers nearly all emoji; the others are
# misc symbols / dingbats / enclosed chars commonly rendered as emoji.
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
)
_INVISIBLE = {0x200D, 0xFE0F}  # ZWJ + variation selector: don't count as chars

# Non-Latin script blocks -> a representative ISO-ish language tag. Only used
# to decide "English or not"; the tag itself is informational. ja (kana) must
# come before zh (kanji) since Japanese mixes both.
_SCRIPT_LANGS = (
    ("ja", ((0x3040, 0x30FF),)),
    ("ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("zh", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
    ("ru", ((0x0400, 0x04FF),)),
    ("ar", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("he", ((0x0590, 0x05FF),)),
    ("hi", ((0x0900, 0x097F),)),
    ("th", ((0x0E00, 0x0E7F),)),
    ("el", ((0x0370, 0x03FF),)),
)
_ENGLISH = {"en", "eng", "english"}


def emoji_count(text: str) -> int:
    return sum(
        1 for ch in text if any(a <= ord(ch) <= b for a, b in _EMOJI_RANGES)
    )


def emoji_density(text: str) -> float:
    """Emoji codepoints / non-space, non-invisible chars."""
    chars = [ch for ch in text if not ch.isspace() and ord(ch) not in _INVISIBLE]
    if not chars:
        return 0.0
    return emoji_count(text) / len(chars)


def detect_lang(text: str, min_chars: int = 3) -> str | None:
    """Script-range heuristic: returns a lang tag for dominant non-Latin
    script with >= min_chars hits, else None (treated as English/unknown)."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for tag, ranges in _SCRIPT_LANGS:
            if any(a <= cp <= b for a, b in ranges):
                counts[tag] = counts.get(tag, 0) + 1
                break
    if not counts:
        return None
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] >= min_chars else None


# --------------------------------------------------------------------------
# Duck-typed readers for stage 0/1 outputs
# --------------------------------------------------------------------------

def _pick(obj: Any, *names: str) -> Any:
    for n in names:
        if isinstance(obj, dict) and obj.get(n) is not None:
            return obj[n]
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def norm_text(norm: Any, msg: Message) -> str:
    if norm is None:
        return msg.text or ""
    if isinstance(norm, str):
        return norm
    t = _pick(norm, "text", "normalized", "normalized_text", "text_norm")
    return t if isinstance(t, str) else (msg.text or "")


def norm_lang(norm: Any, msg: Message) -> str | None:
    if msg.lang:
        return msg.lang
    if norm is not None and not isinstance(norm, str):
        lang = _pick(norm, "lang", "language")
        if isinstance(lang, str):
            return lang
    return None


def rule_hits(rule_result: Any) -> list[str]:
    if rule_result is None:
        return []
    if isinstance(rule_result, (list, tuple, set)):
        return [str(h) for h in rule_result]
    hits = _pick(rule_result, "hits", "rule_hits", "matched_rules")
    if hits is None:
        return []
    return [str(h) for h in hits] if isinstance(hits, (list, tuple, set)) else [str(hits)]


def rule_verdict(rule_result: Any) -> Verdict | dict | None:
    """Decisive verdict from the rule gate, if any. Accepts a Verdict, dict
    with verdict/label, object with .verdict, or dict with decisive=True."""
    if rule_result is None or isinstance(rule_result, (list, tuple, set, str)):
        return None
    if isinstance(rule_result, Verdict):
        return rule_result
    v = _pick(rule_result, "verdict")
    if isinstance(v, (Verdict, dict)):
        return v
    if isinstance(v, str) or _pick(rule_result, "decisive") or _pick(rule_result, "label"):
        if isinstance(rule_result, dict):
            return rule_result
        # Object-style results (rules.RuleResult): carry categories, hits and
        # confidence through, not just the label.
        out: dict = {"label": v if isinstance(v, str) else (_pick(rule_result, "label") or "unsafe")}
        cats = _pick(rule_result, "categories")
        if cats:
            out["categories"] = list(cats)
        hits = _pick(rule_result, "hits", "rule_hits")
        if hits:
            out["hits"] = list(hits)
        conf = _pick(rule_result, "confidence")
        if isinstance(conf, (int, float)) and conf > 0:
            p_unsafe = float(conf) if out["label"] == "unsafe" else 1.0 - float(conf)
            out["probs"] = {"safe": round(1.0 - p_unsafe, 4), "unsafe": round(p_unsafe, 4)}
            out["confidence"] = float(conf)
        return out
    return None


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

def extract_features(msg: Message, norm: Any, rule_result: Any, cfg: dict) -> dict:
    text = norm_text(norm, msg)
    rcfg = cfg.get("router", {})
    min_chars = int(rcfg.get("min_script_chars", 3))
    lang = norm_lang(norm, msg)
    detected = detect_lang(text, min_chars=min_chars)
    return {
        "has_image": bool(msg.image_path),
        "emoji_count": emoji_count(text),
        "emoji_density": round(emoji_density(text), 4),
        "lang": lang or detected,
        "lang_source": ("msg" if msg.lang else "norm" if lang else "detected" if detected else None),
        "rule_hits": rule_hits(rule_result),
        "text_len": len(text),
    }


def route(msg: Message, norm: Any = None, rule_result: Any = None,
          cfg: dict | None = None) -> RouteDecision:
    cfg = cfg or {}
    rcfg = cfg.get("router", {})
    english = {str(x).lower() for x in rcfg.get("known_english", sorted(_ENGLISH))}
    feats = extract_features(msg, norm, rule_result, cfg)

    dec = RouteDecision(route=rcfg.get("default_route", "specialist"),
                        reasons=[], features=feats)
    checks = {
        "rule": (lambda: rule_verdict(rule_result) is not None,
                 "rule", "rule gate returned a decisive verdict"),
        "image": (lambda: feats["has_image"],
                  rcfg.get("image_route", "vision"),
                  "image present"),
        "language": (lambda: feats["lang"] is not None
                     and feats["lang"].lower() not in english,
                     rcfg.get("non_english_route", "multilingual"),
                     f"non-English lang={feats['lang']}"),
        "emoji": (lambda: feats["emoji_density"]
                  >= float(rcfg.get("emoji_density_threshold", 0.10)),
                  rcfg.get("emoji_route", "vision"),
                  f"emoji density {feats['emoji_density']:.2f} >= "
                  f"{float(rcfg.get('emoji_density_threshold', 0.10)):.2f}"),
    }
    for step in rcfg.get("route_order", DEFAULT_ROUTE_ORDER):
        if step not in checks:
            continue
        hit, target, reason = checks[step]
        if hit():
            dec.route, dec.reasons = target, [reason]
            return dec
    dec.reasons = ["default: plain English text"]
    return dec


# --------------------------------------------------------------------------
# Stage 3: confidence gate
# --------------------------------------------------------------------------

def gray_zone(verdict: Verdict, cfg: dict | None) -> tuple[float, float]:
    """Effective [lo, hi] band on P(unsafe). Category overrides widen the band
    for high-stakes categories (escalate unless very sure)."""
    g = (cfg or {}).get("confidence_gate", {})
    default = g.get("gray_zone", {}).get("default", [0.35, 0.65])
    overrides = g.get("gray_zone", {}).get("category_overrides", {}) or {}
    bands = [overrides[c] for c in verdict.categories if c in overrides]
    if not bands:
        return float(default[0]), float(default[1])
    return min(float(b[0]) for b in bands), max(float(b[1]) for b in bands)


def decision_threshold(cfg: dict | None) -> float | None:
    """calibration.unsafe_threshold from config (dev-tuned), or None = keep the generated label."""
    t = ((cfg or {}).get("calibration") or {}).get("unsafe_threshold")
    return None if t is None else float(t)


def apply_decision_threshold(verdict: Verdict, cfg: dict | None) -> Verdict:
    """Re-decide the pre-gate label as P(unsafe) >= calibration.unsafe_threshold.

    No-op without a configured threshold or without a 'unsafe' probability.
    The generated label is kept in raw["generated_label"].
    """
    t = decision_threshold(cfg)
    if t is None or "unsafe" not in (verdict.probs or {}):
        return verdict
    label = "unsafe" if float(verdict.probs["unsafe"]) >= t else "safe"
    if label != verdict.label:
        verdict.raw["generated_label"] = verdict.label
        verdict.label = label
        if label == "safe":
            verdict.categories = []
    verdict.raw["decision_threshold"] = t
    return verdict


def needs_escalation(verdict: Verdict, cfg: dict | None) -> bool:
    g = (cfg or {}).get("confidence_gate", {})
    if not g.get("enabled", True):
        return False
    p = verdict.probs.get(
        "unsafe", 1.0 if verdict.label == "unsafe" else 0.0
    )
    # label/prob conflict is itself a gray zone
    t = decision_threshold(cfg)
    if (verdict.label == "unsafe") != (p >= (0.5 if t is None else t)):
        return True
    lo, hi = gray_zone(verdict, cfg)
    return lo <= p <= hi
