"""Router (stage 2) and confidence gate (stage 3) tests."""
import os

import pytest
import yaml

from safety_one.router import (
    detect_lang,
    emoji_density,
    gray_zone,
    needs_escalation,
    route,
)
from safety_one.types import Message, Verdict

CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "router.yaml",
)


@pytest.fixture
def cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def msg(text="hello", **kw):
    return Message(id="m1", text=text, **kw)


# ---- routing table -------------------------------------------------------

def test_default_english_text_goes_specialist(cfg):
    d = route(msg("how do I bake sourdough bread?"), cfg=cfg)
    assert d.route == "specialist"
    assert d.features["text_len"] > 0


def test_image_goes_vision(cfg):
    d = route(msg("what does this say?", image_path="/tmp/x.png"), cfg=cfg)
    assert d.route == "vision"
    assert d.features["has_image"]


def test_image_beats_non_english(cfg):
    d = route(msg("Привет мир, что на картинке?", image_path="/tmp/x.png"), cfg=cfg)
    assert d.route == "vision"


def test_emoji_heavy_goes_configured_route(cfg):
    text = "get out " + "\U0001F595" * 12
    assert emoji_density(text) >= cfg["router"]["emoji_density_threshold"]
    d = route(msg(text), cfg=cfg)
    assert d.route == cfg["router"]["emoji_route"]

    cfg2 = yaml.safe_load(open(CFG_PATH))
    cfg2["router"]["emoji_route"] = "specialist"
    d2 = route(msg(text), cfg=cfg2)
    assert d2.route == "specialist"


def test_emoji_light_stays_specialist(cfg):
    d = route(msg("have a nice day \U0001F600"), cfg=cfg)
    assert d.route == "specialist"


@pytest.mark.parametrize(
    "text",
    [
        "Привет, как ты сегодня?",          # cyrillic
        "你好世界，今天天气怎么样？",        # CJK
        "مرحبا كيف حالك اليوم",             # arabic
        "こんにちは、ですか",            # kana
    ],
)
def test_non_english_goes_multilingual(cfg, text):
    assert detect_lang(text) is not None
    d = route(msg(text), cfg=cfg)
    assert d.route == "multilingual"


def test_explicit_lang_field_goes_multilingual(cfg):
    d = route(msg("Bonjour tout le monde", lang="fr"), cfg=cfg)
    assert d.route == "multilingual"
    assert d.features["lang_source"] == "msg"


def test_rule_verdict_short_circuits(cfg):
    rr = {"hits": ["pii:email"], "decisive": True,
          "verdict": {"label": "unsafe", "categories": ["pii"]}}
    d = route(msg("mail me at a@b.com"), rule_result=rr, cfg=cfg)
    assert d.route == "rule"
    assert d.features["rule_hits"] == ["pii:email"]


def test_rule_hits_without_verdict_still_route(cfg):
    rr = {"hits": ["keyword:watchlist"], "decisive": False}
    d = route(msg("ordinary sentence"), rule_result=rr, cfg=cfg)
    assert d.route == "specialist"
    assert d.features["rule_hits"] == ["keyword:watchlist"]


def test_route_uses_normalized_text(cfg):
    norm = {"text": "Привет мир это проверка", "lang": None}
    d = route(msg("x"), norm=norm, cfg=cfg)
    assert d.route == "multilingual"


# ---- confidence gate ------------------------------------------------------

@pytest.mark.parametrize(
    "p,expected",
    [(0.10, False), (0.34, False), (0.35, True), (0.50, True),
     (0.65, True), (0.66, False), (0.95, False)],
)
def test_gray_zone_default(cfg, p, expected):
    v = Verdict(label="unsafe" if p >= 0.5 else "safe",
                probs={"unsafe": p, "safe": 1 - p})
    assert needs_escalation(v, cfg) is expected


def test_category_override_widens_gray_zone(cfg):
    # sexual_minor band is [0.05, 0.95]: p=0.9 would pass the default gate
    # but must escalate for this category.
    v = Verdict(label="unsafe", categories=["sexual_minor"],
                probs={"unsafe": 0.90})
    assert needs_escalation(v, cfg) is True
    assert gray_zone(v, cfg) == (0.05, 0.95)

    v2 = Verdict(label="unsafe", categories=["sexual_minor"],
                 probs={"unsafe": 0.97})
    assert needs_escalation(v2, cfg) is False


def test_label_prob_conflict_escalates(cfg):
    v = Verdict(label="unsafe", probs={"unsafe": 0.20})
    assert needs_escalation(v, cfg) is True


def test_gate_disabled(cfg):
    cfg["confidence_gate"]["enabled"] = False
    v = Verdict(label="safe", probs={"unsafe": 0.5})
    assert needs_escalation(v, cfg) is False


def test_config_category_keys_are_taxonomy_slugs():
    """router.yaml gray-zone overrides and mock keyword categories must use
    data/TAXONOMY.md slugs, or the per-category gate silently never fires."""
    import re
    from pathlib import Path
    import yaml
    from safety_one.providers.mock import _KEYWORDS
    root = Path(__file__).resolve().parents[1]
    tax = (root / "data" / "TAXONOMY.md").read_text(encoding="utf-8")
    section = tax.split("## Categories", 1)[1].split("\n## ", 1)[0]
    slugs = set(re.findall(r"^\| `([a-z_]+)` \|", section, flags=re.M))
    assert len(slugs) == 26
    cfg = yaml.safe_load((root / "config" / "router.yaml").read_text(encoding="utf-8"))
    overrides = cfg["confidence_gate"]["gray_zone"]["category_overrides"]
    assert set(overrides) <= slugs, set(overrides) - slugs
    assert {c for c, _ in _KEYWORDS.values()} <= slugs
