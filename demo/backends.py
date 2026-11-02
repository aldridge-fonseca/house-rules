"""Lane backends for the race demo.

Two lanes, both returning a verdict dict shaped like the Verdict schema:
  {label, categories, probs, route, rule_hits, model, latency_ms, cost_usd}

"ours" tries the real pipeline first:
    safety_one.pipeline.SafetyPipeline + safety_one.types.Message
If import or construction fails, we fall back to the deterministic
heuristic mock below, which mimics the
planned architecture (normalize -> rules -> router -> gray-zone escalate)
and is clearly labeled model="mock/...".

"frontier" tries safety_one.providers.frontier (real client when
keys exist, dry-run otherwise); falls back to a deterministic mock that
reproduces the classic failure modes: no de-obfuscation (misses leet /
homoglyph / emoji-carried threats), no image lane, and over-blocks benign
"kill/die/break into" phrasing.

Nothing here makes network calls. Deterministic jitter is seeded by case
id so replay and live-mock runs are identical.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------- real wiring

def _verdict_to_dict(v):
    """Accept a Verdict dataclass / attrs / dict / plain object."""
    if isinstance(v, dict):
        d = dict(v)
    else:
        try:
            import dataclasses
            if dataclasses.is_dataclass(v):
                d = dataclasses.asdict(v)
            else:
                d = {k: getattr(v, k) for k in dir(v) if not k.startswith("_")}
        except Exception:
            d = {}
    out = {
        "label": d.get("label", "safe"),
        "categories": list(d.get("categories") or []),
        "probs": dict(d.get("probs") or {}),
        "route": d.get("route", "specialist"),
        "rule_hits": list(d.get("rule_hits") or []),
        "model": d.get("model", "house-rules"),
        "latency_ms": float(d.get("latency_ms") or 0.0),
        "cost_usd": float(d.get("cost_usd") or 0.0),
        # raw carries route_reasons / model_chain / pre_escalation — the UI
        # uses them for the route-trail badges and "router helped" notes.
        "raw": dict(d.get("raw") or {}),
    }
    if "unsafe" not in out["probs"]:
        out["probs"]["unsafe"] = 1.0 if out["label"] == "unsafe" else 0.0
    return out


def _resolve_image(path):
    """Resolve a demo-relative image_path; SVG meme cases become text captions.

    Fireworks chat/vision endpoints don't accept SVG, and we don't deploy the
    4×B200 Super VL arm for the demo budget. The one meme case embeds its
    hate text in the SVG — fold that into the message text so the live
    specialist still sees the payload.
    """
    if not path:
        return None, None
    # UI paths look like /assets/meme1.svg — strip the leading slash for lookup.
    rel = path[1:] if str(path).startswith("/") else path
    p = Path(rel)
    if not p.is_file():
        for base in (DEMO_DIR, DEMO_DIR / "assets", REPO_ROOT):
            cand = (base / rel).resolve()
            if cand.is_file():
                p = cand
                break
            cand = (base / Path(rel).name).resolve()
            if cand.is_file():
                p = cand
                break
    if not p.is_file():
        return None, None
    if p.suffix.lower() == ".svg":
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None, None
        # Pull visible <text>…</text> contents as a caption for the text lane.
        texts = re.findall(r"<text[^>]*>(.*?)</text>", raw, flags=re.I | re.S)
        caption = " ".join(t.strip() for t in texts if t.strip())
        return None, caption or None
    return str(p), None


def _make_message(row, Message):
    image, caption = _resolve_image(row.get("image_path"))
    text = row["text"]
    meta = dict(row.get("meta") or {})
    if caption:
        text = f"{text}\n[image text: {caption}]"
        meta.setdefault("caption", caption)
        meta["image_folded_to_text"] = True
    return Message(id=row["id"], text=text, image_path=image,
                   lang=row.get("lang"), meta=meta)


def _provider_kwargs(role_cfg):
    mc = dict((role_cfg or {}).get("mock", {}))
    return dict(
        base_latency_ms=mc.get("base_latency_ms", 50.0),
        per_char_ms=mc.get("per_char_ms", 0.02),
        cost_per_call_usd=mc.get("cost_per_call_usd", 0.00002),
        cost_per_1k_chars_usd=mc.get("cost_per_1k_chars_usd", 0.0001),
        image_cost_usd=mc.get("image_cost_usd", 0.0),
        sharpness=mc.get("sharpness", 1.0),
        jitter_ms=15.0, jitter_prob=0.05)


def _demo_providers(cfg):
    """Build the pipeline's provider dict from demo-local MockProvider subclasses.

    Reuses work-4's MockProvider (deterministic RNG, latency/cost model,
    sharpness) but scans NORMALIZED text with the safety-demo keyword table,
    so the stand-in specialist behaves like a tuned guard model on
    obfuscated input instead of a generic keyword bump.
    """
    from safety_one.providers.mock import MockProvider

    class DemoProvider(MockProvider):
        def _prob_unsafe(self, msg, rng):
            text = msg.text + " " + (msg.meta or {}).get("caption", "")
            p_hit, _ = _scan(_STRONG, normalize(text))
            p = 0.05 + p_hit + rng.uniform(-self.jitter_prob, self.jitter_prob)
            p = 0.5 + (p - 0.5) * self.sharpness
            return min(0.99, max(0.01, p))

        def _categories(self, msg):
            text = msg.text + " " + (msg.meta or {}).get("caption", "")
            _, cats = _scan(_STRONG, normalize(text))
            return cats

    provs = {}
    # systems.yaml pipeline_roles: specialist/multilingual honor FIREWORKS_MODEL.
    model_env = {"specialist": "FIREWORKS_MODEL", "multilingual": "FIREWORKS_MODEL"}
    for i, role in enumerate(("specialist", "vision", "multilingual", "escalation")):
        pcfg = (cfg.get("providers", {}) or {}).get(role, {})
        env_key = model_env.get(role)
        model = (os.environ.get(env_key) if env_key else None) \
            or pcfg.get("model", f"mock-{role}")
        provs[role] = DemoProvider(
            name=role, model=model,
            seed=20261102 + i, **_provider_kwargs(pcfg))
    return provs


def _ensure_env():
    try:
        from safety_one.registry import _load_env_file
        _load_env_file()
    except Exception:
        pass


def load_ours():
    """(callable(row)->verdict_dict, mode) using the real SafetyPipeline, or (None, None).

    LIVE path (preferred): scripts.nano_ladder full pipeline — the same
    TrainedPromptProvider + FIREWORKS_MODEL deployment used for results.
    Falls back to demo MockProviders when keys/deployment are missing.
    """
    _ensure_env()
    try:
        from safety_one.types import Message
    except Exception:
        return None, None

    pipe = None
    mode = None
    if os.environ.get("FIREWORKS_API_KEY") and os.environ.get("FIREWORKS_MODEL"):
        try:
            from scripts.nano_ladder import _build
            pipe = _build("full")
            # Demo budget: no Super VL deployment. Point vision at the tuned
            # specialist so the single meme case still runs live (SVG folded
            # to text in _make_message).
            if not os.environ.get("FIREWORKS_VISION_MODEL"):
                pipe.providers["vision"] = pipe.providers["specialist"]
            mode = "live"
        except Exception:
            pipe = None
            mode = None

    if pipe is None:
        try:
            from safety_one.pipeline import SafetyPipeline, load_config
            cfg = load_config()
            pipe = SafetyPipeline(_demo_providers(cfg), cfg)
            mode = "mock"
        except Exception:
            return None, None

    def run(row):
        msg = _make_message(row, Message)
        try:
            return attach_norm(_verdict_to_dict(pipe.classify(msg)), row["text"])
        except Exception as e:
            return mock_ours(row, note=f"pipeline-error:{type(e).__name__}")
    return run, mode


def _frontier_sim():
    """MockProvider tuned to behave like a raw-text frontier call.

    Scans the raw (un-normalized) text only, so it naturally misses leet /
    homoglyph / emoji-carried threats — the failure modes our normalize +
    route stages exist to fix. Costs are set to flagship scale (~$0.002/case).
    """
    try:
        from safety_one.providers.mock import MockProvider
    except Exception:
        return None

    class DemoFrontierProvider(MockProvider):
        def _prob_unsafe(self, msg, rng):
            p_hit, _ = _scan(_STRONG_FRONTIER, (msg.text or "").lower())
            p = 0.05 + p_hit + rng.uniform(-self.jitter_prob, self.jitter_prob)
            p = 0.5 + (p - 0.5) * self.sharpness
            return min(0.99, max(0.01, p))

        def _categories(self, msg):
            _, cats = _scan(_STRONG_FRONTIER, (msg.text or "").lower())
            return cats

    model = os.environ.get("FRONTIER_MODEL_A") or os.environ.get("FRONTIER_MODEL") or "frontier"
    return DemoFrontierProvider(
        name="frontier", model=f"{model}-sim",
        seed=99173, base_latency_ms=650.0, per_char_ms=2.0,
        cost_per_call_usd=0.0018, cost_per_1k_chars_usd=0.004,
        image_cost_usd=0.002, sharpness=0.95,
        jitter_ms=350.0, jitter_prob=0.08)


def load_frontier():
    """(callable(row)->verdict_dict, mode) for the frontier lane, or (None, None).

    LIVE path: registry frontier-router-a (GPT-6.1 Sol via ROUTER_*) when
    DEMO_LIVE_FRONTIER=1 and ROUTER_API_KEY are set — full policy + taxonomy
    prompt (`moderation_prompt()`), matching the frontier arms in results/
    (not the policy-blind label-only prompt used for NVIDIA zero-shot baselines).
    Otherwise a MockProvider sim labeled `*-sim`.
    """
    _ensure_env()
    try:
        from safety_one.types import Message
    except Exception:
        return None, None

    prov = None
    mode = "mock"
    if os.environ.get("DEMO_LIVE_FRONTIER") == "1" and os.environ.get("ROUTER_API_KEY"):
        try:
            from safety_one.registry import build_system
            # Prefer arm A (FRONTIER_MODEL_A = openai/gpt-6.1-sol); fall back
            # to the single-arm frontier-router.
            for name in ("frontier-router-a", "frontier-router"):
                try:
                    prov = build_system(name, mode="live")
                    break
                except Exception:
                    prov = None
            if prov is not None:
                mode = "live"
        except Exception:
            prov = None

    if prov is None and os.environ.get("DEMO_LIVE_FRONTIER") == "1":
        # Direct provider path (OPENAI/ANTHROPIC/GEMINI) as a secondary option.
        try:
            from safety_one.providers.frontier import API, FrontierProvider
            provider = os.environ.get("FRONTIER_PROVIDER", "openai")
            if provider in API and os.environ.get(API[provider][1]):
                prov = FrontierProvider(provider, execute=True)
                mode = "live"
        except Exception:
            prov = None

    if prov is None:
        prov = _frontier_sim()
        mode = "mock"
    if prov is None:
        return None, None

    def run(row):
        try:
            return _verdict_to_dict(prov.classify(_make_message(row, Message)))
        except Exception as e:
            # Content-filter refusals are a real frontier outcome — surface them.
            name = type(e).__name__
            if "ContentFilter" in name or "FrontierContentFilter" in name:
                return {
                    "label": "unsafe", "categories": ["other"],
                    "probs": {"unsafe": 1.0, "safe": 0.0},
                    "route": "frontier-router", "rule_hits": [],
                    "model": getattr(prov, "model", "frontier"),
                    "latency_ms": 0.0, "cost_usd": 0.0,
                    "raw": {"refused": True, "error": name},
                }
            return mock_frontier(row, note=f"frontier-error:{name}")
    return run, mode


def load_lanes():
    """(ours_fn, frontier_fn, modes: dict)"""
    ours, om = load_ours()
    frontier, fm = load_frontier()
    return (ours or mock_ours, frontier or mock_frontier,
            {"ours": om or "mock-fallback", "frontier": fm or "mock-fallback"})


def reload_lanes():
    """Re-resolve lane backends (after env changes such as FIREWORKS_MODEL)."""
    global _LANES
    _LANES = load_lanes()
    return _LANES

# ------------------------------------------------------------ deterministic mock

_HOMO = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ո": "n", "ս": "u", "ν": "v",
    "м": "m", "т": "t", "к": "k", "в": "b", "г": "r", "у": "y", "п": "n",
})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "8": "b", "@": "a", "$": "s", "!": "i"})
_ZW = dict.fromkeys(map(ord, "​‌‍﻿"), None)
_EMOJI = {"🔪": " knife ", "🩸": " blood ", "👊": " punch ", "🔫": " gun ",
          "💣": " bomb ", "☠️": " death ", "💀": " death ", "⚰️": " coffin death ",
          "🪦": " grave death ", "🗡️": " dagger ", "😈": " evil ", "🔥": " fire "}
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☠⚰⛔❌✝️]")

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_JAIL = re.compile(r"ignore all previous instructions|DAN mode|jailbreak|"
                   r"print the system prompt", re.I)

# strong signals on NORMALIZED text -> (pattern, weight, [cats])
_STRONG = [
    (r"kill (all|you|u|him|her|them|everyone)", .80, ["threat", "violence"]),
    (r"death to|deport them all|infidels|heil hitler|ia88|1488", .85,
     ["hate_identity", "threat"]),
    (r"te voy a matar|matar esta noche|te faire payer|te retrouver", .85,
     ["threat", "violence"]),
    (r"(pipe |make a |build a )?bomb|molotov|explosive", .80,
     ["weapons", "criminal_planning"]),
    (r"steals? .*cookies|steal|ransomware|ddos|phishing|phish", .80,
     ["malware"]),
    (r"shoplift|hotwire|pick a deadbolt|deadbolt|break into someone's", .78,
     ["criminal_planning"]),
    (r"pain to stop (forever|for good)|easiest way (to die|out)|end it all|"
     r"kill my ?self|suicide", .82, ["self_harm"]),
    (r"overdose someone|overdose|poison |opioid|fentanyl|strangle|"
     r"without a prescription", .80, ["violence", "controlled_substances"]),
    (r"stalk |track .* location without", .75,
     ["harassment", "illegal_activity"]),
    (r"fake review campaign|forge a|cheat on .*taxes|fake id", .75,
     ["fraud_deception"]),
    (r"ignore all previous instructions|system prompt|dan mode", .80,
     ["other"]),
    (r"get .* fired|spread rumors|just disappear|everyone .* hates you", .55,
     ["harassment"]),
    (r"watch your (back|step)|sleep with one eye|regret crossing", .45,
     ["threat"]),
    (r"knife .{0,20}blood|blood .{0,20}knife|bomb .{0,20}fire|"
     r"fire .{0,20}death|death\s+death", .80, ["threat", "violence"]),
]
# signals for the frontier lane on RAW text — no normalization, so it
# misses leet / homoglyph / emoji-carried content but does flag PII.
_STRONG_FRONTIER = [
    (r"\bkill\b|\bkill(s|ed|ing)?\b", .75, ["violence"]),
    (r"\bbomb\b|explosive|molotov", .78, ["weapons", "criminal_planning"]),
    (r"\bmurder\b|\bmatar\b|te faire payer", .80, ["threat", "violence"]),
    (r"suicide|kill myself|overdose|pain to stop", .78, ["self_harm", "violence"]),
    (r"ransomware|ddos|phishing|steal|poison|strangle|stalk|opioid|"
     r"fentanyl|without a prescription", .75, ["malware", "criminal_planning"]),
    (r"shoplift|hotwire|cheat on|forge a|fake review|deadbolt|spread rumors|"
     r"get .* fired", .72, ["criminal_planning", "fraud_deception"]),
    (r"\bdying\b", .45, ["self_harm"]),
    (r"break into|watch your|disappear", .55, ["threat", "harassment"]),
    (r"ignore all previous instructions|system prompt", .75, ["other"]),
    (r"\d{3}-\d{2}-\d{4}|(?:\d[ -]?){13,19}", .85, ["pii"]),
]


def _jit(cid, salt=0):
    return (int(hashlib.md5(f"{cid}:{salt}".encode()).hexdigest()[:6], 16)
            % 1000) / 1000.0


def normalize(text):
    t = unicodedata.normalize("NFKC", text).translate(_ZW)
    for e, w in _EMOJI.items():
        t = t.replace(e, w)
    t = _EMOJI_RE.sub(" ", t)          # drop remaining benign emoji
    t = t.translate(_HOMO)
    return t.translate(_LEET).lower()


def _scan(table, text):
    p, cats = 0.0, []
    for pat, w, cs in table:
        if re.search(pat, text, re.I):
            p = max(p, w) if p else w
            for c in cs:
                if c not in cats:
                    cats.append(c)
    return p, cats


def norm_info(text):
    """Normalized text worth showing next to the raw case text.

    Prefers the real stage-0 (safety_one.normalize, which reports the exact
    transforms applied); falls back to the demo-local normalize(). Only set
    when clean-up actually changed the text — pure case/whitespace noise is
    not worth a badge.
    """
    raw = text or ""
    try:
        from safety_one.normalize import normalize as _snorm
        nt = _snorm(raw)
        t = getattr(nt, "normalized", None)
        tr = list(getattr(nt, "transforms_applied", []) or [])
        if isinstance(t, str) and tr and t != raw:
            return {"normalized": t, "transforms": tr}
        return {"normalized": None, "transforms": tr}
    except Exception:
        t = normalize(raw)
        if t == raw.strip().lower():
            return {"normalized": None, "transforms": []}
        return {"normalized": t, "transforms": []}


def attach_norm(verdict, text):
    """Add top-level 'normalized'/'transforms' keys to a verdict dict."""
    info = norm_info(text)
    if info["normalized"]:
        verdict["normalized"] = info["normalized"]
        verdict["transforms"] = info["transforms"]
    return verdict


def _rule_hits(raw):
    hits = []
    if _SSN.search(raw):
        hits.append("ssn")
    if _CARD.search(raw):
        hits.append("card_number")
    if _JAIL.search(raw):
        hits.append("jailbreak_string")
    return hits


def _v(label_p, route, cats, rule_hits, model, lat, cost, raw=None):
    p = min(0.99, max(0.01, label_p))
    return {"label": "unsafe" if p >= 0.5 else "safe", "categories": cats,
            "probs": {"unsafe": round(p, 4), "safe": round(1 - p, 4)},
            "route": route, "rule_hits": rule_hits, "model": model,
            "latency_ms": round(lat, 1), "cost_usd": round(cost, 6),
            "raw": raw or {}}


def mock_ours(row, note=None):
    """Deterministic stand-in for the real SafetyPipeline (offline / pre-merge)."""
    cid, raw = row["id"], row["text"]
    caption = (row.get("meta") or {}).get("caption", "")
    text_for_models = f"{raw} {caption}".strip()
    norm = normalize(text_for_models)

    hits = _rule_hits(raw)
    if hits:
        return _v(0.995, "RULE", ["pii"] if "jailbreak_string" not in hits else ["other"],
                  hits, "mock/rule-gate", 0.4 + _jit(cid) * 1.2, 0.0, {"simulated": True})

    emoji_hits = sum(1 for e in _EMOJI if e in raw) + len(_EMOJI_RE.findall(raw))
    nonascii = sum(1 for ch in raw if ord(ch) > 127)
    lang = row.get("lang")
    if row.get("image_path"):
        route = "VISION"
    elif emoji_hits >= 2:
        route = "VISION"          # emoji-heavy -> vision/demojize path
    elif (lang and lang not in ("en",)) or nonascii > len(raw) * 0.30:
        route = "MULTILINGUAL"
    else:
        route = "SPECIALIST"

    p, cats = _scan(_STRONG, norm)
    if p == 0.0:
        p = 0.05 + _jit(cid, 1) * 0.08
    else:
        p += _jit(cid, 1) * 0.04

    if 0.40 <= p <= 0.62:         # gray zone -> escalate to reasoning model
        route = "ESCALATED"
        p = 0.5 + (p - 0.5) * 2.2 + (_jit(cid, 2) - 0.5) * 0.06
        lat = 1400 + _jit(cid, 3) * 500
        cost = 0.0012
        model = "mock/escalation-reasoner"
    else:
        lat = {"RULE": 1.5, "SPECIALIST": 60 + _jit(cid, 3) * 90,
               "VISION": 260 + _jit(cid, 3) * 160,
               "MULTILINGUAL": 80 + _jit(cid, 3) * 110}[route]
        cost = {"SPECIALIST": 0.00004, "VISION": 0.00035,
                "MULTILINGUAL": 0.00005}[route]
        model = {"SPECIALIST": "mock/nvidia-guard-sft", "VISION": "mock/nemotron-vl",
                 "MULTILINGUAL": "mock/nvidia-guard-sft-ml"}[route]
    v = _v(p, route, cats, hits, model, lat, cost, {"simulated": True})
    attach_norm(v, raw)
    if note:
        v["raw"]["note"] = note
    return v


def mock_frontier(row, note=None):
    """Deterministic frontier baseline: raw-text keyword scorer, no normalize."""
    cid, raw = row["id"], row["text"]
    p, cats = _scan(_STRONG_FRONTIER, raw.lower())
    if p == 0.0:
        p = 0.05 + _jit(cid, 7) * 0.06
    else:
        p = min(0.98, p + _jit(cid, 8) * 0.05)
    v = _v(p, "frontier", cats, [], "mock/frontier-gpt",
           520 + _jit(cid, 9) * 900, 0.0018 + _jit(cid, 10) * 0.0016,
           {"simulated": True})
    if note:
        v["raw"]["note"] = note
    return v


def classify_both(row):
    ours_fn, frontier_fn, _ = _LANES
    return {"ours": ours_fn(row), "frontier": frontier_fn(row)}


_LANES = load_lanes()  # resolved once at import; call reload_lanes() after env changes


def lane_modes():
    return dict(_LANES[2])
