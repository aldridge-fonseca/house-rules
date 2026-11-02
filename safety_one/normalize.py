"""Stage 0 — deterministic text normalization and de-obfuscation.

Pure stdlib plus the optional ``emoji`` package (a small built-in table is
used when it is not installed). Typical cost is well under 1 ms per message.

Pipeline: emoji descriptions -> script-aware NFKC/lookalike folding ->
invisible stripping -> payload decoding -> leetspeak -> spacing cleanup.
Both the raw and the normalized text are kept for downstream stages.
"""

from __future__ import annotations

import base64
import codecs
import re
import unicodedata
from dataclasses import dataclass, field

try:
    import emoji as _emoji
except ImportError:  # pragma: no cover - exercised only without the package
    _emoji = None

__all__ = ["NormalizedText", "normalize"]


@dataclass
class NormalizedText:
    raw: str
    normalized: str
    transforms_applied: list[str] = field(default_factory=list)
    emoji_list: list[str] = field(default_factory=list)
    emoji_density: float = 0.0
    decoded_payloads: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# Emoji handling
# --------------------------------------------------------------------------

_FALLBACK_EMOJI = {
    "😀": "grinning_face", "😂": "joy", "🙂": "slightly_smiling_face",
    "😍": "heart_eyes", "😘": "kissing_heart", "😭": "sob", "😡": "rage",
    "🤬": "face_with_symbols_on_mouth", "🖕": "middle_finger", "💀": "skull",
    "🔥": "fire", "❤️": "red_heart", "💔": "broken_heart", "👍": "thumbs_up",
    "💥": "explosion", "❄️": "snowflake", "🌿": "herb", "🚬": "cigarette",
    "🔞": "no_one_under_eighteen", "☠️": "skull_and_crossbones", "💻": "computer",
    "🦠": "microbe", "⛓️": "chains", "🚓": "police_car", "👮": "police_officer",
    "🥊": "boxing_glove", "🎣": "fishing_pole", "🗡️": "sword", "👩": "woman",
    "👨": "man", "👧": "girl", "👦": "boy", "🧒": "child", "👶": "baby",
    "👥": "people", "🚗": "car", "🏠": "house", "🏫": "school", "📖": "book",
    "📱": "phone", "📷": "camera", "📸": "photo", "🍺": "beer", "🍷": "wine",
    "💋": "kiss", "😢": "cry", "😊": "happy", "💍": "wedding", "🐈": "cat",
    "🐕": "dog", "👋": "waving_hand", "🦵": "leg",
    "👎": "thumbs_down", "🙏": "pray", "👏": "clap", "🎉": "tada",
    "💣": "bomb", "🔪": "knife", "🔫": "gun", "💊": "pill", "💉": "syringe",
    "🩸": "drop_of_blood", "😴": "sleeping", "🤮": "vomit", "💩": "poop",
    "👊": "fist", "🖤": "black_heart", "🍆": "eggplant", "🍑": "peach",
    "💦": "sweat_drops", "👅": "tongue", "🤑": "money_mouth", "💰": "moneybag",
    "⚠️": "warning", "🚫": "no_entry", "✅": "check_mark", "❌": "x",
}


def _extract_emoji(text: str) -> tuple[list[str], str, int]:
    """Return (emoji names, demojized text, emoji codepoint count)."""
    names: list[str] = []
    n_chars = 0

    for ch, name in _FALLBACK_EMOJI.items():
        if ch in text:
            names.append(name)
            n_chars += len(ch) * text.count(ch)
            text = text.replace(ch, f" :{name}: ")
    if _emoji is not None:
        matches = _emoji.emoji_list(text)
        names.extend(_emoji.demojize(m["emoji"]).strip(":") for m in matches)
        n_chars += sum(len(m["emoji"]) for m in matches)
        text = _emoji.demojize(text)
    return names, text, n_chars


# --------------------------------------------------------------------------
# Invisible / zero-width characters
# --------------------------------------------------------------------------

def _is_invisible(ch: str) -> bool:
    if unicodedata.category(ch) == "Cf":  # ZWSP, ZWNJ, ZWJ, soft hyphen, BOM...
        return True
    cp = ord(ch)
    return 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF  # selectors


# --------------------------------------------------------------------------
# Lookalike folding: Cyrillic/Greek homoglyphs + smart punctuation
# --------------------------------------------------------------------------

_LOOKALIKE = str.maketrans({
    # Cyrillic lowercase
    "а": "a", "с": "c", "е": "e", "і": "i", "ј": "j", "о": "o",
    "р": "p", "ѕ": "s", "х": "x", "у": "y", "һ": "h", "ԁ": "d",
    "ӏ": "l", "ԛ": "q", "ԍ": "g", "қ": "k", "м": "m", "н": "h", "т": "t",
    # Cyrillic uppercase
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I",
    "Ј": "J", "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S",
    "Т": "T", "Х": "X", "У": "Y", "Һ": "H",
    # Greek lowercase
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "μ": "m",
    "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ω": "w",
    # Greek uppercase
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
    # Smart punctuation -> ASCII
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "‐": "-", "⋅": ".", "・": ".",
})


# --------------------------------------------------------------------------
# Leetspeak (conservative)
# --------------------------------------------------------------------------

# Only digits/symbols that almost always stand for a letter. Deliberately
# excludes ambiguous chars (2, 6, @, +, |) so emails/numbers survive.
_LEET_TRANS = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "8": "b", "9": "g", "!": "i", "$": "s",
})
_LEET_CHARS = frozenset("01345789!$")


# Small explicit vocabulary resolves the irreversible 1 -> i/l ambiguity.
# No row-id or full-message lookup; unknown candidates retain a conservative fallback.
_LEET_WORDS = frozenset("""
black white legal illegal adult adults small large old older young family families
male female male female men women man woman boy boys girl girls assault sexual
religion religious muslim muslims jew jewish jews islam christian christians
explosives police plan plans planning look looking feel feeling should would could
believe possible explain details detail useful harmful harmless health healthy
normal example examples specific exactly question answer email mail gmail learn
learning valuable value available online local global personal financial social
public private general individual special professional political biological
wrong what drugs safest thanks this has was yes subtle capital donald quickly illegal all will well tell still kill killed killing illegal little really people please help
hello world love like life live lives child children girl girls school cool call
called also only always false full following instructions previous ignore learn
build building bomb bombs drug drugs poison gun guns hate hacker hack attack
explosive steal sexual assault violent violence weapon weapons suicide self
""".split())

def _deleet_token(tok: str, dense: bool = False) -> str:
    # Restrict to word cores, preserving punctuation, versions, URLs and emails.
    if len(tok) < 3 or not re.fullmatch(r"[A-Za-z01345789!$]+", tok):
        return tok
    if (not dense and not any(c.isalpha() for c in tok)) or not any(c in _LEET_CHARS for c in tok):
        return tok
    if re.fullmatch(r"[A-Za-z]+[0-9]+", tok) and tok.translate(_LEET_TRANS).lower() not in _LEET_WORDS:
        return tok  # identifiers such as B1, mp3, GPT4
    if re.fullmatch(r"\d+(?:st|nd|rd|th|mg|kg|ml|am|pm)", tok, re.I):
        return tok
    candidate = tok.translate(_LEET_TRANS).replace("9", "g")
    if "1" in tok:
        # Resolve only unambiguous dictionary candidates; cap combinations.
        options = [""]
        for c in tok:
            letters = "il" if c == "1" else c.translate(_LEET_TRANS).replace("9", "g")
            options = [p + x for p in options for x in letters][:256]
        known = [w for w in options if w.lower() in _LEET_WORDS]
        if len(known) == 1:
            candidate = known[0]
    if not any(c.isalpha() for c in tok) and candidate.lower() not in _LEET_WORDS:
        return tok
    return candidate


def _deleet(text: str) -> tuple[str, int]:
    changed = 0
    # Dense substitution is needed before resolving all-digit words like 70.
    cores = re.findall(r"[A-Za-z0-9!$]+", text)
    suspicious = [w for w in cores if any(c.isalpha() for c in w) and any(c in _LEET_CHARS for c in w.rstrip("!$"))]
    dense = len(suspicious) >= 3 and len(suspicious) / max(1, len(cores)) > .25
    def replace(m):
        nonlocal changed
        tok = m.group()
        if "@" in tok or "://" in tok or re.search(r"\d\.\d", tok):
            return tok
        def word(w):
            core = w.group()
            if dense and core in {"70", "15", "4", "1", "!"}:
                return {"70":"to", "15":"is", "4":"a", "1":"I", "!":"I"}[core]
            if dense and re.fullmatch(r"[01345789!$]{3,}", core):
                candidate = core.translate(_LEET_TRANS)
                if candidate in _LEET_WORDS:
                    return candidate
            return _deleet_token(core, dense=dense)
        new = re.sub(r"[A-Za-z0-9]+(?:[!$][A-Za-z0-9]+)*|[!$][A-Za-z0-9]+|(?<!\S)!(?!\S)", word, tok)
        changed += new != tok
        return new
    return re.sub(r"\S+", replace, text), changed


# --------------------------------------------------------------------------
# Spacing tricks: "k i l l" -> "kill", "k.i.l.l" -> "kill"
# --------------------------------------------------------------------------

# Runs of single chars separated by single spaces. The lookarounds keep the
# run aligned to whole tokens so "is a bad" is not collapsed to "isabad".
_SPACE_RUN = re.compile(r"(?<!\w)[A-Za-z](?: [A-Za-z]){2,}(?!\w)")
_SEP_RUN = re.compile(r"\b[A-Za-z](?:[.\-*·_/\\][A-Za-z]){2,}\b")
_SEP_CHARS = re.compile(r"[.\-*·_/\\]")


def _despacify(text: str) -> tuple[str, int]:
    n = 0

    def join_space(m: re.Match) -> str:
        nonlocal n
        n += 1
        parts = m.group(0).split(" ")
        if all(p.isupper() for p in parts):
            return m.group(0)  # initials / acronyms

        prefix = parts.pop(0) + " " if parts[0] in ("I", "a", "A") else ""
        suffix = " " + parts.pop() if parts and parts[-1] in ("I", "a", "A") else ""
        return prefix + "".join(parts) + suffix

    def join_sep(m: re.Match) -> str:
        nonlocal n
        n += 1
        return _SEP_CHARS.sub("", m.group(0))

    text = _SEP_RUN.sub(join_sep, text)
    text = _SPACE_RUN.sub(join_space, text)
    return text, n


# --------------------------------------------------------------------------
# Encoded payloads: base64, hex, rot13
# --------------------------------------------------------------------------

_B64_TOKEN = re.compile(r"(?<![\w])[A-Za-z0-9+/]{8,}={0,2}(?![\w])")
_HEX_TOKEN = re.compile(r"(?<![\w])(?:0[xX])?[0-9a-fA-F]{8,}(?![\w])")
_HEX_ESCAPED = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")
_WORD3 = re.compile(r"[a-zA-Z]{3,}")

# Small hint list: only used to decide whether a rot13 decode is plausible.
_ROT13_HINTS = frozenset({
    "the", "and", "you", "are", "all", "how", "what", "your", "can", "not",
    "this", "that", "with", "have", "will", "from", "they", "make", "tell",
    "give", "kill", "bomb", "hack", "hate", "die", "attack", "weapon",
    "drugs", "password", "ignore", "instructions", "system", "prompt",
    "secret", "explosive", "poison", "steal", "credit", "card", "nudes",
    "suicide", "humans", "world",
})


def _plausible_text(s: str) -> bool:
    if not 4 <= len(s) <= 16000:
        return False
    printable = sum(1 for c in s if c in " \t\r\n" or 33 <= ord(c) <= 126)
    if printable < len(s) * 0.9:
        return False
    return any(c.isalpha() for c in s)


def _try_b64(tok: str) -> str | None:
    padded = tok + "=" * (-len(tok) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
        s = raw.decode("utf-8")
    except Exception:
        return None
    return s if _plausible_text(s) and (_dict_hits(s) or re.search(r"\s", s)) else None


def _try_hex(tok: str) -> str | None:
    t = tok[2:] if tok[:2].lower() == "0x" else tok
    if len(t) % 2:
        return None
    try:
        raw = bytes.fromhex(t)
        s = raw.decode("utf-8")
    except Exception:
        return None
    return s if _plausible_text(s) else None


def _try_hex_escaped(tok: str) -> str | None:
    try:
        raw = bytes(int(h, 16) for h in re.findall(r"[0-9a-fA-F]{2}", tok))
        s = raw.decode("utf-8")
    except Exception:
        return None
    return s if _plausible_text(s) else None


def _dict_hits(text: str) -> int:
    return sum(1 for w in _WORD3.findall(text.lower()) if w in _ROT13_HINTS)


def _decode_payloads(text: str) -> tuple[list[dict], list[str]]:
    payloads: list[dict] = []
    decoded: list[str] = []

    for m in _B64_TOKEN.finditer(text):
        s = _try_b64(m.group(0))
        if s:
            payloads.append(
                {"encoding": "base64", "source": m.group(0)[:120], "text": s})
            decoded.append(s)

    for m in _HEX_ESCAPED.finditer(text):
        s = _try_hex_escaped(m.group(0))
        if s:
            payloads.append(
                {"encoding": "hex", "source": m.group(0)[:120], "text": s})
            decoded.append(s)

    for m in _HEX_TOKEN.finditer(text):
        s = _try_hex(m.group(0))
        if s:
            payloads.append(
                {"encoding": "hex", "source": m.group(0)[:120], "text": s})
            decoded.append(s)

    # rot13: only when decoding produces clearly more English than the input.
    rot = codecs.decode(text, "rot_13")
    if _dict_hits(rot) >= 2 and _dict_hits(rot) > _dict_hits(text):
        payloads.append(
            {"encoding": "rot13", "source": text[:120], "text": rot[:500]})
        decoded.append(rot[:500])

    return payloads, decoded


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def normalize(text, *, _depth: int = 0) -> NormalizedText:
    """Normalize ``text``; returns NormalizedText with raw + normalized kept.

    Accepts a plain string or any object with a ``.text`` attribute
    (e.g. types.Message)."""
    if not isinstance(text, str):
        # Message-like objects: use .text (empty/None -> ""), never the repr.
        text = (text.text or "") if hasattr(text, "text") else ("" if text is None else str(text))
    raw = text
    transforms: list[str] = []

    emoji_names, work, n_emoji_chars = _extract_emoji(raw)
    if emoji_names:
        transforms.append("emoji_demojize")

    native_script = sum(c.isalpha() and ord(c) > 0x024f and not c.translate(_LOOKALIKE).isascii() for c in work) >= 3
    new = unicodedata.normalize("NFKC", work) if not native_script else "".join(
        unicodedata.normalize("NFKC", c) if c.isascii() or "FULLWIDTH LATIN" in unicodedata.name(c, "") else c for c in work
    )
    if new != work:
        transforms.append("nfkc")
    work = new

    # Fold only mixed tokens, or confusable-only words in a Latin context.
    # Native Greek/Cyrillic sentences must not be transliterated.
    letters = [c for c in work if c.isalpha()]
    latin_context = bool(letters) and (
        sum(c.isascii() for c in letters) / len(letters) > .55 or (
            sum(c.isascii() for c in letters) / len(letters) > .15 and
            all(c.translate(_LOOKALIKE).isascii() for c in letters)
        )
    )
    mixed_tokens = sum(bool(re.search(r"[A-Za-z]", w)) and any(ord(c) > 127 for c in w) for w in re.findall(r"[^\W\d_]+", work))
    def fold(m):
        token = m.group()
        mapped = token.translate(_LOOKALIKE)
        if len(token) == 1 and not token.isascii() and mixed_tokens < 2:
            return token  # standalone scientific Greek letters / native words
        if all(c.isascii() for c in mapped) and (
            any(c.isascii() for c in token) or latin_context
        ):
            return mapped
        return token
    new = re.sub(r"[^\W\d_]+", fold, work)
    new = "".join(c.translate(_LOOKALIKE) if not c.isalpha() else c for c in new)
    if new != work:
        transforms.append("homoglyph")
    work = new

    def strip_invisible(i, c):
        if not _is_invisible(c):
            return False
        if c in ("\u200c", "\u200d"):
            left = work[:i].rstrip("\u200c\u200d")[-1:]
            right = work[i+1:].lstrip("\u200c\u200d")[:1]
            return bool(left and right and all(x.isascii() and x.isalnum() for x in left + right))
        return True
    new = "".join(c for i, c in enumerate(work) if not strip_invisible(i, c))
    if new != work:
        transforms.append("invisible_strip")
    work = new

    # Payload decode runs before leet/despacing so encoded tokens are not
    # mangled; decoded text replaces its encoding and gets the cleanup pass.
    payloads, decoded = _decode_payloads(work)
    for p in payloads:
        tag = p["encoding"] + "_decode"
        if tag not in transforms:
            transforms.append(tag)
    if decoded:
        # Replace lossless encodings inline, rather than contaminating provider
        # input with both the encoded and decoded copies. Keep receipts + raw.
        for p in payloads:
            if p["encoding"] == "rot13":
                work = p["text"]
            else:
                pattern = _B64_TOKEN if p["encoding"] == "base64" else (_HEX_ESCAPED if "\\x" in p["source"] else _HEX_TOKEN)
                work = pattern.sub(lambda m: p["text"] if m.group()[:120] == p["source"] else m.group(), work)
        # A labeled encoding wrapper is transport metadata, not payload.
        if len(payloads) == 1 and payloads[0]["encoding"] == "base64":
            prefix, sep, tail = work.partition(payloads[0]["text"])
            if sep and not tail.strip() and (not prefix.strip() or re.fullmatch(
                r"(?is)(?:base64:\s*|(?:decode|respond|the next user message)[^\n]*base64[^\n]*(?:[:.]|says\.)\s*)", prefix
            )):
                work = payloads[0]["text"]

    work, n = _deleet(work)
    if n:
        transforms.append("leetspeak")

    work, n = _despacify(work)
    if n:
        transforms.append("spacing")

    # Preserve normal paragraph/word boundaries. Only emoji expansion inserts
    # synthetic whitespace; other cleanup must not flatten ordinary prose.
    if emoji_names:
        work = re.sub(r" +", " ", work).strip()

    if decoded and _depth < 2:
        inner = normalize(work, _depth=_depth + 1)
        work = inner.normalized
        transforms.extend(t for t in inner.transforms_applied if t not in transforms)
        emoji_names.extend(n for n in inner.emoji_list if n not in emoji_names)

    density = (n_emoji_chars / len(raw)) if raw else 0.0
    return NormalizedText(
        raw=raw,
        normalized=work,
        transforms_applied=transforms,
        emoji_list=emoji_names,
        emoji_density=density,
        decoded_payloads=payloads,
    )
