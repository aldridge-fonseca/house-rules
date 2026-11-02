#!/usr/bin/env python3
"""Render eval text rows to PNG images (emoji A/B, render->vision route).

Turns rows of an eval JSONL (default: data/public/hatemojicheck.jsonl) into
chat-style PNGs so the same items can be sent to a vision-language model.
Emoji are drawn with an open-licensed emoji font; regular text with a system
sans font. Gold labels are carried through unchanged.

Fonts:
  emoji  data/vision/fonts/NotoEmoji-Variable.ttf (vendored, SIL OFL-1.1,
         ~2 MB monochrome). Pass --emoji-font path/to/NotoColorEmoji.ttf for
         realistic color renders (CBDT embedded_color auto-detected; the file
         is ~25 MB so it is not vendored — see data/VISION.md).
  text   DejaVu Sans (system); override with --text-font.

Stratified subset: proportional over (gold_label x functionality) using the
raw HatemojiCheck CSV when available, else gold_label only. Deterministic
(--seed). Output JSONL rows keep the eval schema; ids get an "-img" suffix and
meta.paired_id links back to the source row for the A/B join.

Usage:
  python3 scripts/render_emoji.py                     # 400-row hatemoji subset
  python3 scripts/render_emoji.py --n 0 --all         # render every row
  python3 scripts/render_emoji.py --input some.jsonl --out o.jsonl --outdir imgs/
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SAFETY_ONE = Path(__file__).resolve().parent.parent
VISION = SAFETY_ONE / "data" / "vision"

DEFAULT_INPUT = SAFETY_ONE / "data" / "public" / "hatemojicheck.jsonl"
DEFAULT_CSV = SAFETY_ONE / "data" / "public" / "raw" / "hatemoji" / "test.csv"
DEFAULT_FONT = VISION / "fonts" / "NotoEmoji-Variable.ttf"
DEFAULT_TEXT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TEXT_SIZE = 40
EMOJI_SIZE = 46          # drawn size; CBDT renders at strike then downscales
IMG_W = 900
PAD_X, PAD_Y = 32, 28
LINE_GAP = 14
BG = (255, 255, 255)
BUBBLE = (247, 247, 248)
BUBBLE_EDGE = (212, 212, 216)
INK = (17, 17, 20)


# --------------------------------------------------------------------------- #
# Emoji cluster segmentation (self-contained; stdlib only)
# --------------------------------------------------------------------------- #

def _is_emoji_base(cp: int) -> bool:
    # Ranges whose chars are emoji-presentation by default — safe to draw with
    # the emoji font unconditionally.
    return (0x1F000 <= cp <= 0x1FAFF or      # main emoji planes
            0x2600 <= cp <= 0x26FF or        # misc symbols (☀ ⚠ ❤ …)
            0x2700 <= cp <= 0x27BF)          # dingbats (✈ ✉ ✅ …)


def _is_emoji_weak(cp: int) -> bool:
    # Ambiguous ranges/singletons: only emoji when followed by VS16 or an emoji
    # modifier (e.g. → vs →️, ‼ vs ‼️, © vs ©️).
    if cp in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x24C2,
              0x3030, 0x303D, 0x3297, 0x3299):
        return True
    return (0x2190 <= cp <= 0x21FF or        # arrows
            0x2300 <= cp <= 0x23FF or        # ⌚ ⏰ ⏩ ⛔
            0x25A0 <= cp <= 0x25FF or        # geometric shapes
            0x2B00 <= cp <= 0x2BFF)          # stars, more arrows


def _is_modifier(cp: int) -> bool:
    return (cp in (0xFE0E, 0xFE0F, 0x20E3) or      # VS15/16, keycap mark
            0x1F3FB <= cp <= 0x1F3FF or            # skin tones
            0xE0020 <= cp <= 0xE007F)              # tag chars (flags)


def _is_ri(cp: int) -> bool:
    return 0x1F1E6 <= cp <= 0x1F1FF


def _is_keycap_base(ch: str) -> bool:
    return ch == "#" or ch == "*" or ch.isdigit()


def is_emoji_start(text: str, i: int) -> bool:
    ch, cp = text[i], ord(text[i])
    if _is_emoji_base(cp):
        return True
    if _is_keycap_base(ch) and i + 1 < len(text):
        nxt = ord(text[i + 1])
        if nxt == 0x20E3 or (nxt == 0xFE0F and i + 2 < len(text)
                             and ord(text[i + 2]) == 0x20E3):
            return True
    # Weak candidates need explicit emoji presentation (VS16 or modifier next)
    if _is_emoji_weak(cp) and i + 1 < len(text):
        nxt = ord(text[i + 1])
        if nxt == 0xFE0F or 0x1F3FB <= nxt <= 0x1F3FF:
            return True
    return False


def segment(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_emoji, cluster) atoms; preserves all characters."""
    atoms: list[tuple[bool, str]] = []
    buf = ""
    i, n = 0, len(text)

    def flush():
        nonlocal buf
        if buf:
            atoms.append((False, buf))
            buf = ""

    while i < n:
        if is_emoji_start(text, i):
            flush()
            j = i + 1
            while j < n:
                c2 = ord(text[j])
                if _is_modifier(c2):
                    j += 1
                elif c2 == 0x200D and j + 1 < n and is_emoji_start(text, j + 1):
                    j += 2
                else:
                    break
            cluster = text[i:j]
            # regional-indicator flags pair up
            if (_is_ri(ord(cluster[0])) and j < n
                    and _is_ri(ord(text[j]))):
                j += 1
                cluster = text[i:j]
            atoms.append((True, cluster))
            i = j
        else:
            buf += text[i]
            i += 1
    flush()
    return atoms


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

class EmojiFont:
    """Emoji drawing backend: monochrome TTF or CBDT color (embedded_color)."""

    def __init__(self, path: str, size: int):
        from PIL import Image, ImageDraw, ImageFont
        self.mono = ImageFont.truetype(path, size)
        self.size = size
        self.color = None
        # Probe for CBDT/color strikes: a supported strike renders the glyph
        # WITH chroma; wrong-size CBDT loads raise OSError, and non-color fonts
        # draw fine but stay grayscale.
        for strike in (109, 128, 136, size):
            try:
                f = ImageFont.truetype(path, strike)
                probe = Image.new("RGBA", (strike * 2, strike * 2))
                ImageDraw.Draw(probe).text((0, 0), "😀", font=f,
                                           embedded_color=True)
                px = list(probe.getdata())
                chroma = any(abs(r - g) > 30 or abs(g - b) > 30
                             for r, g, b, a in px if a > 100)
                if chroma:
                    self.color, self.color_size = f, strike
                break
            except Exception:
                continue

    def measure(self, cluster: str) -> int:
        try:
            w = self.mono.getlength(cluster)
            return int(w) if w > 1 else self.size
        except Exception:
            return self.size

    def draw(self, img, draw, pos, cluster, baseline_y):
        x = pos
        if self.color is not None:
            from PIL import Image, ImageDraw
            tmp = Image.new("RGBA", (self.color_size * 2, self.color_size * 2))
            ImageDraw.Draw(tmp).text((0, 0), cluster, font=self.color,
                                     embedded_color=True)
            bbox = tmp.getbbox()
            if bbox:
                tmp = tmp.crop(bbox)
            scale = self.size / max(tmp.width, 1)
            tmp = tmp.resize((max(1, int(tmp.width * scale)), self.size))
            img.paste(tmp, (x, int(baseline_y - self.size * 0.82)), tmp)
            return x + tmp.width + 4
        try:
            draw.text((x, baseline_y), cluster, font=self.mono,
                      fill=INK, anchor="ls")
        except Exception:
            draw.text((x, baseline_y), "\uFFFD" * len(cluster),
                      font=self.mono, fill=INK, anchor="ls")
        return x + self.measure(cluster) + 4


def render_text_image(text: str, emoji_font: "EmojiFont", out_path: Path,
                      text_font_path: str = DEFAULT_TEXT_FONT) -> tuple[int, int]:
    from PIL import Image, ImageDraw, ImageFont

    tf = ImageFont.truetype(text_font_path, TEXT_SIZE)
    meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    space_w = tf.getlength(" ")

    # Build drawable items: ("w", word) breakable at spaces; ("e", cluster).
    items: list[tuple[str, str]] = []
    for is_e, chunk in segment(text):
        if is_e:
            items.append(("e", chunk))
        else:
            for piece in re.split(r"(\s+)", chunk):
                if not piece:
                    continue
                items.append(("s", piece) if piece.isspace() else ("w", piece))

    def width(kind: str, s: str) -> float:
        return emoji_font.measure(s) + 4 if kind == "e" else tf.getlength(s)

    # Greedy wrap.
    max_w = IMG_W - 2 * PAD_X
    lines: list[list[tuple[str, str]]] = [[]]
    cur = 0.0
    pending_space = False
    for kind, s in items:
        if kind == "s":
            pending_space = True
            continue
        w = width(kind, s) + (space_w if pending_space and lines[-1] else 0)
        if lines[-1] and cur + w > max_w:
            lines.append([])
            cur = 0.0
            w = width(kind, s)
        elif pending_space and lines[-1]:
            cur += space_w
        lines[-1].append((kind, s))
        cur += w
        pending_space = False

    asc, desc = tf.getmetrics()
    line_h = max(asc + desc, int(EMOJI_SIZE * 1.25)) + LINE_GAP
    inner_h = len(lines) * line_h
    h = inner_h + 2 * PAD_Y + 2
    img = Image.new("RGB", (IMG_W, h), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 10, IMG_W - 10, h - 10], radius=24,
                        fill=BUBBLE, outline=BUBBLE_EDGE, width=2)

    y = PAD_Y
    for line in lines:
        x = PAD_X
        baseline = y + max(asc, int(EMOJI_SIZE * 0.82))
        for kind, s in line:
            if kind == "e":
                x = emoji_font.draw(img, d, x, s, baseline)
            else:
                d.text((x, baseline), s, font=tf, fill=INK, anchor="ls")
                x += tf.getlength(s) + space_w
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return img.size


# --------------------------------------------------------------------------- #
# Stratified sampling + JSONL emit
# --------------------------------------------------------------------------- #

def stratified(rows: list[dict], key_fn, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    total = len(rows)
    quota = {k: len(v) * n / total for k, v in groups.items()}
    take = {k: int(q) for k, q in quota.items()}
    remainder = max(0, n - sum(take.values()))
    for k in sorted(quota, key=lambda k: quota[k] - int(quota[k]),
                    reverse=True)[:remainder]:
        take[k] += 1
    out = []
    for k, v in groups.items():
        v = sorted(v, key=lambda r: r["id"])
        rng.shuffle(v)
        out.extend(v[: take[k]])
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="raw HatemojiCheck CSV for stratification columns")
    ap.add_argument("--out", default=str(VISION / "hatemoji_render_eval.jsonl"))
    ap.add_argument("--outdir", default=str(VISION / "images" / "hatemoji"))
    ap.add_argument("--emoji-font", default=str(DEFAULT_FONT))
    ap.add_argument("--text-font", default=DEFAULT_TEXT_FONT)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--id-suffix", default="-img")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]

    # Join raw CSV for functionality/target/set metadata (hatemoji-<case_id>).
    meta_by_id: dict[str, dict] = {}
    csv_p = Path(args.csv)
    if csv_p.exists():
        for r in csv.DictReader(open(csv_p)):
            meta_by_id[f"hatemoji-{r['case_id']}"] = {
                "functionality": r["functionality"], "set": r["set"],
                "target": r["target"],
            }

    if args.n and args.n < len(rows):
        rows = stratified(
            rows,
            lambda r: (r["gold_label"],
                       (meta_by_id.get(r["id"]) or {}).get("functionality", "?")),
            args.n, args.seed)
        print(f"[render_emoji] stratified sample: {len(rows)} rows "
              f"{Counter(r['gold_label'] for r in rows)}")

    efont = EmojiFont(args.emoji_font, EMOJI_SIZE)
    print(f"[render_emoji] font={Path(args.emoji_font).name} "
          f"color={'yes' if efont.color else 'no'}")

    outdir = Path(args.outdir)
    out_rows, fails = [], 0
    for r in rows:
        rid = r["id"] + args.id_suffix
        dst = outdir / f"{rid}.png"
        if not dst.exists():
            try:
                render_text_image(r["text"], efont, dst, args.text_font)
            except Exception as e:
                print(f"  render failed {r['id']}: {e}")
                fails += 1
                continue
        m = dict(meta_by_id.get(r["id"]) or {})
        m.update({"paired_id": r["id"], "render_side": "image",
                  "render_font": Path(args.emoji_font).name})
        nr = dict(r)
        nr["id"] = rid
        nr["image_path"] = str(dst.relative_to(SAFETY_ONE))
        nr["source"] = r["source"] + "-rendered"
        nr["meta"] = m
        out_rows.append(nr)

    with open(args.out, "w") as f:
        for nr in out_rows:
            f.write(json.dumps(nr, ensure_ascii=False) + "\n")
    print(f"[render_emoji] wrote {len(out_rows)} rows -> {args.out} "
          f"({fails} render failures)")


if __name__ == "__main__":
    main()
