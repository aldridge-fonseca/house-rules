#!/usr/bin/env python3
"""Regenerate 1920×1080 demo-video chart PNGs from LIVE receipts.

Usage (from repo root):
  .venv/bin/python docs/video/make_charts.py

Reads (never writes outside docs/video/):
  results/nvidia-v1/ours_summary.json
  results/nvidia-v1/{final,nano-untuned,ultra-untuned}.jsonl
  results/nvidia-v1-frontier/frontier-router-{a,b}.jsonl
  docs/img/architecture.svg

Writes:
  docs/video/quality_vs_cost.png
  docs/video/monthly_bill.png
  docs/video/latency.png
  docs/video/title_card.png
  docs/video/architecture_card.png
  docs/video/numbers.json
  docs/video/numbers.md
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OURS_SUMMARY = ROOT / "results" / "nvidia-v1" / "ours_summary.json"
FRONTIER_DIR = ROOT / "results" / "nvidia-v1-frontier"
OURS_JSONL = {
    "final": ROOT / "results" / "nvidia-v1" / "final.jsonl",
    "nano-untuned": ROOT / "results" / "nvidia-v1" / "nano-untuned.jsonl",
    "ultra-untuned": ROOT / "results" / "nvidia-v1" / "ultra-untuned.jsonl",
}
ARCH_SVG = ROOT / "docs" / "img" / "architecture.svg"

W, H = 1920, 1080
BG = "#FAFAF8"
INK = "#1A1A1A"
MUTED = "#5A5A5A"
GRID = "#D8D8D4"
ACCENT = "#76B900"  # NVIDIA green — ours highlight
BOOTSTRAP_N = 2000
SEED = 7

# Okabe–Ito-ish, colorblind-safe; ours stays brand green.
COLORS = {
    "final": ACCENT,
    "nano-untuned": "#0072B2",
    "ultra-untuned": "#E69F00",
    "frontier-router-a": "#CC79A7",
    "frontier-router-b": "#D55E00",
}
SHORT = {
    "final": "House Rules",
    "nano-untuned": "Nano untuned",
    "ultra-untuned": "Ultra 550B",
    "frontier-router-a": "GPT-6.1 Sol",
    "frontier-router-b": "Opus 5.5",
}
FULL = {
    "final": "House Rules (tuned Nano pipeline)",
    "nano-untuned": "Nemotron Nano, untuned",
    "ultra-untuned": "Nemotron Ultra 550B",
    "frontier-router-a": "GPT-6.1 Sol",
    "frontier-router-b": "Claude Opus 5.5",
}
ORDER = [
    "final",
    "nano-untuned",
    "ultra-untuned",
    "frontier-router-a",
    "frontier-router-b",
]

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), hex_rgb(BG))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 10], fill=hex_rgb(ACCENT))
    return img, draw


def footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.text((64, H - 48), text, font=font(22), fill=hex_rgb(MUTED))


def title(draw: ImageDraw.ImageDraw, text: str, y: int = 48) -> None:
    draw.text((64, y), text, font=font(48, bold=True), fill=hex_rgb(INK))


# ------------------------------------------------------------------ scoring


def is_nvidia_test(row: dict) -> bool:
    return str(row.get("id", "")).startswith("nv-test")


def load_jsonl(path: Path, keep) -> list[dict]:
    last: dict[str, dict] = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if keep(r):
                last[r["id"]] = r
    return list(last.values())


def pred_label(row: dict) -> str | None:
    return (row.get("prediction") or {}).get("label")


def is_refusal_or_error(row: dict) -> bool:
    lab = pred_label(row)
    return bool(row.get("error")) or lab not in ("safe", "unsafe")


def scored_pairs(rows: list[dict]) -> list[tuple[str, str, bool]]:
    """(gold, pred, refused). Refusal/error → flipped gold (counts as wrong)."""
    flip = {"safe": "unsafe", "unsafe": "safe"}
    out: list[tuple[str, str, bool]] = []
    for r in rows:
        gold = r["gold_label"]
        if is_refusal_or_error(r):
            out.append((gold, flip[gold], True))
        else:
            out.append((gold, pred_label(r), False))
    return out


def macro_f1(pairs: list[tuple[str, str, bool]]) -> float:
    f1s = []
    for c in ("safe", "unsafe"):
        tp = sum(g == c and p == c for g, p, _ in pairs)
        fp = sum(g != c and p == c for g, p, _ in pairs)
        fn = sum(g == c and p != c for g, p, _ in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return 50.0 * sum(f1s)


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    a = sorted(vals)
    k = (len(a) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return a[int(k)]
    return a[f] * (c - k) + a[c] * (k - f)


def bootstrap_f1_ci(pairs: list[tuple[str, str, bool]], n: int = BOOTSTRAP_N, seed: int = SEED):
    if len(pairs) < 2:
        return None, None
    rng = random.Random(seed)
    scores = []
    m = len(pairs)
    for _ in range(n):
        sample = [pairs[rng.randrange(m)] for _ in range(m)]
        scores.append(macro_f1(sample))
    scores.sort()
    lo = scores[int(0.025 * (n - 1))]
    hi = scores[int(0.975 * (n - 1))]
    return lo, hi


def ledger_costs(path: Path = ROOT / "results" / "spend.json") -> dict:
    """(model, row_id) -> last settled frontier call cost. Refusal receipts store
    $0 — their billed cost survives only in this ledger."""
    out = {}
    if path.exists():
        for e in json.loads(path.read_text(encoding="utf-8")).get("entries", []):
            md = e.get("metadata") or {}
            if e.get("provider") == "frontier-router" and md.get("row_id"):
                out[(md.get("model"), md["row_id"])] = float(e.get("cost_usd") or 0.0)
    return out


def score_frontier(rows: list[dict]) -> dict:
    """Refusal = wrong. $/1M = (verdict receipt costs + ledger-billed refusal
    costs) ÷ all rows — same convention as stats_vs_frontier.py /
    final_headline.py / baselines/frontier_summary.py."""
    pairs = scored_pairs(rows)
    ledger = ledger_costs()
    models = sorted({str((r.get("prediction") or {}).get("model"))
                     for r in rows if not r.get("error")})
    model = models[0] if len(models) == 1 else None
    lat, cost, refused = [], 0.0, 0
    for r, (_g, _p, ref) in zip(rows, pairs):
        pred = r.get("prediction") or {}
        rc = float(pred.get("cost_usd") or 0.0)
        if ref:
            refused += 1
            if model:  # refusal rows are billed per the spend ledger
                rc = ledger.get((model, str(r["id"])), rc)
        else:
            lat.append(float(pred.get("latency_ms") or 0.0))
        cost += rc
    n = len(pairs)
    return {
        "n": n,
        "macro_f1": round(macro_f1(pairs), 1),
        "f1_ci95": None,  # filled below
        "refused_pct": round(100.0 * refused / n, 1) if n else 0.0,
        "p50_ms": round(percentile(lat, 50) or 0.0, 1),
        "p95_ms": round(percentile(lat, 95) or 0.0, 1),
        "usd_per_1m_msgs": round(1e6 * cost / n, 1) if n else None,
        "partial": n < 3340,
        "source": None,
    }


def collect_systems() -> dict[str, dict]:
    summary = json.loads(OURS_SUMMARY.read_text(encoding="utf-8"))
    systems: dict[str, dict] = {}

    for key in ("final", "nano-untuned", "ultra-untuned"):
        m = summary["systems"][key]["nvidia_test"]
        rows = load_jsonl(OURS_JSONL[key], is_nvidia_test)
        pairs = scored_pairs(rows)
        lo, hi = bootstrap_f1_ci(pairs)
        systems[key] = {
            "label": FULL[key],
            "short": SHORT[key],
            "n": m["n"],
            "macro_f1": float(m["macro_f1"]),
            "f1_ci95": [round(lo, 1), round(hi, 1)] if lo is not None else None,
            "refused_pct": 0.0,
            "p50_ms": round(float(m["p50_ms"]), 1),
            "p95_ms": round(float(m["p95_ms"]), 1),
            "usd_per_1m_msgs": float(m["usd_per_1m_msgs"]),
            "partial": False,
            "source": "results/nvidia-v1/ours_summary.json → systems.{}.nvidia_test".format(key),
            "ci_source": f"bootstrap macro-F1 on {OURS_JSONL[key].relative_to(ROOT)} (nv-test ids; refusal=wrong)",
            "highlight": key == "final",
            "cost_note": "GPU rental amortized at full utilization" if key != "ultra-untuned" else "serverless per-token",
        }
        # Ultra is serverless (gpu_alloc=0); Nano/final use GPU alloc.
        if key == "ultra-untuned":
            systems[key]["cost_note"] = "serverless per-token (receipts)"
        elif key == "final":
            systems[key]["cost_note"] = "GPU rental amortized at full utilization (+ Ultra escalate tokens)"
        else:
            systems[key]["cost_note"] = "GPU rental amortized at full utilization"

    for key, fname in (
        ("frontier-router-a", "frontier-router-a.jsonl"),
        ("frontier-router-b", "frontier-router-b.jsonl"),
    ):
        path = FRONTIER_DIR / fname
        rows = load_jsonl(path, is_nvidia_test)
        m = score_frontier(rows)
        pairs = scored_pairs(rows)
        lo, hi = bootstrap_f1_ci(pairs)
        m["f1_ci95"] = [round(lo, 1), round(hi, 1)] if lo is not None else None
        m["label"] = FULL[key]
        m["short"] = SHORT[key]
        m["source"] = f"results/nvidia-v1-frontier/{fname} (refusal=wrong; $/1M = verdicts + ledger refusals ÷ n)"
        m["ci_source"] = f"bootstrap macro-F1 on {path.relative_to(ROOT)}"
        m["highlight"] = False
        m["cost_note"] = "measured frontier token spend"
        systems[key] = m

    return systems


# ------------------------------------------------------------------ charts


def draw_quality_vs_cost(systems: dict[str, dict]) -> Path:
    img, draw = new_canvas()
    title(draw, "Quality vs cost — NVIDIA-policy held-out test")
    draw.text(
        (64, 110),
        "Macro-F1 (refusals = wrong)  ·  $ per 1M messages (log scale)",
        font=font(28),
        fill=hex_rgb(MUTED),
    )

    # Plot area
    left, right, top, bottom = 160, W - 120, 200, H - 140
    costs = [systems[k]["usd_per_1m_msgs"] for k in ORDER if systems[k]["n"]]
    f1s = [systems[k]["macro_f1"] for k in ORDER if systems[k]["n"]]
    xmin, xmax = min(costs) * 0.7, max(costs) * 1.35
    ymin, ymax = 55.0, 100.0

    def x_of(c: float) -> float:
        return left + (math.log10(c) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (right - left)

    def y_of(f: float) -> float:
        return bottom - (f - ymin) / (ymax - ymin) * (bottom - top)

    # Axes + grid
    draw.rectangle([left, top, right, bottom], outline=hex_rgb(INK), width=2)
    for f in range(60, 101, 10):
        y = y_of(f)
        draw.line([(left, y), (right, y)], fill=hex_rgb(GRID), width=1)
        draw.text((left - 70, y - 14), f"{f}", font=font(24), fill=hex_rgb(MUTED))
    # log tick candidates
    ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    for t in ticks:
        if xmin <= t <= xmax:
            x = x_of(t)
            draw.line([(x, top), (x, bottom)], fill=hex_rgb(GRID), width=1)
            label = f"${t:,.0f}" if t < 1000 else f"${t/1000:.0f}k"
            draw.text((x - 28, bottom + 12), label, font=font(22), fill=hex_rgb(MUTED))

    draw.text((left - 10, top - 48), "Macro-F1", font=font(24, bold=True), fill=hex_rgb(INK))
    draw.text((right - 260, bottom + 48), "$ / 1M messages →", font=font(24, bold=True), fill=hex_rgb(INK))

    # Points
    for key in ORDER:
        s = systems[key]
        if not s["n"]:
            continue
        x, y = x_of(s["usd_per_1m_msgs"]), y_of(s["macro_f1"])
        color = hex_rgb(COLORS[key])
        # CI bars
        if s.get("f1_ci95"):
            lo, hi = s["f1_ci95"]
            draw.line([(x, y_of(lo)), (x, y_of(hi))], fill=color, width=3)
            draw.line([(x - 8, y_of(lo)), (x + 8, y_of(lo))], fill=color, width=3)
            draw.line([(x - 8, y_of(hi)), (x + 8, y_of(hi))], fill=color, width=3)
        r = 18 if s.get("highlight") else 12
        if s.get("highlight"):
            draw.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], outline=color, width=4)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        # label offset to avoid collisions
        lx, ly = x + 22, y - 28
        if key == "nano-untuned":
            lx, ly = x + 22, y + 8
        elif key == "ultra-untuned":
            lx, ly = x + 22, y - 40
        elif key == "frontier-router-a":
            lx, ly = x - 160, y - 36
        elif key == "frontier-router-b":
            lx, ly = x - 140, y - 36
        tag = f'{s["short"]}  {s["macro_f1"]:.1f}'
        draw.text((lx, ly), tag, font=font(26, bold=s.get("highlight")), fill=color)

    opus_n = systems["frontier-router-b"]["n"]
    foot = (
        f"Ours = GPU rental amortized at full utilization. Refusals scored as wrong. "
        f"95% bootstrap CI on macro-F1. Opus n={opus_n}/3,340 (partial — re-run after finish)."
    )
    footer(draw, foot)
    path = OUT / "quality_vs_cost.png"
    img.save(path, "PNG")
    return path


def money(v: float) -> str:
    if v >= 1000:
        return f"${v:,.0f}"
    if v >= 100:
        return f"${v:.0f}"
    return f"${v:.1f}"


def draw_monthly_bill(systems: dict[str, dict]) -> Path:
    img, draw = new_canvas()
    title(draw, "Monthly bill at 100M messages")
    draw.text(
        (64, 110),
        "USD / month = ($ / 1M messages) × 100   ·   same NVIDIA-policy receipts",
        font=font(28),
        fill=hex_rgb(MUTED),
    )

    left, right, top, bottom = 280, W - 100, 200, H - 140
    bills = {k: systems[k]["usd_per_1m_msgs"] * 100 for k in ORDER if systems[k]["n"]}
    vmax = max(bills.values()) * 1.08

    n = len(bills)
    gap = 36
    bar_h = ((bottom - top) - gap * (n - 1)) / n
    y = top
    for key in ORDER:
        if key not in bills:
            continue
        s = systems[key]
        bill = bills[key]
        bw = max(12.0, (right - left) * (bill / vmax))  # min width so tiny bills stay visible
        color = hex_rgb(COLORS[key])
        y1 = y + bar_h
        draw.rounded_rectangle([left, y, left + bw, y1], radius=8, fill=color)
        if s.get("highlight"):
            draw.rounded_rectangle([left, y, left + bw, y1], radius=8, outline=hex_rgb(INK), width=4)
        draw.text((64, y + bar_h / 2 - 16), s["short"], font=font(30, bold=s.get("highlight")), fill=hex_rgb(INK))
        label = money(bill) + (" / mo" if bill < 1e6 else "")
        # put label inside bar if wide enough else outside
        tw = draw.textlength(label, font=font(30, bold=True))
        if bw > tw + 40:
            draw.text((left + bw - tw - 16, y + bar_h / 2 - 16), label, font=font(30, bold=True), fill=(255, 255, 255))
        else:
            draw.text((left + bw + 16, y + bar_h / 2 - 16), label, font=font(30, bold=True), fill=color)
        y = y1 + gap

    opus_n = systems["frontier-router-b"]["n"]
    footer(
        draw,
        f"Ours = GPU rental amortized at full utilization. Opus n={opus_n}/3,340 (partial). "
        f"House Rules ≈ {money(bills['final'])}/mo vs Opus ≈ {money(bills['frontier-router-b'])}/mo.",
    )
    path = OUT / "monthly_bill.png"
    img.save(path, "PNG")
    return path


def draw_latency(systems: dict[str, dict]) -> Path:
    img, draw = new_canvas()
    title(draw, "Latency — p50 and p95")
    draw.text(
        (64, 110),
        "Client-side wall ms per message (concurrency 48; includes escalation round-trips)",
        font=font(28),
        fill=hex_rgb(MUTED),
    )

    left, right, top, bottom = 280, W - 120, 200, H - 140
    vals = []
    for k in ORDER:
        if systems[k]["n"]:
            vals.extend([systems[k]["p50_ms"], systems[k]["p95_ms"]])
    vmax = max(vals) * 1.12

    n = sum(1 for k in ORDER if systems[k]["n"])
    gap = 28
    group_h = ((bottom - top) - gap * (n - 1)) / n
    bar_h = (group_h - 10) / 2
    y = top
    legend_x = right - 280
    draw.rectangle([legend_x, 110, legend_x + 28, 138], fill=hex_rgb("#4C4C4C"))
    draw.text((legend_x + 40, 112), "p50", font=font(24), fill=hex_rgb(INK))
    draw.rectangle([legend_x + 120, 110, legend_x + 148, 138], fill=hex_rgb("#9A9A9A"))
    draw.text((legend_x + 160, 112), "p95", font=font(24), fill=hex_rgb(INK))

    for key in ORDER:
        s = systems[key]
        if not s["n"]:
            continue
        color = hex_rgb(COLORS[key])
        draw.text((64, y + group_h / 2 - 16), s["short"], font=font(28, bold=s.get("highlight")), fill=hex_rgb(INK))
        for i, (metric, shade) in enumerate((("p50_ms", 0), ("p95_ms", 1))):
            v = s[metric]
            bw = (right - left) * (v / vmax)
            yy = y + i * (bar_h + 10)
            fill = color if shade == 0 else tuple(int(c * 0.55 + 255 * 0.45) for c in color)
            draw.rounded_rectangle([left, yy, left + max(bw, 4), yy + bar_h], radius=6, fill=fill)
            if s.get("highlight") and shade == 0:
                draw.rounded_rectangle([left, yy, left + max(bw, 4), yy + bar_h], radius=6, outline=hex_rgb(INK), width=3)
            lab = f"{v:,.0f} ms"
            draw.text((left + max(bw, 4) + 12, yy + bar_h / 2 - 14), lab, font=font(24), fill=hex_rgb(INK))
        y += group_h + gap

    opus_n = systems["frontier-router-b"]["n"]
    footer(draw, f"Opus n={opus_n}/3,340 (partial). House Rules p50 = {systems['final']['p50_ms']:.0f} ms.")
    path = OUT / "latency.png"
    img.save(path, "PNG")
    return path


def draw_title_card() -> Path:
    img, draw = new_canvas()
    # atmosphere: soft corner wash
    for i in range(0, 420, 2):
        shade = 250 - i // 20
        draw.ellipse([-200 + i, H - 500 + i // 2, 700 + i, H + 200], outline=(shade, shade - 2, 230))
    draw.rectangle([0, 0, W, 14], fill=hex_rgb(ACCENT))
    draw.text((120, 340), "House Rules", font=font(110, bold=True), fill=hex_rgb(INK))
    draw.rectangle([120, 480, 520, 492], fill=hex_rgb(ACCENT))
    draw.text(
        (120, 530),
        "Your safety policy, enforced.",
        font=font(54),
        fill=hex_rgb(MUTED),
    )
    draw.text(
        (120, 920),
        "Open NVIDIA models on Fireworks  ·  LIVE receipts",
        font=font(28),
        fill=hex_rgb(MUTED),
    )
    path = OUT / "title_card.png"
    img.save(path, "PNG")
    return path


def rasterize_svg(svg: Path, out_png: Path, size: tuple[int, int]) -> bool:
    chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    rsvg = shutil.which("rsvg-convert")
    out_png.unlink(missing_ok=True)
    if rsvg:
        subprocess.run(
            [rsvg, "-w", str(size[0]), "-h", str(size[1]), "-o", str(out_png), str(svg)],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if out_png.is_file() and out_png.stat().st_size > 0:
            return True
    if not chrome:
        return False
    with tempfile.TemporaryDirectory(prefix="hr-video-") as profile:
        # Wrap SVG in HTML sized to the frame so Chrome screenshots cleanly.
        html = Path(profile) / "arch.html"
        html.write_text(
            f"""<!doctype html><html><head><style>
html,body{{margin:0;background:#FAFAF8;width:{size[0]}px;height:{size[1]}px;overflow:hidden}}
.wrap{{display:flex;align-items:center;justify-content:center;width:100%;height:100%;padding:80px 60px 60px;box-sizing:border-box;flex-direction:column}}
h1{{font-family:DejaVu Sans,Arial,sans-serif;font-size:48px;margin:0 0 28px;color:#1A1A1A}}
img{{max-width:100%;max-height:820px}}
.bar{{position:fixed;top:0;left:0;right:0;height:10px;background:#76B900}}
</style></head><body><div class="bar"></div>
<div class="wrap"><h1>How House Rules decides</h1>
<img src="{svg.resolve().as_uri()}"/></div></body></html>""",
            encoding="utf-8",
        )
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}/chrome",
            "--force-device-scale-factor=1",
            f"--window-size={size[0]},{size[1]}",
            f"--screenshot={out_png}",
            html.resolve().as_uri(),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=90, check=False)
        except subprocess.TimeoutExpired:
            return False
    return out_png.is_file() and out_png.stat().st_size > 0


def draw_architecture_card() -> Path:
    path = OUT / "architecture_card.png"
    if rasterize_svg(ARCH_SVG, path, (W, H)):
        # Ensure exact size
        img = Image.open(path).convert("RGB")
        if img.size != (W, H):
            canvas = Image.new("RGB", (W, H), hex_rgb(BG))
            img_r = img.copy()
            img_r.thumbnail((W - 80, H - 120), Image.Resampling.LANCZOS)
            canvas.paste(img_r, ((W - img_r.width) // 2, (H - img_r.height) // 2))
            d = ImageDraw.Draw(canvas)
            d.rectangle([0, 0, W, 10], fill=hex_rgb(ACCENT))
            canvas.save(path, "PNG")
        return path
    # Fallback: PIL card pointing at the SVG path
    img, draw = new_canvas()
    title(draw, "How House Rules decides")
    draw.text((64, 200), "Architecture diagram:", font=font(36), fill=hex_rgb(INK))
    draw.text((64, 260), str(ARCH_SVG.relative_to(ROOT)), font=font(28), fill=hex_rgb(MUTED))
    draw.rounded_rectangle([64, 340, W - 64, H - 120], radius=16, outline=hex_rgb(ACCENT), width=4)
    draw.text((100, 500), "Normalize → Rule gate → Router → Specialist / Vision / Multilingual", font=font(32), fill=hex_rgb(INK))
    draw.text((100, 560), "→ Confidence gate → Ultra escalation (gray zone only) → Verdict", font=font(32), fill=hex_rgb(INK))
    footer(draw, "Source: docs/img/architecture.svg (Chrome rasterizer unavailable)")
    img.save(path, "PNG")
    return path


def write_numbers(systems: dict[str, dict]) -> tuple[Path, Path]:
    payload = {
        "dataset": "nvidia_test (held-out NVIDIA Nemotron Content Safety 3.5 policy)",
        "dataset_n_target": 3340,
        "refusal_policy": "refusals and errors scored as incorrect (label flipped vs gold)",
        "cost_footnote": "Ours = GPU rental amortized at full utilization ($13/h × wall ÷ calls). Ultra/frontier = measured tokens.",
        "systems": {},
        "charts": {
            "quality_vs_cost.png": "macro_f1 vs usd_per_1m_msgs (log x), 95% bootstrap CI on F1",
            "monthly_bill.png": "usd_per_1m_msgs × 100 (100M messages/month)",
            "latency.png": "p50_ms and p95_ms",
            "title_card.png": "House Rules — Your safety policy, enforced.",
            "architecture_card.png": "from docs/img/architecture.svg",
        },
        "regenerate": ".venv/bin/python docs/video/make_charts.py",
    }
    for key in ORDER:
        s = systems[key]
        payload["systems"][key] = {
            "label": s["label"],
            "short": s["short"],
            "n": s["n"],
            "partial": s["partial"],
            "macro_f1": s["macro_f1"],
            "f1_ci95": s.get("f1_ci95"),
            "usd_per_1m_msgs": s["usd_per_1m_msgs"],
            "monthly_bill_100m": round(s["usd_per_1m_msgs"] * 100, 1),
            "p50_ms": s["p50_ms"],
            "p95_ms": s["p95_ms"],
            "refused_pct": s["refused_pct"],
            "source": s["source"],
            "ci_source": s.get("ci_source"),
            "cost_note": s.get("cost_note"),
        }

    jpath = OUT / "numbers.json"
    jpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Demo-video numbers (LIVE receipts)",
        "",
        "Regenerate charts + this file: `.venv/bin/python docs/video/make_charts.py`",
        "",
        "Dataset: `nvidia_test` held-out NVIDIA-policy rows (target n=3,340).",
        "Refusals/errors scored as **incorrect**. Cost footnote: ours = GPU rental amortized at full utilization.",
        "",
        "| System | n | Macro-F1 (95% CI) | $/1M msgs | Bill @100M/mo | p50 ms | p95 ms | refused % | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in ORDER:
        s = systems[key]
        ci = s.get("f1_ci95")
        ci_s = f"{s['macro_f1']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else f"{s['macro_f1']:.1f}"
        note = f"partial {s['n']}/3340" if s["partial"] else str(s["n"])
        lines.append(
            f"| {s['short']} | {note} | {ci_s} | {s['usd_per_1m_msgs']:,.1f} | "
            f"{s['usd_per_1m_msgs'] * 100:,.0f} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} | "
            f"{s['refused_pct']:.1f} | `{s['source']}` |"
        )
    lines.extend(
        [
            "",
            "## Chart files",
            "",
            "- `quality_vs_cost.png` — scatter, log $ / 1M vs macro-F1",
            "- `monthly_bill.png` — bar chart at 100M messages/month",
            "- `latency.png` — p50 / p95 bars",
            "- `title_card.png` — title",
            "- `architecture_card.png` — from `docs/img/architecture.svg`",
            "",
        ]
    )
    mpath = OUT / "numbers.md"
    mpath.write_text("\n".join(lines), encoding="utf-8")
    return jpath, mpath


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    systems = collect_systems()
    paths = [
        draw_title_card(),
        draw_quality_vs_cost(systems),
        draw_monthly_bill(systems),
        draw_latency(systems),
        draw_architecture_card(),
    ]
    jpath, mpath = write_numbers(systems)
    paths.extend([jpath, mpath])

    print("systems:")
    for key in ORDER:
        s = systems[key]
        ci = s.get("f1_ci95")
        ci_s = f" CI[{ci[0]}, {ci[1]}]" if ci else ""
        print(
            f"  {s['short']:14s} n={s['n']:4d}  F1={s['macro_f1']:5.1f}{ci_s}"
            f"  ${s['usd_per_1m_msgs']:,.1f}/1M  bill@100M=${s['usd_per_1m_msgs']*100:,.0f}"
            f"  p50={s['p50_ms']:.0f}  refused={s['refused_pct']:.1f}%"
        )
    print("wrote:")
    for p in paths:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
