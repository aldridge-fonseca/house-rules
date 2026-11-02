"""Write compact Markdown/CSV reports and dependency-free SVG charts."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .metrics import evaluate, paired_comparison


def _svg_chart(path: Path, title: str, labels: list[str], values: list[float], *, ylabel: str) -> None:
    width, height, left, bottom, top = 720, 400, 70, 55, 40
    maxv = max(values or [1]) or 1
    bars = []
    slot = (width - left - 30) / max(1, len(values))
    for i, (label, value) in enumerate(zip(labels, values)):
        h = max(1, (height - bottom - top) * value / maxv)
        x = left + i * slot + slot * .2
        y = height - bottom - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*.6:.1f}" height="{h:.1f}" fill="#457b9d"/>')
        bars.append(f'<text x="{x+slot*.3:.1f}" y="{height-bottom+20}" text-anchor="middle" font-size="11">{html.escape(label)}</text>')
        bars.append(f'<text x="{x+slot*.3:.1f}" y="{y-5:.1f}" text-anchor="middle" font-size="11">{value:.3g}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18">{html.escape(title)}</text>'
           f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">{html.escape(ylabel)}</text>'
           f'<line x1="{left}" y1="{height-bottom}" x2="{width-20}" y2="{height-bottom}" stroke="#333"/>'
           + "".join(bars) + "</svg>")
    path.write_text(svg, encoding="utf-8")


def _pareto_svg(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    priced = [(name, m) for name, m in summaries.items() if m["cost_per_1k_usd"] is not None]
    w, h, l, r, t, b = 720, 420, 80, 30, 50, 70
    if not priced:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420"><text x="360" y="210" text-anchor="middle">No priced systems</text></svg>')
        return
    xs = [m["cost_per_1k_usd"] for _, m in priced]
    ys = [m["accuracy"] for _, m in priced]
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1.0
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymin, ymax = max(0.0, ymin - .05), min(1.0, ymax + .05)
    px = lambda x: l + (x - xmin) / (xmax - xmin) * (w - l - r)
    py = lambda y: h - b - (y - ymin) / (ymax - ymin) * (h - t - b)
    content = [f'<text x="{w/2}" y="25" text-anchor="middle" font-size="18">Accuracy vs cost Pareto view</text>',
               f'<line x1="{l}" y1="{h-b}" x2="{w-r}" y2="{h-b}" stroke="#333"/>',
               f'<line x1="{l}" y1="{t}" x2="{l}" y2="{h-b}" stroke="#333"/>',
               f'<text x="{(l+w-r)/2}" y="{h-18}" text-anchor="middle">USD per 1,000 rows</text>',
               f'<text x="20" y="{h/2}" transform="rotate(-90 20 {h/2})" text-anchor="middle">Accuracy</text>']
    for name, m in priced:
        x, y = px(m["cost_per_1k_usd"]), py(m["macro_f1"])
        content += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#e76f51"/>',
                    f'<text x="{x+9:.1f}" y="{y-8:.1f}" font-size="11">{html.escape(name)}</text>']
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">{"".join(content)}</svg>', encoding="utf-8")


def _latency_svg(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    w, h, l, b, t = 760, 420, 70, 75, 50
    vals = [(name, m["latency_p50_ms"] or 0, m["latency_p95_ms"] or 0) for name, m in summaries.items()]
    vmax = max([max(x[1], x[2]) for x in vals] or [1]) or 1
    slot = (w - l - 25) / max(1, len(vals))
    body = [f'<text x="{w/2}" y="25" text-anchor="middle" font-size="18">Latency by system</text>',
            f'<text x="20" y="{h/2}" transform="rotate(-90 20 {h/2})" text-anchor="middle">Milliseconds</text>',
            f'<line x1="{l}" y1="{h-b}" x2="{w-20}" y2="{h-b}" stroke="#333"/>']
    for i, (name, p50, p95) in enumerate(vals):
        x = l + i * slot + slot * .18
        for j, (v, color, tag) in enumerate(((p50, "#2a9d8f", "p50"), (p95, "#e76f51", "p95"))):
            bh = max(1, (h-b-t) * v / vmax)
            y = h-b-bh
            bx = x + j * slot * .32
            body.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{slot*.25:.1f}" height="{bh:.1f}" fill="{color}"/>')
            body.append(f'<text x="{bx+slot*.125:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="9">{v:.0f}</text>')
        body.append(f'<text x="{x+slot*.3:.1f}" y="{h-b+18}" text-anchor="middle" font-size="10">{html.escape(name)}</text>')
    body.append(f'<rect x="{w-150}" y="42" width="10" height="10" fill="#2a9d8f"/><text x="{w-135}" y="51" font-size="11">p50</text>')
    body.append(f'<rect x="{w-95}" y="42" width="10" height="10" fill="#e76f51"/><text x="{w-80}" y="51" font-size="11">p95</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">{"".join(body)}</svg>', encoding="utf-8")


def _group_sections(summaries: dict[str, dict[str, Any]], out: Path) -> list[str]:
    """Markdown sections + by_group.csv for perturbation / route / source breakdowns."""
    fmt = lambda x: "n/a" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
    cols = ["group_by", "group", "system", "n", "accuracy", "unsafe_f1", "unsafe_recall", "macro_f1",
            "over_block_rate", "latency_p50_ms", "latency_p95_ms", "cost_per_1k_usd"]
    lines: list[str] = []
    with (out / "by_group.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for key, title in (("perturbation", "By perturbation (absent = none)"),
                           ("route", "By route trail"), ("source", "By source")):
            lines += ["", f"## {title}", "",
                      "| System | " + key + " | n | Accuracy | Unsafe F1 | Unsafe recall | Over-block | Δ unsafe recall vs none |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
            for name, m in summaries.items():
                groups = m.get(f"by_{key}", {})
                clean = (groups.get("none") or {}).get("unsafe_recall") if key == "perturbation" else None
                for group, g in groups.items():
                    writer.writerow({"group_by": key, "group": group, "system": name, **{c: g.get(c) for c in cols[3:]}})
                    delta = (g["unsafe_recall"] - clean) if clean is not None and g["unsafe_recall"] is not None and group != "none" else None
                    lines.append(f"| {name} | {group} | {g['n']} | {fmt(g['accuracy'])} | {fmt(g['unsafe_f1'])} | "
                                 f"{fmt(g['unsafe_recall'])} | {fmt(g['over_block_rate'])} | {fmt(delta)} |")
    return lines


def build_report(receipts: dict[str, list[dict[str, Any]]], out_dir: str | Path, *,
                 baseline: str = "frontier", seed: int = 0,
                 banner: str | None = None) -> dict[str, dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {name: evaluate(rows) for name, rows in receipts.items()}
    comparisons = {name: paired_comparison(receipts[baseline], rows, seed=seed)
                   for name, rows in receipts.items() if baseline in receipts and name != baseline}
    cols = ["system", "n", "unsafe_f1", "unsafe_precision", "unsafe_recall", "macro_f1", "over_block_rate",
            "latency_p50_ms", "latency_p95_ms", "cost_total_usd", "cost_per_1k_usd", "vs_frontier"]
    with (out / "table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for name, metrics in summaries.items():
            xstest = next((v for src, v in metrics["over_block_by_source"].items() if "xstest" in src.lower()), metrics["over_block_rate"])
            writer.writerow({"system": name, **{k: metrics.get(k) for k in cols if k not in ("system", "over_block_rate")},
                             "over_block_rate": xstest,
                             "vs_frontier": json.dumps(comparisons.get(name), sort_keys=True) if name in comparisons else ""})
    lines = [f"> {banner}", ""] if banner else []
    lines += ["| System | Unsafe F1 | Precision | Recall | Macro F1 | Over-block | p50 ms | p95 ms | Total $ | $/1k |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in summaries.items():
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
        xstest = next((v for src, v in m["over_block_by_source"].items() if "xstest" in src.lower()), m["over_block_rate"])
        lines.append(f"| {name} | {fmt(m['unsafe_f1'])} | {fmt(m['unsafe_precision'])} | {fmt(m['unsafe_recall'])} | {fmt(m['macro_f1'])} | {fmt(xstest)} | {fmt(m['latency_p50_ms'])} | {fmt(m['latency_p95_ms'])} | {fmt(m['cost_total_usd'])} | {fmt(m['cost_per_1k_usd'])} |")
    lines += ["", "Paired comparisons use ID-matched rows; CI is the paired bootstrap 95% interval for unsafe F1 difference (system − frontier), with exact two-sided McNemar on correctness.", "", "## Paired comparisons vs frontier", ""]
    for name, comp in comparisons.items():
        lines.append(f"- **{name}:** Δ={comp['difference']:.4f}, 95% CI [{comp['ci95'][0]:.4f}, {comp['ci95'][1]:.4f}], McNemar p={comp['mcnemar_p']:.4g}, n={comp['n_paired']}.")
    lines += ["", "## Per-category F1", ""]
    for name, m in summaries.items():
        vals = ", ".join(f"{cat}={v['f1']:.3f}" for cat, v in m["per_category"].items()) or "no category labels"
        lines.append(f"- **{name}:** {vals}")
    lines += ["", "## Route share and accuracy", ""]
    for name, m in summaries.items():
        route_vals = []
        for route, v in m["routes"].items():
            accuracy = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.3f}"
            route_vals.append(f"{route}={v['share']:.1%} share/{accuracy} accuracy (n={v['n']})")
        vals = ", ".join(route_vals)
        lines.append(f"- **{name}:** {vals}")
    lines += _group_sections(summaries, out)
    lines += ["", "Cost uses the per-verdict cost_usd and excludes training/deployment infrastructure. Mock-provider latency and cost are synthetic."]
    (out / "summary.json").write_text(json.dumps({"banner": banner, "systems": summaries}, indent=1, default=str),
                                      encoding="utf-8")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    labels = list(summaries)
    _pareto_svg(out / "pareto.svg", summaries)
    _latency_svg(out / "latency.svg", summaries)
    failures = []
    for name, rows in receipts.items():
        for r in rows:
            pred = r.get("prediction") or {}
            if (r.get("error") or pred.get("label") != r.get("gold_label") or
                    not set(r.get("gold_categories") or []).issubset(set(pred.get("categories") or []))):
                failures.append({"system": name, "id": r.get("id"), "text": r.get("text"),
                                 "gold_label": r.get("gold_label"), "prediction": r.get("prediction"), "error": r.get("error")})
    (out / "failure-examples.md").write_text("# Failure examples\n\n" + ("\n".join(
        f"- **{r['system']} / {r['id']}** gold={r['gold_label']!r}, predicted={(r.get('prediction') or {}).get('label')!r}; {r.get('text','')[:240]!r}"
        for r in failures[:20]) if failures else "No failures in this run.") + "\n", encoding="utf-8")
    return summaries
