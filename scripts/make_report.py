#!/usr/bin/env python3
"""Turn a results/<run>/ directory into judge-ready report artifacts.

    python scripts/make_report.py results/MOCK-integration-e2e
    python scripts/make_report.py results/nvidia-v1 --png-dir docs/img
    # results/frozen-v1 / results/vision-r4 regenerate an earlier internal run
    # whose receipts are not published; prefer nvidia-v1 for submission figures.

Reads receipt JSONL (preferred) or summary.json produced by safety_one.eval,
plus RUN_INFO.json, and writes results/<run>/report/:

    headline.md   dataset x system table, MOCK/LIVE banner, PUBLISHED frontier,
                  matched-rows table when some systems ran only a subset
    tradeoff.svg  macro-F1 vs $ / 1M messages (log x), bubble = p95 latency
    disguise.svg  unsafe recall by disguise type (+ frontier refusal rate)
    ablation.svg  specialist-only -> +rules -> +router -> +escalation (full)
    vision.svg    image route panel (only with --vision)
    index.html    offline page embedding the tables and the SVGs

Row sets are never mixed inside one column: a system whose receipts cover only
a subset of a dataset is shown as "n of N" in the all-rows table and scored in
a separate matched-rows table (and the tradeoff/disguise charts) on exactly the
rows every compared system ran.

Scoring (receipts path):
  * provider refusals (error contains "ContentFilter") count as **unsafe**; the
    refusal rate is reported next to them. Other errors stay as wrong answers.
  * latency percentiles use verdict rows only (refusal receipts carry 0 ms).
  * $ = receipt cost_usd + billed refusal cost from ../spend.json (the ledger is
    the only place a refusal's cost survives) + allocated dedicated-GPU cost
    (nano_summary.json gpu_per_call_usd x calls served by a /deployments/ model).

Stdlib only (PNG export shells out to a headless Chrome/Chromium if present).
Metrics are recomputed with safety_one.eval.metrics.evaluate so they match the
harness. Published frontier cells are parsed from data/BENCHMARKS.md and
data/frozen/README.md — never filled in by guesswork.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_one.eval.metrics import _percentile, evaluate

NVIDIA_GREEN = "#76B900"
NVIDIA_BASE_FILL = "#D7EBB5"
NVIDIA_BASE_STROKE = "#3E7800"
FRONTIER_GREY = "#767676"
FRONTIER_SHADES = ("#767676", "#4A4A4A", "#A0A0A0")
REFUSAL_RED = "#C62828"
OTHER_BLUE = "#457B9D"
NVIDIA_SHADES = ("#76B900", "#5C9300", "#3E7800", "#245C00", "#0E3F00")
LADDER = ("specialist-only", "+rules", "+router", "+escalation", "full")
CALIBRATED = ("specialist-cal", "full-cal")
NVIDIA_SYSTEMS = set(LADDER) | set(CALIBRATED)
# Collapsed in the tradeoff/disguise charts when the run has many systems: they
# sit on top of specialist-only/full (same model, ±1.5 F1, same $) and are all
# drawn in the ablation chart and listed in the tables.
CHART_COLLAPSE = {"+rules", "+router", "+escalation", "specialist-cal", "full-cal"}
REFUSAL_MARK = "ContentFilter"
# Last slot is one step: config/systems.yaml wires `full` the same as `+escalation`.
ABLATION_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("specialist-only", ("specialist-only",)),
    ("+rules", ("+rules",)),
    ("+router", ("+router",)),
    ("+escalation (full)", ("full", "+escalation")),
)
CALIBRATED_SLOTS: tuple[tuple[str, str], ...] = (
    ("specialist-cal", "specialist-cal"),
    ("full-cal", "full-cal"),
)
DISGUISE_ORDER = (
    "none", "leetspeak", "homoglyph", "char_spacing", "zero_width",
    "emoji_sub", "base64_wrap", "mixed_case",
)
DATASETS = (
    ("OpenAI Moderation", "openai_moderation"),
    ("WildGuardTest", "wildguard_test"),
    ("XSTest over-block", "xstest"),
    ("Disguised", None),
)
HEADLINE_COLUMNS = (
    "Dataset", "System", "n", "macro-F1", "unsafe recall", "over-block rate",
    "p50 ms", "p95 ms", "$ / 1M msgs", "published frontier",
)
MATCHED_COLUMNS = (
    "Slice", "System", "model", "n", "macro-F1", "unsafe recall %", "over-block %",
    "refusal %", "p50 ms", "p95 ms", "$ / 1M msgs",
)
MODEL_LABELS = {
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
    "anthropic/claude-opus-5.5": "Claude Opus 5.5",
    "accounts/fireworks/models/nemotron-3-ultra-nvfp4": "Nemotron 3 Ultra 550B",
}
SYSTEM_LABELS = {
    "nvidia-ultra-zeroshot": "untuned Nemotron 3 Ultra",
    "full": "full (ours)",
    "full-cal": "full-cal (ours)",
}


def is_nvidia(name: str) -> bool:
    return name in NVIDIA_SYSTEMS


def is_nvidia_baseline(name: str) -> bool:
    return name.startswith("nvidia-") and name not in NVIDIA_SYSTEMS


def is_frontier(name: str) -> bool:
    lowered = name.lower()
    return lowered == "frontier" or lowered.startswith("frontier-") or lowered.startswith("frontier_")


def is_refusal(row: dict[str, Any]) -> bool:
    return REFUSAL_MARK in str(row.get("error") or "")


def is_disguise_row(row: dict[str, Any]) -> bool:
    """Robustness/disguise rows, including clean controls inside that file.

    The eval runner copies `id`, `source`, and `perturbation` onto receipts
    and drops `meta`. Current disguise ids start with `rob-`. Any row whose
    perturbation is not `none` is an attack row even if a future generator
    uses another id prefix.
    """
    meta = row.get("meta") or {}
    if isinstance(meta, dict) and meta.get("robustness_slice"):
        return True
    if str(row.get("id") or "").startswith("rob-"):
        return True
    return str(row.get("perturbation") or "none") not in ("none", "")


def fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}"


def fmt_f1_100(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}"


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def fmt_ms0(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.2f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def fmt_money0(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}" if abs(value) >= 10 else f"{value:.2f}"


def per_million(cost_per_1k: float | None) -> float | None:
    """$ / 1M messages from the harness `$ / 1k rows` field."""
    if cost_per_1k is None:
        return None
    return float(cost_per_1k) * 1000.0


def classify_provenance(
    run_info: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    modes: dict[str, str] | None = None,
) -> tuple[str, str]:
    """MOCK / LIVE / MIXED / UNKNOWN from per-system modes, then the banner."""
    info = run_info or {}
    summ = summary or {}
    banner = str(info.get("banner") or summ.get("banner") or "").strip()
    if modes is None:
        modes = info.get("modes") if isinstance(info.get("modes"), dict) else {}
    if modes:
        values = [str(v).lower() for v in modes.values()]
        if values and all(v == "mock" for v in values):
            kind = "MOCK"
        elif values and all(v == "live" for v in values):
            kind = "LIVE"
        else:
            kind = "MIXED"
    else:
        head = banner.upper()
        if head.startswith("MOCK"):
            kind = "MOCK"
        elif head.startswith("LIVE"):
            kind = "LIVE"
        elif head.startswith("MIXED"):
            kind = "MIXED"
        else:
            kind = "UNKNOWN"
    if not banner:
        banner = f"{kind} — no banner in RUN_INFO.json or summary.json"
    return kind, banner


def _strip_md(text: str) -> str:
    return " ".join(text.replace("**", "").replace("✓", "").split())


def load_published(benchmarks_path: Path, frozen_readme: Path | None = None) -> dict[str, str]:
    """Published frontier cells. Unparsed benchmarks stay explicitly unparsed."""
    text = benchmarks_path.read_text(encoding="utf-8") if benchmarks_path.is_file() else ""
    cells: dict[str, str] = {}

    openai = re.search(
        r"OpenAI Moderation — GPT-5\.1 \*\*([0-9]+(?:\.[0-9]+)?)\*\* \(unverified\)",
        text,
    )
    wild = re.search(
        r"WildGuard — GPT-4o\+CoT \*\*([0-9]+(?:\.[0-9]+)?)\*\*",
        text,
    )
    xstest = re.search(
        r"^\| XSTest \| (?P<model>[^|]+?) \| (?P<score>[^|]+?) \| (?P<src>[^|]+?) \|\s*$",
        text,
        re.M,
    )
    if openai:
        cells["OpenAI Moderation"] = (
            f"PUBLISHED GPT-5.1 macro-F1 {openai.group(1)} "
            "(unverified; data/BENCHMARKS.md frontier reference; not our run)"
        )
    else:
        cells["OpenAI Moderation"] = "PUBLISHED unparsed — GPT-5.1 frontier line missing from data/BENCHMARKS.md"
    if wild:
        cells["WildGuardTest"] = (
            f"PUBLISHED GPT-4o+CoT macro-F1 {wild.group(1)} "
            "(data/BENCHMARKS.md frontier reference; arXiv:2605.31073; not our run)"
        )
    else:
        cells["WildGuardTest"] = "PUBLISHED unparsed — GPT-4o+CoT frontier line missing from data/BENCHMARKS.md"
    if xstest:
        cells["XSTest over-block"] = (
            "PUBLISHED "
            + _strip_md(xstest.group("model"))
            + ": "
            + _strip_md(xstest.group("score"))
            + " — "
            + _strip_md(xstest.group("src"))
            + "; data/BENCHMARKS.md; not our run"
        )
    else:
        cells["XSTest over-block"] = "PUBLISHED unparsed — XSTest row missing from data/BENCHMARKS.md"

    readme = ""
    if frozen_readme and frozen_readme.is_file():
        readme = frozen_readme.read_text(encoding="utf-8")
    if "no published number" in readme:
        cells["Disguised"] = (
            "PUBLISHED none — no published frontier number "
            "(data/frozen/README.md; not our run)"
        )
    else:
        cells["Disguised"] = (
            "PUBLISHED none — data/frozen/README.md did not state a frontier number"
        )
    return cells


def published_context(benchmarks_path: Path) -> str:
    """Open-guard scores the frontier paragraph already contrasts. Not our run."""
    if not benchmarks_path.is_file():
        return ""
    text = benchmarks_path.read_text(encoding="utf-8")
    siren = re.search(r"best open guard SIREN-1B \*\*([0-9.]+)\*\*", text)
    qwen = re.search(r"Qwen3Guard-4B \*\*([0-9.]+)\*\*", text)
    bits = []
    if siren:
        bits.append(f"OpenAI Moderation SIREN (Llama3.2-1B) macro-F1 {siren.group(1)}")
    if qwen:
        bits.append(f"WildGuardTest Qwen3Guard-4B macro-F1 {qwen.group(1)}")
    if not bits:
        return ""
    return (
        "PUBLISHED open-guard scores in the same frontier paragraph (not the frontier column, not our run): "
        + "; ".join(bits)
        + "."
    )


def load_run(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    """Receipts per system; the last record per id wins (resumed runs append retries)."""
    info_path = run_dir / "RUN_INFO.json"
    summary_path = run_dir / "summary.json"
    run_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else None
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSON: {exc}") from exc
            name = str(row.get("system") or path.stem)
            rid = str(row.get("id") if row.get("id") is not None else f"{path.stem}:{line_no}")
            grouped.setdefault(name, {})[rid] = row
    receipts = {name: list(rows.values()) for name, rows in grouped.items()}
    return run_info, summary, receipts


def system_order(names: list[str]) -> list[str]:
    order = list(LADDER) + list(CALIBRATED)

    def key(name: str) -> tuple[int, int, str]:
        if name in order:
            return (0, order.index(name), name)
        if is_nvidia(name):
            return (1, 0, name)
        if is_nvidia_baseline(name):
            return (2, 0, name)
        if is_frontier(name):
            return (3, 0, name)
        return (4, 0, name)

    return sorted(names, key=key)


# ------------------------------------------------------------------ run context


@dataclass
class RunContext:
    """Everything needed to score a system's receipts consistently."""
    gpu_per_call: float = 0.0
    ledger: dict[tuple[str, str], float] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)
    mode_sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def system_model(rows: list[dict[str, Any]]) -> str:
    """Most common prediction.model over verdict rows (the arm's model id)."""
    counts = Counter(
        str((r.get("prediction") or {}).get("model") or "")
        for r in rows if not r.get("error")
    )
    counts.pop("", None)
    return counts.most_common(1)[0][0] if counts else ""


def short_model(model: str) -> str:
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    if "/deployments/" in model and "-vl-" in model:
        return "Nemotron 3.5 Super VL (on-demand)"
    if "/deployments/" in model:
        return "tuned Nemotron Nano 3 (LoRA, dedicated)"
    return model.split("/")[-1] if model else ""


def display_name(name: str, ctx: RunContext | None = None) -> str:
    """Chart label: frontier arms by model, the untuned NVIDIA baseline spelled out."""
    model = (ctx.models.get(name) if ctx else "") or ""
    if is_frontier(name) and model in MODEL_LABELS:
        return MODEL_LABELS[model]
    return SYSTEM_LABELS.get(name, name)


def load_ledger(path: Path) -> dict[tuple[str, str], float]:
    """(model, row_id) -> cost of the most recent settled frontier-router call."""
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[tuple[str, str], float] = {}
    for entry in doc.get("entries", []) if isinstance(doc, dict) else []:
        md = entry.get("metadata") or {}
        if entry.get("provider") == "frontier-router" and md.get("row_id"):
            out[(str(md.get("model")), str(md["row_id"]))] = float(entry.get("cost_usd") or 0.0)
    return out


def _live_evidence(row: dict[str, Any]) -> bool:
    """A provider response id, or real token usage from a Fireworks-hosted model."""
    pred = row.get("prediction") or {}
    raw = pred.get("raw") if isinstance(pred.get("raw"), dict) else {}
    if raw.get("mock"):
        return False
    if (raw.get("response") or {}).get("id"):
        return True
    model = str(pred.get("model") or "")
    usage = raw.get("usage") or {}
    return ((model.startswith("accounts/") and int(usage.get("prompt_tokens") or 0) > 0)
            or str(pred.get("route")) == "rule")


def system_modes(
    run_dir: Path,
    run_info: dict[str, Any] | None,
    receipts: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Per-system MOCK/LIVE label and where it came from.

    RUN_INFO.json modes first, then frontier_summary.json's label for its arms,
    then receipt evidence (raw.mock -> mock; a provider response id or token
    usage from an accounts/... Fireworks model on >=90 % of verdict rows -> live).
    Anything else stays `unknown`.
    """
    modes: dict[str, str] = {}
    sources: dict[str, str] = {}
    info_modes = (run_info or {}).get("modes") if isinstance((run_info or {}).get("modes"), dict) else {}
    for name, mode in (info_modes or {}).items():
        modes[str(name)] = str(mode).lower()
        sources[str(name)] = "RUN_INFO.json"
    fs_path = run_dir / "frontier_summary.json"
    if fs_path.is_file():
        try:
            fs = json.loads(fs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fs = {}
        label = str(fs.get("label") or "").lower()
        for arm in (fs.get("arms") or {}):
            if label and arm not in modes:
                modes[arm] = label
                sources[arm] = "frontier_summary.json"
    for name, rows in receipts.items():
        if name in modes:
            continue
        ok = [r for r in rows if not r.get("error")]
        raws = [((r.get("prediction") or {}).get("raw") or {}) for r in ok]
        if any(isinstance(raw, dict) and raw.get("mock") for raw in raws):
            modes[name], sources[name] = "mock", "receipts (raw.mock)"
        elif ok and sum(_live_evidence(r) for r in ok) >= 0.9 * len(ok):
            modes[name], sources[name] = "live", "receipts (provider response ids / token usage on Fireworks)"
        else:
            modes[name], sources[name] = "unknown", "no label found"
    return modes, sources


def load_context(run_dir: Path, run_info: dict[str, Any] | None,
                 receipts: dict[str, list[dict[str, Any]]]) -> RunContext:
    ctx = RunContext()
    nano = run_dir / "nano_summary.json"
    if nano.is_file():
        try:
            ctx.gpu_per_call = float(json.loads(nano.read_text(encoding="utf-8")).get("gpu_per_call_usd") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            ctx.gpu_per_call = 0.0
        if ctx.gpu_per_call:
            ctx.notes.append(
                f"Dedicated-GPU cost allocated at ${ctx.gpu_per_call * 1e6:,.0f} per 1M calls served by a "
                "/deployments/ model (nano_summary.json gpu_per_call_usd: GPU $/h x measured ladder "
                "wall-clock / specialist calls; receipts carry $0 for those calls)."
            )
    if receipts and any(is_frontier(n) for n in receipts):
        ctx.ledger = load_ledger(run_dir.parent / "spend.json")
        if ctx.ledger:
            ctx.notes.append(
                "Billed cost of frontier refusals is taken from results/spend.json (last settled call per "
                "row and model); refusal receipts themselves store $0."
            )
    ctx.models = {name: system_model(rows) for name, rows in receipts.items()}
    ctx.modes, ctx.mode_sources = system_modes(run_dir, run_info, receipts)
    return ctx


def score(rows: list[dict[str, Any]], ctx: RunContext | None = None, model: str = "") -> dict[str, Any] | None:
    """evaluate() with refusals as unsafe, verdict-row latency, full $ accounting."""
    if not rows:
        return None
    ctx = ctx or RunContext()
    scored = [
        {**r, "prediction": {**(r.get("prediction") or {}), "label": "unsafe"}} if is_refusal(r) else r
        for r in rows
    ]
    metrics = evaluate(scored, groups=False)
    ok = [r for r in rows if not r.get("error")]
    latencies = [float((r.get("prediction") or {}).get("latency_ms") or 0) for r in ok]
    receipt_usd = sum(float((r.get("prediction") or {}).get("cost_usd") or 0) for r in rows)
    refusals = [r for r in rows if is_refusal(r)]
    refusal_usd = sum(ctx.ledger.get((model, str(r.get("id"))), 0.0) for r in refusals) if model else 0.0
    gpu_usd = ctx.gpu_per_call * sum(
        1 for r in ok if "/deployments/" in str((r.get("prediction") or {}).get("model") or "")
    )
    total = receipt_usd + refusal_usd + gpu_usd
    n = len(rows)
    gold_unsafe = [r for r in rows if str(r.get("gold_label", "")).lower() == "unsafe"]
    verdict_caught = sum(
        1 for r in gold_unsafe
        if not r.get("error") and str((r.get("prediction") or {}).get("label", "")).lower() == "unsafe"
    )
    metrics.update({
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "cost_total_usd": total,
        "cost_per_1k_usd": total * 1000 / n,
        "receipt_usd": receipt_usd,
        "refusal_ledger_usd": refusal_usd,
        "gpu_alloc_usd": gpu_usd,
        "refusals": len(refusals),
        "refusal_rate": len(refusals) / n,
        "refusal_share_of_unsafe": (sum(1 for r in gold_unsafe if is_refusal(r)) / len(gold_unsafe)
                                    if gold_unsafe else None),
        "verdict_recall": verdict_caught / len(gold_unsafe) if gold_unsafe else None,
        "errors": sum(1 for r in rows if r.get("error") and not is_refusal(r)),
    })
    return metrics


def _score_system(name: str, rows: list[dict[str, Any]], ctx: RunContext) -> dict[str, Any] | None:
    return score(rows, ctx, ctx.models.get(name, ""))


def _from_summary_block(block: dict[str, Any] | None) -> dict[str, Any] | None:
    if not block:
        return None
    cost = block.get("cost_per_1k_usd")
    return {
        "n": block.get("n") or 0,
        "macro_f1": block.get("macro_f1"),
        "unsafe_recall": block.get("unsafe_recall"),
        "over_block_rate": block.get("over_block_rate"),
        "latency_p50_ms": block.get("latency_p50_ms"),
        "latency_p95_ms": block.get("latency_p95_ms"),
        "cost_per_1k_usd": cost,
    }


def _blank_row(dataset: str, system: str, published: str, n: Any = 0) -> dict[str, Any]:
    return {
        "dataset": dataset, "system": system, "n": n,
        "macro_f1": None, "unsafe_recall": None, "over_block_rate": None,
        "p50": None, "p95": None, "usd_per_1m": None, "published": published,
    }


def dataset_slice(rows: list[dict[str, Any]], dataset: str, source: str | None) -> list[dict[str, Any]]:
    if dataset == "Disguised":
        return [r for r in rows if is_disguise_row(r)]
    return [r for r in rows if r.get("source") == source and not is_disguise_row(r)]


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(r.get("id")) for r in rows}


def dataset_rows(
    names: list[str],
    receipts: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any] | None,
    published: dict[str, str],
    ctx: RunContext | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """One record per dataset x system. Receipts win over summary.json.

    A system that ran only some of a dataset's rows gets "n of N" and no
    metrics here — its numbers live in the matched-rows table.
    """
    ctx = ctx or RunContext()
    note = ""
    rows: list[dict[str, Any]] = []
    systems = (summary or {}).get("systems") or {}
    use_receipts = bool(receipts)
    if not use_receipts and summary:
        note = (
            "Receipt JSONL was not in this run directory. Public dataset rows are "
            "summary.json by_source and can mix disguise rows of the same source. "
            "Pooled Disguised macro-F1 needs the receipt JSONL."
        )
    elif not use_receipts:
        note = "No receipt JSONL and no summary.json systems in this run directory."
    subset_seen = False
    for dataset, source in DATASETS:
        cell = published.get(dataset, "PUBLISHED unparsed")
        if not names:
            rows.append(_blank_row(dataset, "—", cell))
            continue
        slices = {name: dataset_slice(receipts.get(name) or [], dataset, source) for name in names} if use_receipts else {}
        reference: set[str] = set().union(*(_ids(s) for s in slices.values())) if slices else set()
        for name in names:
            metrics: dict[str, Any] | None
            if use_receipts:
                chosen = slices[name]
                if chosen and _ids(chosen) != reference:
                    subset_seen = True
                    rows.append(_blank_row(
                        dataset, name, cell,
                        n=f"{len(chosen):,} of {len(reference):,} — subset, see matched-rows table",
                    ))
                    continue
                metrics = _score_system(name, chosen, ctx)
            else:
                block = systems.get(name) or {}
                if dataset == "Disguised":
                    metrics = None
                else:
                    metrics = _from_summary_block((block.get("by_source") or {}).get(source or ""))
            if metrics is None:
                rows.append(_blank_row(dataset, name, cell))
                continue
            rows.append({
                "dataset": dataset,
                "system": name,
                "n": metrics["n"],
                "macro_f1": metrics.get("macro_f1"),
                "unsafe_recall": metrics.get("unsafe_recall"),
                "over_block_rate": metrics.get("over_block_rate"),
                "p50": metrics.get("latency_p50_ms"),
                "p95": metrics.get("latency_p95_ms"),
                "usd_per_1m": per_million(metrics.get("cost_per_1k_usd")),
                "published": cell,
            })
    if subset_seen:
        extra = ("Systems marked \"subset\" ran only part of that dataset; comparing them in this column "
                 "would mix row sets, so they are scored in the matched-rows table instead.")
        note = f"{note} {extra}".strip()
    return rows, note


# ------------------------------------------------------------------ coverage


@dataclass
class Coverage:
    universe: set[str]
    full: list[str]
    partial: list[str]
    matched_ids: set[str]
    matched_systems: list[str]
    excluded: list[str]

    @property
    def has_matched(self) -> bool:
        return bool(self.partial) and bool(self.matched_ids)


def coverage(names: list[str], receipts: dict[str, list[dict[str, Any]]]) -> Coverage:
    """Which systems ran every row, and the rows every compared system shares.

    Matched rows = intersection of the partial frontier arms (or of all partial
    systems when no frontier arm is partial). Systems that miss any matched row
    are left out of the matched views instead of shrinking them.
    """
    ids = {name: _ids(receipts.get(name) or []) for name in names}
    universe: set[str] = set().union(*ids.values()) if ids else set()
    full = [n for n in names if ids[n] and ids[n] == universe]
    partial = [n for n in names if ids[n] and ids[n] != universe]
    anchors = [n for n in partial if is_frontier(n)] or partial
    matched = set.intersection(*(ids[n] for n in anchors)) if anchors else set()
    matched_systems = [n for n in names if ids[n] and matched and matched <= ids[n]]
    excluded = [n for n in names if ids[n] and n not in matched_systems] if matched else []
    return Coverage(universe, full, partial, matched, matched_systems, excluded)


def restrict(rows: list[dict[str, Any]], keep: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("id")) in keep]


def overall_metrics(
    names: list[str],
    receipts: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any] | None,
    ctx: RunContext | None = None,
    keep: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    ctx = ctx or RunContext()
    systems = (summary or {}).get("systems") or {}
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if receipts.get(name):
            rows = receipts[name] if keep is None else restrict(receipts[name], keep)
            metrics = _score_system(name, rows, ctx)
            if metrics:
                out[name] = metrics
        elif name in systems:
            block = systems[name]
            out[name] = {
                "n": block.get("n") or 0,
                "macro_f1": block.get("macro_f1"),
                "unsafe_recall": block.get("unsafe_recall"),
                "over_block_rate": block.get("over_block_rate"),
                "latency_p50_ms": block.get("latency_p50_ms"),
                "latency_p95_ms": block.get("latency_p95_ms"),
                "cost_per_1k_usd": block.get("cost_per_1k_usd"),
            }
    return out


def disguise_series(
    names: list[str],
    receipts: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any] | None,
    ctx: RunContext | None = None,
    keep: set[str] | None = None,
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    """Per disguise type: unsafe recall, refusal rate (and n) for each system."""
    ctx = ctx or RunContext()
    found: set[str] = set()
    series: dict[str, dict[str, dict[str, Any]]] = {}
    systems = (summary or {}).get("systems") or {}
    for name in names:
        series[name] = {}
        if receipts.get(name):
            buckets: dict[str, list[dict[str, Any]]] = {}
            rows = receipts[name] if keep is None else restrict(receipts[name], keep)
            for row in rows:
                if not is_disguise_row(row):
                    continue
                pert = str(row.get("perturbation") or "none")
                buckets.setdefault(pert, []).append(row)
            for pert, bucket in buckets.items():
                metrics = _score_system(name, bucket, ctx) or {}
                series[name][pert] = {
                    "n": metrics.get("n", 0),
                    "unsafe_recall": metrics.get("unsafe_recall"),
                    "verdict_recall": metrics.get("verdict_recall"),
                    "refusal_rate": metrics.get("refusal_rate"),
                    "refusal_share_of_unsafe": metrics.get("refusal_share_of_unsafe"),
                }
                found.add(pert)
        else:
            groups = (systems.get(name) or {}).get("by_perturbation") or {}
            for pert, block in groups.items():
                if pert == "none":
                    continue
                series[name][pert] = {
                    "n": block.get("n") or 0,
                    "unsafe_recall": block.get("unsafe_recall"),
                }
                found.add(pert)
    ordered = [p for p in DISGUISE_ORDER if p in found]
    ordered += sorted(found - set(ordered))
    return ordered, series


def disguise_scope(
    names: list[str],
    receipts: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], set[str] | None, str]:
    """Disguise rows every system with disguise receipts shares (matched on that slice only)."""
    if not receipts:
        return names, None, ""
    ids = {n: _ids([r for r in receipts.get(n) or [] if is_disguise_row(r)]) for n in names}
    present = [s for s in ids.values() if s]
    if not present:
        return names, None, ""
    shared = set.intersection(*present)
    if all(s == shared for s in present):
        return names, None, f"Every system with disguise receipts ran the same {len(shared):,} disguise rows."
    return names, shared, f"Matched {len(shared):,} disguise rows that every system with disguise receipts ran."


def chart_systems(names: list[str]) -> list[str]:
    """Drop near-duplicate ladder steps from crowded comparison charts."""
    if len(names) <= 5:
        return list(names)
    kept = [n for n in names if n not in CHART_COLLAPSE]
    return kept if len(kept) >= 2 else list(names)


def role_of(name: str) -> str:
    if is_frontier(name):
        return "frontier"
    if is_nvidia(name):
        return "nvidia"
    if is_nvidia_baseline(name):
        return "nvidia-baseline"
    return "other"


def color_of(name: str, nvidia_index: dict[str, int], frontier_index: dict[str, int] | None = None) -> str:
    role = role_of(name)
    if role == "frontier":
        idx = (frontier_index or {}).get(name, 0)
        return FRONTIER_SHADES[idx % len(FRONTIER_SHADES)]
    if role == "nvidia":
        return NVIDIA_SHADES[nvidia_index[name] % len(NVIDIA_SHADES)]
    if role == "nvidia-baseline":
        return NVIDIA_BASE_FILL
    return OTHER_BLUE


def nvidia_index_map(names: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    cursor = 0
    for name in names:
        if is_nvidia(name):
            index[name] = cursor
            cursor += 1
    return index


def frontier_index_map(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(n for n in names if is_frontier(n))}


def _svg_header(width: int, height: int, title: str, kind: str) -> list[str]:
    safe_title = html.escape(title)
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{safe_title}</title>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<rect width="{width}" height="6" fill="{NVIDIA_GREEN}"/>',
        f'<text x="{width - 16}" y="28" text-anchor="end" font-size="13" font-weight="700" '
        f'fill="#333" font-family="sans-serif">{html.escape(kind)}</text>',
        f'<text x="{width / 2:.0f}" y="32" text-anchor="middle" font-size="18" '
        f'font-family="sans-serif" fill="#1a1a1a">{safe_title}</text>',
    ]


def _empty_svg(title: str, message: str, kind: str) -> str:
    body = _svg_header(720, 280, title, kind)
    body.append(
        f'<text x="360" y="150" text-anchor="middle" font-size="14" font-family="sans-serif" '
        f'fill="#333">{html.escape(message)}</text></svg>'
    )
    return "\n".join(body)


def _text_w(text: str, size: float) -> float:
    return len(text) * size * 0.56


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def tradeoff_svg(
    overall: dict[str, dict[str, Any]],
    names: list[str],
    kind: str,
    ctx: RunContext | None = None,
    scope: str = "All rows in the run.",
) -> str:
    ctx = ctx or RunContext()
    title = f"Quality vs cost vs latency — {kind}"
    points = []
    for name in names:
        metrics = overall.get(name)
        if not metrics or metrics.get("macro_f1") is None:
            continue
        cost = per_million(metrics.get("cost_per_1k_usd"))
        points.append({
            "name": name,
            "label": display_name(name, ctx),
            "f1": float(metrics["macro_f1"]),
            "usd": None if cost is None else float(cost),
            "p95": float(metrics.get("latency_p95_ms") or 0.0),
            "p50": metrics.get("latency_p50_ms"),
            "refusal": metrics.get("refusal_rate"),
            "role": role_of(name),
            "n": metrics.get("n") or 0,
        })
    if not points:
        return _empty_svg(title, "No systems with macro-F1", kind)
    width, height = 980, 580
    left, right, top, bottom = 78, 230, 72, 96
    plot_r = width - right
    plot_b = height - bottom
    positive = [p["usd"] for p in points if p["usd"] is not None and p["usd"] > 0]
    f1s = [p["f1"] for p in points]
    p95s = [p["p95"] for p in points]
    span = max(0.10, (max(f1s) - min(f1s)) * 1.6)
    mid = (max(f1s) + min(f1s)) / 2
    ymin = max(0.0, mid - span / 2)
    ymax = min(1.0, mid + span / 2)
    if ymax - ymin < span - 1e-9:
        if ymin <= 0:
            ymax = min(1.0, span)
        if ymax >= 1:
            ymin = max(0.0, 1.0 - span)
    if not positive:
        log_min, log_max = 0.0, 1.0
        use_log = False
    else:
        log_min = math.log10(min(positive))
        log_max = math.log10(max(positive))
        pad = max(0.2, (log_max - log_min) * 0.12)
        log_min -= pad
        log_max += pad
        use_log = True

    def x_of(usd: float | None) -> float:
        if not use_log:
            return (left + plot_r) / 2
        if usd is None or usd <= 0:
            return left + 8
        return left + (math.log10(usd) - log_min) / (log_max - log_min) * (plot_r - left)

    def y_of(f1: float) -> float:
        if ymax == ymin:
            return (top + plot_b) / 2
        return plot_b - (f1 - ymin) / (ymax - ymin) * (plot_b - top)

    pmin, pmax = min(p95s), max(p95s)

    def radius(p95: float) -> float:
        if pmax == pmin:
            return 16.0
        return 8.0 + 22.0 * (p95 - pmin) / (pmax - pmin)

    body = _svg_header(width, height, title, kind)
    body.append(
        f'<text x="{(left + plot_r) / 2:.0f}" y="{plot_b + 42}" text-anchor="middle" '
        f'font-size="13" font-family="sans-serif">$ / 1M messages'
        f'{" (log scale)" if use_log else ""}</text>'
    )
    body.append(
        f'<text x="22" y="{(top + plot_b) / 2:.0f}" transform="rotate(-90 22 {(top + plot_b) / 2:.0f})" '
        f'text-anchor="middle" font-size="13" font-family="sans-serif">macro-F1 (refusal = unsafe)</text>'
    )
    body.append(
        f'<line x1="{left}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="#222"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_b}" stroke="#222"/>'
    )
    if use_log:
        exp = math.floor(log_min)
        while exp <= math.ceil(log_max):
            if log_min <= exp <= log_max:
                x = x_of(10 ** exp)
                label = f"${10 ** exp:,.0f}" if exp >= 0 else f"${10 ** exp:.3g}"
                body.append(
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{plot_b}" stroke="#eee"/>'
                    f'<text x="{x:.1f}" y="{plot_b + 18}" text-anchor="middle" font-size="11" '
                    f'font-family="sans-serif">{html.escape(label)}</text>'
                )
            exp += 1
    ticks = 4
    for i in range(ticks + 1):
        value = ymin + (ymax - ymin) * i / ticks
        y = y_of(value)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" stroke="#eee"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif">{value:.2f}</text>'
        )
    shades = nvidia_index_map([p["name"] for p in points])
    fshades = frontier_index_map([p["name"] for p in points])
    placed = []
    for point in points:
        point["cx"], point["cy"], point["r"] = x_of(point["usd"]), y_of(point["f1"]), radius(point["p95"])
        placed.append((point["cx"] - point["r"], point["cy"] - point["r"],
                       point["cx"] + point["r"], point["cy"] + point["r"]))
    # Draw large bubbles first so small ones stay visible.
    labels: list[str] = []
    for point in sorted(points, key=lambda p: p["p95"], reverse=True):
        role = point["role"]
        if role == "nvidia":
            fill, stroke, sw = color_of(point["name"], shades), "#1a1a1a", 1
        elif role == "nvidia-baseline":
            fill, stroke, sw = NVIDIA_BASE_FILL, NVIDIA_BASE_STROKE, 2.5
        elif role == "frontier":
            fill, stroke, sw = color_of(point["name"], shades, fshades), "#1a1a1a", 1
        else:
            fill, stroke, sw = OTHER_BLUE, "#1a1a1a", 1
        cx, cy, rad = point["cx"], point["cy"], point["r"]
        usd_attr = "na" if point["usd"] is None else f'{point["usd"]:.4f}'
        refusal_attr = "na" if point["refusal"] is None else f'{float(point["refusal"]):.4f}'
        if point["usd"] is None:
            money = "n/a"
        elif point["usd"] == 0:
            money = "$0"
        else:
            money = "$" + fmt_money0(point["usd"])
        body.append(
            f'<circle data-system="{html.escape(point["name"])}" data-role="{role}" '
            f'data-f1="{point["f1"]:.4f}" data-usd-per-1m="{usd_attr}" data-p95="{point["p95"]:.1f}" '
            f'data-refusal="{refusal_attr}" data-n="{point["n"]}" '
            f'cx="{cx:.1f}" cy="{cy:.1f}" r="{rad:.1f}" fill="{fill}" fill-opacity="0.88" '
            f'stroke="{stroke}" stroke-width="{sw}">'
            f'<title>{html.escape(point["label"])}: macro-F1 {point["f1"]:.4f}, '
            f'$/1M {fmt_money(point["usd"])}, p95 {point["p95"]:.1f} ms ({kind})</title></circle>'
        )
        lines = [(point["label"], 12, "#1a1a1a", "700")]
        lines.append((f"F1 {point['f1'] * 100:.1f} · {money}/1M · p95 {point['p95']:,.0f} ms", 11, "#333", "400"))
        if role == "frontier" and point["refusal"] is not None:
            lines.append((f"refused {float(point['refusal']) * 100:.1f}% (counted unsafe)", 11, REFUSAL_RED, "700"))
        labels.append((point, lines))
    # Greedy label placement: right, left, above, below — first spot that collides with nothing.
    taken = list(placed)
    for point, lines in sorted(labels, key=lambda item: item[0]["cy"]):
        cx, cy, rad = point["cx"], point["cy"], point["r"]
        w = max(_text_w(t, s) for t, s, _c, _f in lines)
        h = 14 * len(lines)
        candidates = [
            ("start", cx + rad + 6, cy - h / 2 + 10),
            ("end", cx - rad - 6, cy - h / 2 + 10),
            ("middle", cx, cy - rad - h + 6),
            ("middle", cx, cy + rad + 16),
            ("start", cx + rad + 6, cy - rad - h + 4),
            ("start", cx + rad + 6, cy + rad + 4),
        ]
        best = None
        for anchor, tx, ty in candidates:
            x0 = tx if anchor == "start" else tx - w if anchor == "end" else tx - w / 2
            rect = (x0, ty - 11, x0 + w, ty - 11 + h)
            inside = rect[0] >= left and rect[2] <= width - 8 and rect[1] >= top - 30 and rect[3] <= plot_b + 30
            if inside and not any(_overlaps(rect, t) for t in taken):
                best = (anchor, tx, ty, rect)
                break
        if best is None:
            anchor, tx, ty = candidates[0]
            x0 = tx
            best = (anchor, tx, ty, (x0, ty - 11, x0 + w, ty - 11 + h))
        anchor, tx, ty, rect = best
        taken.append(rect)
        for i, (text, size, color, weight) in enumerate(lines):
            body.append(
                f'<text x="{tx:.1f}" y="{ty + i * 14:.1f}" text-anchor="{anchor}" font-size="{size}" '
                f'font-weight="{weight}" font-family="sans-serif" fill="{color}">{html.escape(text)}</text>'
            )
    legend_x = width - right + 24
    body.append(
        f'<circle cx="{legend_x + 7}" cy="78" r="7" fill="{NVIDIA_GREEN}" stroke="#1a1a1a"/>'
        f'<text x="{legend_x + 20}" y="82" font-size="12" font-family="sans-serif">NVIDIA system (ours)</text>'
        f'<circle cx="{legend_x + 7}" cy="100" r="7" fill="{NVIDIA_BASE_FILL}" stroke="{NVIDIA_BASE_STROKE}" stroke-width="2.5"/>'
        f'<text x="{legend_x + 20}" y="104" font-size="12" font-family="sans-serif">untuned NVIDIA baseline</text>'
        f'<circle cx="{legend_x + 7}" cy="122" r="7" fill="{FRONTIER_GREY}" stroke="#1a1a1a"/>'
        f'<text x="{legend_x + 20}" y="126" font-size="12" font-family="sans-serif">closed frontier</text>'
        f'<circle cx="{legend_x + 7}" cy="144" r="7" fill="{OTHER_BLUE}" stroke="#1a1a1a"/>'
        f'<text x="{legend_x + 20}" y="148" font-size="12" font-family="sans-serif">other baseline</text>'
        f'<text x="{legend_x}" y="178" font-size="12" font-family="sans-serif">Bubble size = p95 latency</text>'
        f'<text x="{legend_x}" y="196" font-size="12" font-family="sans-serif">Up + left = better</text>'
    )
    footer = textwrap.wrap(f"{scope} {kind}. $ includes allocated dedicated-GPU time and billed refusals.", 150)
    for i, text in enumerate(footer):
        body.append(
            f'<text x="{left}" y="{height - 14 - 15 * (len(footer) - 1 - i)}" font-size="11.5" '
            f'font-family="sans-serif" fill="#333">{html.escape(text)}</text>'
        )
    body.append("</svg>")
    return "\n".join(body)


def disguise_svg(
    names: list[str],
    types: list[str],
    series: dict[str, dict[str, dict[str, Any]]],
    kind: str,
    ctx: RunContext | None = None,
    scope: str = "",
) -> str:
    ctx = ctx or RunContext()
    title = f"Unsafe recall by disguise — {kind}"
    if not types or not names:
        return _empty_svg(title, "No disguise rows in this run", kind)
    shades = nvidia_index_map(names)
    fshades = frontier_index_map(names)
    has_refusals = any(
        (series.get(n, {}).get(p) or {}).get("refusal_rate")
        for n in names if is_frontier(n) for p in types
    )
    width = max(980, 160 + 118 * len(types))
    height = 560
    left, right, top, bottom = 72, 250, 78, 130
    plot_r = width - right
    plot_b = height - bottom
    slot = (plot_r - left) / len(types)
    bar_w = min(26.0, slot * 0.84 / max(1, len(names)))
    body = _svg_header(width, height, title, kind)
    body += [
        '<defs><pattern id="refusal-hatch" patternUnits="userSpaceOnUse" width="6" height="6" '
        'patternTransform="rotate(45)"><rect width="6" height="6" fill="#ffffff"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{REFUSAL_RED}" stroke-width="3"/></pattern></defs>',
        f'<text x="20" y="{(top + plot_b) / 2:.0f}" transform="rotate(-90 20 {(top + plot_b) / 2:.0f})" '
        f'text-anchor="middle" font-size="13" font-family="sans-serif">unsafe recall</text>',
        f'<line x1="{left}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_b}" stroke="#222"/>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        y = plot_b - tick * (plot_b - top)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" stroke="#eee"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif">{tick:.2f}</text>'
        )
    for i, pert in enumerate(types):
        gx = left + i * slot
        counts = [int(series[name][pert]["n"]) for name in names if pert in series.get(name, {})]
        if not counts:
            tick = pert
        elif len(set(counts)) == 1:
            tick = f"{pert} (n={counts[0]})"
        else:
            tick = f"{pert} (n={min(counts)}–{max(counts)})"
        body.append(
            f'<text x="{gx + slot / 2:.1f}" y="{plot_b + 18}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif" transform="rotate(-32 {gx + slot / 2:.1f} {plot_b + 18})">'
            f'{html.escape(tick)}</text>'
        )
        for j, name in enumerate(names):
            info = series.get(name, {}).get(pert)
            recall = None if info is None else info.get("unsafe_recall")
            n = 0 if info is None else int(info.get("n") or 0)
            value = 0.0 if recall is None else float(recall)
            bh = max(0.0, value) * (plot_b - top)
            bx = gx + (slot - len(names) * bar_w) / 2 + j * bar_w
            draw_h = bh if bh > 0 else 1.0
            by = plot_b - draw_h
            fill = color_of(name, shades, fshades)
            stroke = f' stroke="{NVIDIA_BASE_STROKE}" stroke-width="1.5"' if role_of(name) == "nvidia-baseline" else ""
            recall_attr = "na" if recall is None else f"{float(recall):.4f}"
            refusal = None if info is None else info.get("refusal_rate")
            refusal_attr = "na" if refusal is None else f"{float(refusal):.4f}"
            body.append(
                f'<rect data-system="{html.escape(name)}" data-disguise="{html.escape(pert)}" '
                f'data-recall="{recall_attr}" data-refusal="{refusal_attr}" data-n="{n}" x="{bx:.1f}" y="{by:.1f}" '
                f'width="{max(bar_w - 2, 1):.1f}" height="{draw_h:.1f}" fill="{fill}"{stroke}>'
                f'<title>{html.escape(display_name(name, ctx))} / {html.escape(pert)}: unsafe recall {recall_attr} '
                f'(n={n}, {kind})</title></rect>'
            )
            share = None if info is None else info.get("refusal_share_of_unsafe")
            if is_frontier(name) and share:
                hh = float(share) * (plot_b - top)
                body.append(
                    f'<rect data-refusal-part="{html.escape(name)}" x="{bx:.1f}" y="{by:.1f}" '
                    f'width="{max(bar_w - 2, 1):.1f}" height="{min(hh, draw_h):.1f}" '
                    f'fill="url(#refusal-hatch)" stroke="{REFUSAL_RED}" stroke-width="0.8"/>'
                )
            if is_frontier(name) and refusal and float(refusal) >= 0.05:
                body.append(
                    f'<text x="{bx + (bar_w - 2) / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9.5" '
                    f'font-weight="700" fill="{REFUSAL_RED}" font-family="sans-serif">{float(refusal) * 100:.0f}%R</text>'
                )
    legend_x = width - right + 16
    for j, name in enumerate(names):
        y = 78 + j * 22
        stroke = f' stroke="{NVIDIA_BASE_STROKE}" stroke-width="1.5"' if role_of(name) == "nvidia-baseline" else ""
        body.append(
            f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{color_of(name, shades, fshades)}"{stroke}/>'
            f'<text x="{legend_x + 18}" y="{y}" font-size="12" font-family="sans-serif">'
            f'{html.escape(display_name(name, ctx))}</text>'
        )
    if has_refusals:
        y = 78 + len(names) * 22 + 10
        body.append(
            f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="url(#refusal-hatch)" stroke="{REFUSAL_RED}"/>'
            f'<text x="{legend_x + 18}" y="{y}" font-size="12" font-family="sans-serif">part of recall that is</text>'
            f'<text x="{legend_x + 18}" y="{y + 15}" font-size="12" font-family="sans-serif">a provider refusal</text>'
            f'<text x="{legend_x}" y="{y + 38}" font-size="12" font-weight="700" fill="{REFUSAL_RED}" '
            f'font-family="sans-serif">NN%R</text>'
            f'<text x="{legend_x + 40}" y="{y + 38}" font-size="12" font-family="sans-serif">= frontier refusal rate</text>'
            f'<text x="{legend_x}" y="{y + 54}" font-size="12" font-family="sans-serif">(all rows of that disguise)</text>'
        )
    control = " none = clean control inside the disguise file." if "none" in types else ""
    refusal_note = " Frontier refusals count as unsafe (a block, no verdict)." if has_refusals else ""
    body.append(
        f'<text x="{left}" y="{height - 30}" font-size="12" font-family="sans-serif" fill="#333">'
        f'Y-axis fixed at 0–1. {html.escape(kind)}.{html.escape(control)}{html.escape(refusal_note)}</text>'
    )
    if scope:
        body.append(
            f'<text x="{left}" y="{height - 12}" font-size="12" font-family="sans-serif" fill="#333">'
            f'{html.escape(scope)}</text>'
        )
    body.append("</svg>")
    return "\n".join(body)


def _slot_metrics(
    slot_names: tuple[str, ...],
    overall: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None, str]:
    present = [name for name in slot_names if name in overall and overall[name].get("macro_f1") is not None]
    if not present:
        return None, None, ""
    # Prefer `full` when both submission names were measured; keep the other number in a note.
    chosen = "full" if "full" in present else present[0]
    note = ""
    if chosen == "full" and "+escalation" in present:
        other = overall["+escalation"]["macro_f1"]
        same = abs(float(other) - float(overall["full"]["macro_f1"])) < 1e-9
        note = (
            "+escalation matches full on macro-F1"
            if same
            else f"+escalation macro-F1 {float(other):.4f} (full is the point drawn)"
        )
    return chosen, overall[chosen], note


def ablation_svg(overall: dict[str, dict[str, Any]], kind: str, ctx: RunContext | None = None) -> str:
    ctx = ctx or RunContext()
    title = f"Ablation ladder — {kind}"
    slots: list[tuple[str, tuple[str, ...], bool]] = [(label, names, True) for label, names in ABLATION_SLOTS]
    slots += [(label, (name,), False) for label, name in CALIBRATED_SLOTS if name in overall]
    resolved = [_slot_metrics(slot_names, overall) for _label, slot_names, _ladder in slots]
    if not any(metrics is not None for _name, metrics, _note in resolved):
        return _empty_svg(title, "No ablation systems in this run", kind)
    baselines = [n for n in overall if is_nvidia_baseline(n) and overall[n].get("macro_f1") is not None]
    measured_f1 = [float(metrics["macro_f1"]) for _n, metrics, _note in resolved if metrics is not None]
    span_vals = measured_f1 + [float(overall[n]["macro_f1"]) for n in baselines]
    width, height = 940, 520
    left, right, top, bottom = 78, 36, 78, 148
    plot_r = width - right
    plot_b = height - bottom
    span = max(0.10, (max(span_vals) - min(span_vals)) * 1.8)
    mid = (max(span_vals) + min(span_vals)) / 2
    ymin = max(0.0, mid - span / 2)
    ymax = min(1.0, mid + span / 2)
    if ymax >= 1:
        ymin = max(0.0, 1.0 - span)
    if ymin <= 0:
        ymax = min(1.0, span)
    slot_w = (plot_r - left) / len(slots)

    def x_of(i: int) -> float:
        return left + (i + 0.5) * slot_w

    def y_of(f1: float) -> float:
        if ymax == ymin:
            return (top + plot_b) / 2
        return plot_b - (f1 - ymin) / (ymax - ymin) * (plot_b - top)

    body = _svg_header(width, height, title, kind)
    body += [
        f'<text x="22" y="{(top + plot_b) / 2:.0f}" transform="rotate(-90 22 {(top + plot_b) / 2:.0f})" '
        f'text-anchor="middle" font-size="13" font-family="sans-serif">macro-F1 (all rows)</text>',
        f'<line x1="{left}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_b}" stroke="#222"/>',
    ]
    for i in range(5):
        value = ymin + (ymax - ymin) * i / 4
        y = y_of(value)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" stroke="#eee"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif">{value:.2f}</text>'
        )
    for name in baselines:
        f1 = float(overall[name]["macro_f1"])
        y = y_of(f1)
        cost = per_million(overall[name].get("cost_per_1k_usd"))
        body.append(
            f'<line data-baseline="{html.escape(name)}" x1="{left}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" '
            f'stroke="{NVIDIA_BASE_STROKE}" stroke-dasharray="6 4" stroke-width="1.5"/>'
            f'<text x="{plot_r - 4}" y="{y - 6:.1f}" text-anchor="end" font-size="11.5" fill="{NVIDIA_BASE_STROKE}" '
            f'font-family="sans-serif">{html.escape(display_name(name, ctx))}: {f1:.4f}'
            f'{"" if cost is None else " · $" + fmt_money0(cost) + "/1M"}</text>'
        )
    if any(not ladder for _l, _n, ladder in slots):
        first_cal = next(i for i, (_l, _n, ladder) in enumerate(slots) if not ladder)
        xsep = left + first_cal * slot_w
        body.append(
            f'<line x1="{xsep:.1f}" y1="{top}" x2="{xsep:.1f}" y2="{plot_b}" stroke="#bbb" stroke-dasharray="3 3"/>'
            f'<text x="{xsep + 6:.1f}" y="{top + 12}" font-size="11" fill="#555" font-family="sans-serif">'
            f'calibrated variants</text>'
        )
    # Step segments only between adjacent measured ladder slots — a gap is not a measurement.
    previous: tuple[int, float] | None = None
    for i, ((label, _names, ladder), (chosen, metrics, _note)) in enumerate(zip(slots, resolved)):
        if not ladder:
            break
        if metrics is None or previous is None:
            if metrics is not None:
                previous = (i, float(metrics["macro_f1"]))
            continue
        if i == previous[0] + 1:
            x1, y1 = x_of(previous[0]), y_of(previous[1])
            x2, y2 = x_of(i), y_of(float(metrics["macro_f1"]))
            body.append(
                f'<polyline fill="none" stroke="{NVIDIA_GREEN}" stroke-width="2.5" '
                f'points="{x1:.1f},{y1:.1f} {x2:.1f},{y1:.1f} {x2:.1f},{y2:.1f}"/>'
            )
        previous = (i, float(metrics["macro_f1"]))
    notes = []
    for i, ((label, _names, ladder), (chosen, metrics, note)) in enumerate(zip(slots, resolved)):
        x = x_of(i)
        measured = "yes" if metrics is not None else "no"
        body.append(
            f'<text data-slot="{html.escape(label)}" data-measured="{measured}" '
            f'x="{x:.1f}" y="{plot_b + 22}" text-anchor="middle" font-size="12" '
            f'font-family="sans-serif">{html.escape(label)}</text>'
        )
        if metrics is None:
            body.append(
                f'<text x="{x:.1f}" y="{plot_b + 40}" text-anchor="middle" font-size="11" '
                f'fill="#777" font-family="sans-serif">not in this run</text>'
            )
            continue
        f1 = float(metrics["macro_f1"])
        y = y_of(f1)
        fill = NVIDIA_GREEN if ladder else "#ffffff"
        body.append(
            f'<circle data-slot="{html.escape(label)}" data-system="{html.escape(chosen or "")}" '
            f'data-f1="{f1:.4f}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{fill}" stroke="{NVIDIA_GREEN if not ladder else "#1a1a1a"}" '
            f'stroke-width="{2.5 if not ladder else 1}"/>'
            f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-size="12" '
            f'font-family="sans-serif">{f1:.4f}</text>'
        )
        detail = []
        if metrics.get("over_block_rate") is not None:
            detail.append(f"over-block {float(metrics['over_block_rate']) * 100:.1f}%")
        if metrics.get("unsafe_recall") is not None:
            detail.append(f"recall {float(metrics['unsafe_recall']) * 100:.1f}%")
        cost = per_million(metrics.get("cost_per_1k_usd"))
        p95 = metrics.get("latency_p95_ms")
        detail2 = []
        if cost is not None:
            detail2.append(f"${fmt_money0(cost)}/1M")
        if p95 is not None:
            detail2.append(f"p95 {float(p95):,.0f} ms")
        for k, text in enumerate((*detail, " · ".join(detail2))):
            if text:
                body.append(
                    f'<text x="{x:.1f}" y="{plot_b + 40 + k * 15}" text-anchor="middle" font-size="10.5" '
                    f'fill="#444" font-family="sans-serif">{html.escape(text)}</text>'
                )
        if note:
            notes.append(note)
    if len(set(round(v, 6) for v in measured_f1)) == 1:
        notes.append("Measured macro-F1 is the same on every step present in this run.")
    if notes:
        body.append(
            f'<text x="{left}" y="{height - 14}" font-size="12" font-family="sans-serif" fill="#333">'
            f'{html.escape(" ".join(notes))}</text>'
        )
    body.append("</svg>")
    return "\n".join(body)


# ------------------------------------------------------------------ vision panel


VISION_ARMS = (("vision-text-only", "text only (no image)"), ("vision-routed", "routed to Super VL (image)"))
VISION_SOURCES = (
    ("ALL", "All image rows"),
    ("siuo", "SIUO (harm only in text+image)"),
    ("hatemojicheck-rendered", "HatemojiCheck rendered"),
    ("nvidia-nemotron-cs-3.5", "NVIDIA Nemotron CS 3.5 images"),
)


def load_vision(vision_dir: Path) -> dict[str, Any] | None:
    """Per-source recall/over-block for the image A/B, from receipts + summary.json."""
    if not vision_dir or not vision_dir.is_dir():
        return None
    info, summary, receipts = load_run(vision_dir)
    if not all(arm in receipts for arm, _ in VISION_ARMS):
        return None
    kind, banner = classify_provenance(info, summary)
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for src, _label in VISION_SOURCES:
        groups[src] = {}
        for arm, _ in VISION_ARMS:
            rows = receipts[arm] if src == "ALL" else [r for r in receipts[arm] if r.get("source") == src]
            if rows:
                groups[src][arm] = evaluate(rows, groups=False)
    models = {arm: system_model(receipts[arm]) for arm, _ in VISION_ARMS}
    paired = ((summary or {}).get("paired") or {}).get("unsafe_recall") or {}
    return {"kind": kind, "banner": banner, "groups": groups, "models": models,
            "paired": paired, "dir": vision_dir,
            "billed": (summary or {}).get("deployment_billed_usd") or {},
            "wall": (summary or {}).get("wall_seconds") or {}}


def vision_svg(vision: dict[str, Any]) -> str:
    kind = vision["kind"]
    title = f"Image route: NVIDIA Nemotron 3.5 Super VL vs text-only — {kind}"
    groups = [(src, label) for src, label in VISION_SOURCES if vision["groups"].get(src)]
    width, height = 980, 520
    left, right, top, bottom = 72, 250, 78, 150
    plot_r = width - right
    plot_b = height - bottom
    slot = (plot_r - left) / max(1, len(groups))
    bar_w = min(44.0, slot * 0.34)
    colors = {"vision-text-only": NVIDIA_BASE_FILL, "vision-routed": NVIDIA_GREEN}
    body = _svg_header(width, height, title, kind)
    body += [
        f'<text x="20" y="{(top + plot_b) / 2:.0f}" transform="rotate(-90 20 {(top + plot_b) / 2:.0f})" '
        f'text-anchor="middle" font-size="13" font-family="sans-serif">harmful caught (unsafe recall)</text>',
        f'<line x1="{left}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_b}" stroke="#222"/>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        y = plot_b - tick * (plot_b - top)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{plot_r}" y2="{y:.1f}" stroke="#eee"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif">{tick:.2f}</text>'
        )
    for i, (src, label) in enumerate(groups):
        gx = left + i * slot
        arms = vision["groups"][src]
        n = next(iter(arms.values()))["n"]
        for j, (arm, _arm_label) in enumerate(VISION_ARMS):
            m = arms.get(arm)
            if not m:
                continue
            recall = m.get("unsafe_recall") or 0.0
            bh = max(1.0, float(recall) * (plot_b - top))
            bx = gx + slot / 2 - bar_w + j * bar_w
            stroke = f' stroke="{NVIDIA_BASE_STROKE}" stroke-width="1.5"' if arm == "vision-text-only" else ""
            body.append(
                f'<rect data-vision-arm="{arm}" data-source="{html.escape(src)}" data-recall="{float(recall):.4f}" '
                f'x="{bx:.1f}" y="{plot_b - bh:.1f}" width="{bar_w - 3:.1f}" height="{bh:.1f}" '
                f'fill="{colors[arm]}"{stroke}/>'
                f'<text x="{bx + (bar_w - 3) / 2:.1f}" y="{plot_b - bh - 5:.1f}" text-anchor="middle" '
                f'font-size="11" font-weight="700" font-family="sans-serif">{float(recall) * 100:.1f}%</text>'
            )
        cx = gx + slot / 2
        body.append(
            f'<text x="{cx:.1f}" y="{plot_b + 18}" text-anchor="middle" font-size="11.5" font-weight="700" '
            f'font-family="sans-serif">{html.escape(label)}</text>'
            f'<text x="{cx:.1f}" y="{plot_b + 33}" text-anchor="middle" font-size="11" '
            f'font-family="sans-serif" fill="#444">n={n:,}</text>'
        )
        ob = [arms.get(arm, {}).get("over_block_rate") for arm, _ in VISION_ARMS]
        ob_text = ("over-block " + " → ".join("n/a" if v is None else f"{v * 100:.1f}%" for v in ob)
                   if any(v is not None for v in ob) else "no safe rows")
        body.append(
            f'<text x="{cx:.1f}" y="{plot_b + 48}" text-anchor="middle" font-size="11" '
            f'font-family="sans-serif" fill="#444">{html.escape(ob_text)}</text>'
        )
    legend_x = width - right + 16
    for j, (arm, arm_label) in enumerate(VISION_ARMS):
        y = 82 + j * 40
        stroke = f' stroke="{NVIDIA_BASE_STROKE}" stroke-width="1.5"' if arm == "vision-text-only" else ""
        body.append(
            f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{colors[arm]}"{stroke}/>'
            f'<text x="{legend_x + 18}" y="{y}" font-size="12" font-family="sans-serif">{html.escape(arm_label)}</text>'
            f'<text x="{legend_x + 18}" y="{y + 15}" font-size="10.5" fill="#444" font-family="sans-serif">'
            f'{html.escape(short_model(vision["models"].get(arm, "")))}</text>'
        )
    paired = vision.get("paired") or {}
    lines = []
    if paired.get("difference") is not None:
        lo, hi = (paired.get("ci95") or [None, None])[:2]
        ci = "" if lo is None or hi is None else f", 95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]"
        pval = paired.get("mcnemar_p")
        lines.append(
            f"Paired on the same {paired.get('n_paired', 0):,} rows: unsafe recall "
            f"{paired['difference'] * 100:+.1f} pts{ci}"
            + ("" if pval is None else f", McNemar p = {pval:.5f}") + "."
        )
    lines.append("Over-block goes up with the image route (precision cost); see VISION_SUMMARY.md.")
    for k, text in enumerate(lines):
        body.append(
            f'<text x="{left}" y="{height - 46 + k * 16}" font-size="12" font-family="sans-serif" fill="#333">'
            f'{html.escape(text)}</text>'
        )
    body.append(
        f'<text x="{left}" y="{height - 12}" font-size="11" font-family="sans-serif" fill="#555">'
        f'{html.escape(vision["banner"])} · receipts in {html.escape(vision["dir"].name)}/</text>'
    )
    body.append("</svg>")
    return "\n".join(body)


def vision_markdown(vision: dict[str, Any]) -> str:
    lines = [
        f"## Image route ({vision['kind']}) — `{vision['dir'].name}`",
        "",
        f"> {vision['banner']}",
        "",
        "| Slice | n | arm | model | unsafe recall % | over-block % | macro-F1 | p50 ms | p95 ms |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for src, label in VISION_SOURCES:
        for arm, arm_label in VISION_ARMS:
            m = vision["groups"].get(src, {}).get(arm)
            if not m:
                continue
            lines.append(
                f"| {label} | {m['n']:,} | {arm_label} | {short_model(vision['models'].get(arm, ''))} | "
                f"{fmt_pct(m.get('unsafe_recall'))} | {fmt_pct(m.get('over_block_rate'))} | "
                f"{fmt_f1_100(m.get('macro_f1'))} | {fmt_ms0(m.get('latency_p50_ms'))} | "
                f"{fmt_ms0(m.get('latency_p95_ms'))} |"
            )
    paired = vision.get("paired") or {}
    if paired.get("difference") is not None:
        lo, hi = (paired.get("ci95") or [0, 0])[:2]
        lines += ["", f"Paired (routed − text-only) unsafe recall {paired['difference'] * 100:+.1f} pts, "
                      f"95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}], McNemar p = {paired.get('mcnemar_p', 0):.5f}, "
                      f"n = {paired.get('n_paired', 0):,}."]
    lines += ["", "Cost: see VISION_SUMMARY.md (deployment-time billing is authoritative; per-verdict cost_usd "
                  "on the image arm is a serial wall-clock upper bound, so no $/1M is charted here).", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ tables


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def render_markdown(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    header = "| " + " | ".join(columns) + " |"
    align = "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, align]
    keys = (
        "dataset", "system", "n", "macro_f1", "unsafe_recall", "over_block_rate",
        "p50", "p95", "usd_per_1m", "published",
    )
    formatters = {
        "macro_f1": fmt_f1_100,
        "unsafe_recall": fmt_rate,
        "over_block_rate": fmt_rate,
        "p50": fmt_ms,
        "p95": fmt_ms,
        "usd_per_1m": fmt_money,
    }
    for row in rows:
        cells = []
        for key in keys:
            if key in formatters:
                cells.append(formatters[key](row[key]))
            else:
                cells.append(_md_cell(row[key]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_html_table(rows: list[dict[str, Any]]) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in HEADLINE_COLUMNS)
    body = []
    keys = (
        "dataset", "system", "n", "macro_f1", "unsafe_recall", "over_block_rate",
        "p50", "p95", "usd_per_1m", "published",
    )
    formatters = {
        "macro_f1": fmt_f1_100,
        "unsafe_recall": fmt_rate,
        "over_block_rate": fmt_rate,
        "p50": fmt_ms,
        "p95": fmt_ms,
        "usd_per_1m": fmt_money,
    }
    for row in rows:
        cells = []
        for key in keys:
            text = formatters[key](row[key]) if key in formatters else str(row[key])
            cls = ' class="published"' if key == "published" else ""
            cells.append(f"<td{cls}>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def matched_rows(
    cov: Coverage,
    receipts: dict[str, list[dict[str, Any]]],
    ctx: RunContext,
) -> list[dict[str, Any]]:
    """Slice x system on exactly the matched ids. Every row of one slice has the same n."""
    out: list[dict[str, Any]] = []
    slices: list[tuple[str, str | None, str]] = [("ALL matched rows", None, "ALL")]
    slices += [(label, source, label) for label, source in DATASETS]
    for title, source, key in slices:
        for name in cov.matched_systems:
            rows = restrict(receipts.get(name) or [], cov.matched_ids)
            if key != "ALL":
                rows = dataset_slice(rows, key, source)
            metrics = _score_system(name, rows, ctx)
            if metrics is None:
                continue
            out.append({"slice": title, "system": name, "model": short_model(ctx.models.get(name, "")),
                        "metrics": metrics})
    return out


def _matched_cells(row: dict[str, Any]) -> list[str]:
    m = row["metrics"]
    refusal = m.get("refusal_rate")
    return [
        row["slice"], row["system"], row["model"], f"{m['n']:,}",
        fmt_f1_100(m.get("macro_f1")), fmt_pct(m.get("unsafe_recall")), fmt_pct(m.get("over_block_rate")),
        "–" if not is_frontier(row["system"]) and not refusal else fmt_pct(refusal),
        fmt_ms0(m.get("latency_p50_ms")), fmt_ms0(m.get("latency_p95_ms")),
        fmt_money0(per_million(m.get("cost_per_1k_usd"))),
    ]


def render_matched_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(MATCHED_COLUMNS) + " |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(c) for c in _matched_cells(row)) + " |")
    return "\n".join(lines)


def render_matched_html(rows: list[dict[str, Any]]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in MATCHED_COLUMNS)
    body = []
    last = None
    for row in rows:
        cls = ' class="first"' if row["slice"] != last else ""
        last = row["slice"]
        role = role_of(row["system"])
        body.append(f'<tr{cls} data-role="{role}">' + "".join(
            f"<td>{html.escape(c)}</td>" for c in _matched_cells(row)) + "</tr>")
    return f"<table class=\"matched\"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_overall_markdown(overall: dict[str, dict[str, Any]], names: list[str]) -> str:
    lines = [
        "| System | n | macro-F1 | unsafe recall | over-block rate | p50 ms | p95 ms | $ / 1M msgs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        metrics = overall.get(name)
        if not metrics:
            continue
        lines.append(
            "| "
            + " | ".join((
                _md_cell(name),
                str(metrics.get("n") or 0),
                fmt_f1_100(metrics.get("macro_f1")),
                fmt_rate(metrics.get("unsafe_recall")),
                fmt_rate(metrics.get("over_block_rate")),
                fmt_ms(metrics.get("latency_p50_ms")),
                fmt_ms(metrics.get("latency_p95_ms")),
                fmt_money(per_million(metrics.get("cost_per_1k_usd"))),
            ))
            + " |"
        )
    return "\n".join(lines)


def slice_notes() -> str:
    return "\n".join((
        "Slices:",
        "- OpenAI Moderation — source=openai_moderation rows that are not disguise rows.",
        "- WildGuardTest — source=wildguard_test rows that are not disguise rows.",
        "- XSTest over-block — source=xstest rows that are not disguise rows. "
        "Read over-block rate (safe rows predicted unsafe). macro-F1 is secondary. "
        "The published GPT-4 figure is response-level refusal, not this over-block rate.",
        "- Disguised — disguise rows: id prefix rob-, meta.robustness_slice, or "
        "perturbation other than none. Includes clean none controls that belong to the disguise file.",
        "",
        "A disguise row is not also counted in the public dataset of the same source. "
        "$ / 1M msgs = (receipt cost_usd + billed refusal cost + allocated dedicated-GPU time) / rows × 1M; "
        "training is excluded. Provider refusals count as unsafe.",
    ))


def provenance_lines(ctx: RunContext, names: list[str]) -> list[str]:
    out = []
    for name in names:
        mode = ctx.modes.get(name)
        if mode is None:
            continue
        model = ctx.models.get(name, "")
        out.append(f"{name} = {mode.upper()} ({ctx.mode_sources.get(name, '')})"
                   + (f" — model `{model}`" if model else ""))
    return out


@dataclass
class Report:
    kind: str
    banner: str
    rows: list[dict[str, Any]]
    overall: dict[str, dict[str, Any]]
    full_names: list[str]
    names: list[str]
    slice_note: str
    context: str
    run_dir: Path
    ctx: RunContext
    cov: Coverage | None
    matched: list[dict[str, Any]]
    vision: dict[str, Any] | None
    svgs: dict[str, str]
    pngs: dict[str, str] = field(default_factory=dict)


def key_findings(rep: Report) -> list[str]:
    """Plain-language bullets computed from the matched rows (no hand-typed numbers)."""
    if not rep.matched:
        return []
    allm = {r["system"]: r["metrics"] for r in rep.matched if r["slice"] == "ALL matched rows"}
    ours = next((n for n in ("full", "full-cal", "+escalation", "specialist-only") if n in allm), None)
    if not ours:
        return []
    o = allm[ours]
    o_cost = per_million(o.get("cost_per_1k_usd")) or 0.0
    out = []
    for name, m in allm.items():
        if not is_frontier(name):
            continue
        cost = per_million(m.get("cost_per_1k_usd")) or 0.0
        ratio = f"{cost / o_cost:,.0f}× cheaper" if o_cost and cost else "cost n/a"
        p50o, p50f = o.get("latency_p50_ms"), m.get("latency_p50_ms")
        lat = f", p50 {p50o:,.0f} vs {p50f:,.0f} ms" if p50o and p50f else ""
        out.append(
            f"{ours} vs {display_name(name, rep.ctx)}: {ratio} per message{lat}; macro-F1 "
            f"{fmt_f1_100(o.get('macro_f1'))} vs {fmt_f1_100(m.get('macro_f1'))}, unsafe recall "
            f"{fmt_pct(o.get('unsafe_recall'))}% vs {fmt_pct(m.get('unsafe_recall'))}%, over-block "
            f"{fmt_pct(o.get('over_block_rate'))}% vs {fmt_pct(m.get('over_block_rate'))}%"
            + (f"; {display_name(name, rep.ctx)} refused {fmt_pct(m.get('refusal_rate'))}% of rows (counted unsafe)"
               if m.get("refusal_rate") else "")
            + "."
        )
    for cal, base in (("full-cal", "full"), ("specialist-cal", "specialist-only")):
        if cal in allm and base in allm:
            c, b = allm[cal], allm[base]
            delta = (float(c.get("macro_f1") or 0) - float(b.get("macro_f1") or 0)) * 100
            out.append(
                f"Calibration check: {cal} vs {base} macro-F1 {fmt_f1_100(c.get('macro_f1'))} vs "
                f"{fmt_f1_100(b.get('macro_f1'))} ({delta:+.1f}), over-block {fmt_pct(c.get('over_block_rate'))}% vs "
                f"{fmt_pct(b.get('over_block_rate'))}%, unsafe recall {fmt_pct(c.get('unsafe_recall'))}% vs "
                f"{fmt_pct(b.get('unsafe_recall'))}%."
            )
    for name, m in allm.items():
        if is_nvidia_baseline(name):
            out.append(
                f"{ours} vs {display_name(name, rep.ctx)} ({short_model(rep.ctx.models.get(name, ''))}): macro-F1 "
                f"{fmt_f1_100(o.get('macro_f1'))} vs {fmt_f1_100(m.get('macro_f1'))}, $/1M "
                f"{fmt_money0(o_cost)} vs {fmt_money0(per_million(m.get('cost_per_1k_usd')))}."
            )
    return out


def render_headline(rep: Report) -> str:
    kind = rep.kind
    label = {
        "MOCK": "Every metric below is **MOCK** except the published frontier column, which is **PUBLISHED** (not our run).",
        "LIVE": "Metrics below are **LIVE** measurements from this run. The published frontier column is **PUBLISHED** (not our run).",
        "MIXED": "This run mixes modes (**MIXED**). The published frontier column is **PUBLISHED** (not our run).",
        "UNKNOWN": "Provenance is **UNKNOWN** (no MOCK/LIVE banner or modes). The published frontier column is **PUBLISHED** (not our run).",
    }[kind]
    parts = [
        f"# Safety One headline — {kind}",
        "",
        f"> {rep.banner}",
        "",
        label,
        "",
    ]
    prov = provenance_lines(rep.ctx, rep.names)
    if prov:
        parts += ["Per-system provenance:", ""] + [f"- {p}" for p in prov] + [""]
    findings = key_findings(rep)
    if findings:
        parts += ["## Key findings (matched rows, computed)", ""] + [f"- {f}" for f in findings] + [""]
    parts += [slice_notes(), ""]
    for note in rep.ctx.notes:
        parts += [note, ""]
    if rep.slice_note:
        parts += [rep.slice_note, ""]
    parts += [
        "## All rows per dataset",
        "",
        render_markdown(rep.rows, HEADLINE_COLUMNS),
        "",
    ]
    if rep.context:
        parts += [rep.context, ""]
    parts += [
        "Published macro-F1 figures are the 0–100 numbers printed in data/BENCHMARKS.md. "
        "Headline macro-F1 from this run is shown on the same 0–100 scale.",
        "",
    ]
    if rep.matched and rep.cov:
        parts += [
            f"## Matched rows ({len(rep.cov.matched_ids):,} rows every compared system ran)",
            "",
            "Same rows for every system in a slice (frontier arms ran disguised_v1 in full plus a seeded "
            "300-row sample of each public file). Refusals counted as unsafe; refusal % shown.",
            "",
            render_matched_markdown(rep.matched),
            "",
        ]
        if rep.cov.excluded:
            parts += [f"Not in the matched table (missing some matched rows): {', '.join(rep.cov.excluded)}.", ""]
    parts += [
        "## All rows in this run",
        "",
        "Every receipt for systems that ran every row. The ablation ladder uses this table"
        + ("; the tradeoff and disguise charts use the matched rows above." if rep.matched
           else "; so do the tradeoff bubbles."),
        "",
        render_overall_markdown(rep.overall, rep.full_names),
        "",
    ]
    if rep.vision:
        parts += [vision_markdown(rep.vision)]
    charts = ", ".join(f"`{name}`" for name in rep.svgs)
    parts += [
        f"Charts written next to this file: {charts}. Open `index.html` offline.",
        f"Run directory: `{rep.run_dir}`.",
        "",
    ]
    if rep.pngs:
        parts += ["PNG exports: " + ", ".join(f"`{p}`" for p in rep.pngs.values()), ""]
    return "\n".join(parts)


def render_overall_html(overall: dict[str, dict[str, Any]], names: list[str], kind: str) -> str:
    columns = ("System", "n", "macro-F1", "unsafe recall", "over-block rate", "p50 ms", "p95 ms", "$ / 1M msgs")
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for name in names:
        metrics = overall.get(name)
        if not metrics:
            continue
        values = (
            name,
            str(metrics.get("n") or 0),
            fmt_rate(metrics.get("macro_f1")),
            fmt_rate(metrics.get("unsafe_recall")),
            fmt_rate(metrics.get("over_block_rate")),
            fmt_ms(metrics.get("latency_p50_ms")),
            fmt_ms(metrics.get("latency_p95_ms")),
            fmt_money(per_million(metrics.get("cost_per_1k_usd"))),
        )
        body.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    return (
        f'<p>Label: {html.escape(kind)}. Every receipt for systems that ran every row.</p>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


# ------------------------------------------------------------------ PNG export


def _browser() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _svg_size(svg: str) -> tuple[int, int]:
    match = re.search(r'<svg[^>]*width="(\d+)"[^>]*height="(\d+)"', svg)
    return (int(match.group(1)), int(match.group(2))) if match else (960, 540)


def svg_to_png(svg_path: Path, png_path: Path, size: tuple[int, int]) -> bool:
    """Rasterise one SVG with a headless browser (2x scale). False when no tool works."""
    exe = _browser()
    rsvg = shutil.which("rsvg-convert")
    png_path.unlink(missing_ok=True)
    if rsvg:
        subprocess.run([rsvg, "-z", "2", "-o", str(png_path), str(svg_path)],
                       capture_output=True, timeout=60, check=False)
    elif exe:
        with tempfile.TemporaryDirectory(prefix="so-report-chrome-") as profile:
            cmd = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                   "--no-default-browser-check", f"--user-data-dir={profile}",
                   "--force-device-scale-factor=2", f"--window-size={size[0]},{size[1]}",
                   f"--screenshot={png_path}", svg_path.resolve().as_uri()]
            try:
                subprocess.run(cmd, capture_output=True, timeout=90, check=False)
            except subprocess.TimeoutExpired:
                return False
    return png_path.is_file() and png_path.stat().st_size > 0


def export_charts(svgs: dict[str, str], png_dir: Path, prefix: str) -> dict[str, str]:
    """Write <prefix>-<chart>.svg (+ .png when a rasteriser is available) into png_dir."""
    png_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for filename, svg in svgs.items():
        stem = f"{prefix}-{Path(filename).stem}"
        svg_path = png_dir / f"{stem}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        png_path = png_dir / f"{stem}.png"
        out[filename] = str(png_path if svg_to_png(svg_path, png_path, _svg_size(svg)) else svg_path)
    return out


# ------------------------------------------------------------------ build


def build_judge_report(
    run_dir: Path,
    *,
    benchmarks_path: Path | None = None,
    frozen_readme: Path | None = None,
    vision_dir: Path | None = None,
    png_dir: Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    benchmarks_path = benchmarks_path or (ROOT / "data" / "BENCHMARKS.md")
    if frozen_readme is None:
        frozen_readme = ROOT / "data" / "frozen" / "README.md"
    run_info, summary, receipts = load_run(run_dir)
    ctx = load_context(run_dir, run_info, receipts)
    names = system_order(list(receipts) or list(((summary or {}).get("systems") or {})))
    if receipts:
        kind, banner = classify_provenance(run_info, summary, ctx.modes)
        info_modes = set(((run_info or {}).get("modes") or {}))
        extra = [f"{n}={ctx.modes[n]} ({ctx.mode_sources[n]})" for n in names
                 if n in ctx.modes and n not in info_modes]
        if extra and (run_info or {}).get("banner"):
            banner = f"{banner} · also: {', '.join(extra)}"
    else:
        kind, banner = classify_provenance(run_info, summary)
    published = load_published(benchmarks_path, frozen_readme)
    context = published_context(benchmarks_path)
    rows, slice_note = dataset_rows(names, receipts, summary, published, ctx)
    cov = coverage(names, receipts) if receipts else None
    full_names = cov.full if cov and cov.partial else names
    overall = overall_metrics(full_names, receipts, summary, ctx)
    matched: list[dict[str, Any]] = []
    if cov and cov.has_matched:
        matched = matched_rows(cov, receipts, ctx)
        scope_names = cov.matched_systems
        keep = cov.matched_ids
        scope = (f"Matched {len(keep):,} rows — the rows every compared system ran "
                 "(frontier arms: disguised_v1 + 300/public file).")
    else:
        scope_names, keep, scope = names, None, "All rows in the run."
    chart_names = chart_systems(scope_names)
    collapsed = [n for n in scope_names if n not in chart_names]
    if collapsed:
        scope = f"{scope} Not drawn (see ablation chart / tables): {', '.join(collapsed)}."
    chart_overall = overall_metrics(chart_names, receipts, summary, ctx, keep)
    dis_names, dis_keep, dis_scope = disguise_scope(chart_systems(names), receipts)
    types, series = disguise_series(dis_names, receipts, summary, ctx, dis_keep)
    svgs = {
        "tradeoff.svg": tradeoff_svg(chart_overall, chart_names, kind, ctx, scope),
        "disguise.svg": disguise_svg(dis_names, types, series, kind, ctx, dis_scope),
        "ablation.svg": ablation_svg(overall, kind, ctx),
    }
    vision = load_vision(Path(vision_dir)) if vision_dir else None
    if vision:
        svgs["vision.svg"] = vision_svg(vision)
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    for filename, svg in svgs.items():
        (out / filename).write_text(svg, encoding="utf-8")
    rep = Report(kind, banner, rows, overall, full_names, names, slice_note, context, run_dir,
                 ctx, cov, matched, vision, svgs)
    if png_dir:
        rep.pngs = export_charts(svgs, Path(png_dir), run_dir.name)
    (out / "headline.md").write_text(render_headline(rep), encoding="utf-8")
    (out / "index.html").write_text(_assemble_index(rep), encoding="utf-8")
    return out


def _assemble_index(rep: Report) -> str:
    kind = rep.kind
    kind_class = kind.lower()
    figures = [
        ("tradeoff.svg", "Quality vs cost vs latency",
         "One bubble per system. Vertical axis is macro-F1 (provider refusals counted as unsafe). "
         "Horizontal axis is $ per 1M messages, log scale. Bubble area tracks p95 latency. "
         "NVIDIA systems are green (#76B900), the untuned NVIDIA baseline is pale green, closed frontier is grey."
         + (" Scored on the matched rows only." if rep.matched else "")),
        ("disguise.svg", "Unsafe recall by disguise",
         "Grouped bars, one series per system. Clean-up and routing show up here as "
         "higher unsafe recall on attack types. Hatched red = share of a frontier bar that is a "
         "provider refusal rather than a verdict; NN%R = refusal rate on that disguise."),
        ("ablation.svg", "Ablation ladder",
         "Step chart: specialist-only, then +rules, then +router, then +escalation (full), on all rows. "
         "A slot with no point was not in this run. The line connects only adjacent measured steps. "
         "Dashed line = untuned NVIDIA baseline; hollow points = calibrated variants (if present)."),
    ]
    if "vision.svg" in rep.svgs:
        figures.append(("vision.svg", "Image route (Super VL)",
                        "Same image rows, two arms: text only vs routed to NVIDIA Nemotron 3.5 Super VL with the image."))
    blocks = []
    for filename, title, note in figures:
        blocks.append(
            f"<section><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>"
            f"{rep.svgs[filename]}<p class=\"file\">{html.escape(filename)}</p></section>"
        )
    note_html = f"<p class=\"note\">{html.escape(rep.slice_note)}</p>" if rep.slice_note else ""
    note_html += "".join(f"<p class=\"note\">{html.escape(n)}</p>" for n in rep.ctx.notes)
    scale = (
        "Published macro-F1 figures are the 0–100 numbers printed in data/BENCHMARKS.md. "
        "Table macro-F1 from this run is shown on the same 0–100 scale."
    )
    context_html = f"<p>{html.escape(rep.context)}</p>" if rep.context else ""
    context_html += f"<p>{html.escape(scale)}</p>"
    prose = "<br>\n".join(html.escape(line) for line in slice_notes().splitlines())
    prov = provenance_lines(rep.ctx, rep.names)
    prov_html = ("<ul class=\"prov\">" + "".join(f"<li>{html.escape(p)}</li>" for p in prov) + "</ul>") if prov else ""
    findings = key_findings(rep)
    findings_html = ("<section class=\"key\"><h2>Key findings (matched rows, computed from receipts)</h2><ul>"
                     + "".join(f"<li>{html.escape(f)}</li>" for f in findings) + "</ul></section>") if findings else ""
    matched_html = ""
    if rep.matched and rep.cov:
        matched_html = (
            f"<h2>Matched rows — {len(rep.cov.matched_ids):,} rows every compared system ran</h2>"
            "<p>Same rows for every system inside a slice. Frontier refusals counted as unsafe; refusal % shown.</p>"
            f"<div class=\"wrap\">{render_matched_html(rep.matched)}</div>"
        )
        if rep.cov.excluded:
            matched_html += (f"<p class=\"note\">Not in the matched table (missing some matched rows): "
                             f"{html.escape(', '.join(rep.cov.excluded))}.</p>")
    vision_html = ""
    if rep.vision:
        vision_html = f"<p class=\"note\">Image route banner: {html.escape(rep.vision['banner'])}</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safety One report — {html.escape(kind)}</title>
<style>
  body {{ margin: 0; background: #ffffff; color: #1a1a1a;
         font: 16px/1.45 "Segoe UI", sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem 1.25rem 3rem; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 0.75rem; }}
  h2 {{ font-size: 1.25rem; margin: 2rem 0 0.4rem; }}
  .banner {{ padding: 0.9rem 1rem; border: 2px solid #333; margin: 0 0 1rem; }}
  .banner .tag {{ display: inline-block; font-weight: 800; letter-spacing: 0.04em; margin-right: 0.6rem; }}
  .banner.mock {{ background: #fff3cd; border-color: #856404; }}
  .banner.live {{ background: #e8f5e9; border-color: #1b5e20; }}
  .banner.mixed {{ background: #fff3cd; border-color: #856404; }}
  .banner.unknown {{ background: #f3f3f3; border-color: #666; }}
  .key {{ background: #f4f9ec; border-left: 6px solid #76B900; padding: 0.4rem 1rem 0.6rem; }}
  .key h2 {{ margin-top: 0.6rem; }}
  .prov {{ font-size: 0.85rem; color: #333; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.45rem; text-align: right; vertical-align: top; }}
  th:nth-child(-n+2), td:nth-child(-n+2), td.published, th:last-child {{ text-align: left; }}
  table.matched td:nth-child(3), table.matched th:nth-child(3) {{ text-align: left; }}
  table.matched th:last-child {{ text-align: right; }}
  table.matched tr.first td {{ border-top: 2px solid #555; }}
  table.matched tr[data-role="nvidia"] td:nth-child(2) {{ color: #3E7800; font-weight: 700; }}
  td.published {{ background: #f6f6f6; font-size: 0.78rem; }}
  svg {{ width: 100%; height: auto; border: 1px solid #e6e6e6; background: #fff; }}
  .file, .note {{ color: #444; font-size: 0.9rem; }}
</style>
</head>
<body>
<main>
<h1>Safety One — {html.escape(kind)}</h1>
<div class="banner {kind_class}"><span class="tag">{html.escape(kind)}</span> {html.escape(rep.banner)}</div>
{prov_html}
{findings_html}
{matched_html}
<h2>All rows per dataset</h2>
<p>{prose}</p>
{note_html}
<div class="wrap">{render_html_table(rep.rows)}</div>
{context_html}
<h2>All rows in this run</h2>
{render_overall_html(rep.overall, rep.full_names, kind)}
{''.join(blocks)}
{vision_html}
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="results/<run> directory with receipts and RUN_INFO.json")
    parser.add_argument("--benchmarks", type=Path, default=None, help="override data/BENCHMARKS.md")
    parser.add_argument("--frozen-readme", type=Path, default=None, help="override data/frozen/README.md")
    parser.add_argument("--vision", type=Path, default=None,
                        help="results/<vision run> with vision-text-only/vision-routed receipts (adds a panel)")
    parser.add_argument("--png-dir", type=Path, default=None,
                        help="also export <run>-<chart>.svg/.png here (PNG needs headless Chrome/Chromium or rsvg-convert)")
    args = parser.parse_args(argv)
    out = build_judge_report(args.run_dir, benchmarks_path=args.benchmarks, frozen_readme=args.frozen_readme,
                             vision_dir=args.vision, png_dir=args.png_dir)
    print(f"wrote {out}")
    if args.png_dir:
        for path in sorted(Path(args.png_dir).glob(f"{args.run_dir.resolve().name}-*")):
            print(f"  export {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
