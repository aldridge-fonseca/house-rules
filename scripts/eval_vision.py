#!/usr/bin/env python3
"""Image-route A/B eval: text-only path vs routing to the NVIDIA VLM.

Arms (same 658 resolved image rows, same IDs):

- vision-text-only : serverless Nemotron Lightning 3.5 30B
  (accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b,
  reasoning_effort=none) on the row's text/caption ONLY — no image bytes.
- vision-routed    : Nemotron 3.5 Super VL 120B-A12B NVFP4 on an on-demand
  4x B200 deployment (env FIREWORKS_VISION_MODEL =
  accounts/<acct>/deployments/<id>, created by scripts/vision_deploy.sh)
  on text + image_url data URI.

The set is imbalanced (507 unsafe / 151 safe), so headline numbers are
harmful-caught (unsafe recall) and safe-wrongly-blocked (over-block rate),
reported separately and per source. Receipts resume by id+row hash.

    python scripts/eval_vision.py                    # auto: live where possible
    python scripts/eval_vision.py --mode mock        # offline plumbing check
    python scripts/eval_vision.py --mode live        # require keys + deployment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.eval.metrics import evaluate, paired_comparison  # noqa: E402
from safety_one.eval.runner import _hash, _read_cache, run_system  # noqa: E402
from safety_one.providers.mock import MockProvider  # noqa: E402
from safety_one.providers import vision_live  # noqa: E402
from safety_one import registry as _registry  # noqa: E402

DEFAULT_DATA = [
    "data/vision/nvidia_image_eval.jsonl",   # 91 resolved rows
    "data/vision/siuo_eval.jsonl",           # 167
    "data/vision/hatemoji_render_eval.jsonl",  # 400
]
TEXT_ARM = "vision-text-only"
ROUTED_ARM = "vision-routed"


def load_rows(path: Path) -> tuple[list[dict], int]:
    """Eval rows with a resolvable image file; returns (rows, skipped)."""
    rows, skipped = [], 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image = row.get("image_path")
            resolved = Path(image).expanduser() if image else None
            if resolved is not None and not resolved.is_absolute():
                resolved = ROOT / resolved
            if resolved is None or not resolved.is_file():
                skipped += 1  # meta.image_resolved:false — no pixels to send
                continue
            if row.get("gold_label") not in ("safe", "unsafe"):
                raise ValueError(f"{path}:{lineno}: bad gold_label")
            row["image_path"] = str(resolved)  # CWD-independent for providers
            rows.append(row)
    return rows, skipped


def _text_arm_system(live: bool, text_model: str = "lightning", timeout_s: float = 45.0):
    if live:
        # Bounded timeout + fewer retries than the library default: a wedged
        # serverless request should fail the row fast, not stall the arm.
        if text_model == "ultra":
            return vision_live.NemotronLightningTextProvider(
                model=vision_live.ULTRA_MODEL, timeout_s=timeout_s, max_retries=3,
                price_input_per_million=vision_live.ULTRA_IN_PER_1M,
                price_output_per_million=vision_live.ULTRA_OUT_PER_1M)
        return vision_live.NemotronLightningTextProvider(
            model=vision_live.LIGHTNING_MODEL, timeout_s=timeout_s, max_retries=3)
    return MockProvider(name=TEXT_ARM, model=vision_live.LIGHTNING_MODEL + "-sim",
                        seed=20261102, base_latency_ms=60, per_char_ms=0.03,
                        cost_per_call_usd=0.000007, cost_per_1k_chars_usd=0.00005,
                        sharpness=0.9)


def _routed_arm_system(live: bool):
    if live:
        return vision_live.SuperVLVisionProvider(timeout_s=120.0, max_retries=4)
    return MockProvider(name=ROUTED_ARM, model=vision_live.SUPER_VL_MODEL + "-sim",
                        seed=20261103, base_latency_ms=220, per_char_ms=0.05,
                        cost_per_call_usd=0.0008, cost_per_1k_chars_usd=0.0003,
                        image_cost_usd=0.0012, sharpness=1.1)


def load_receipts(path: Path, rows: list[dict]) -> list[dict]:
    """Cached receipts aligned to rows (no model calls); missing rows -> error receipt."""
    cache = _read_cache(path)
    out = []
    for row in rows:
        rec = cache.get((str(row["id"]), _hash(row)))
        out.append(rec or {
            "id": str(row["id"]), "gold_label": row.get("gold_label"),
            "source": row.get("source"), "text": row.get("text", ""),
            "image_path": row.get("image_path"), "lang": row.get("lang"),
            "perturbation": row.get("perturbation") or "none",
            "prediction": {"label": "", "latency_ms": 0.0, "cost_usd": 0.0},
            "error": "MissingReceipt: arm not run for this row"})
    return out


def _wins_losses(routed_rows, text_rows):
    """IDs where routed caught unsafe that text-only missed, and vice versa."""
    text = {r["id"]: r for r in text_rows}
    wins, losses = [], []
    for r in routed_rows:
        t = text.get(r["id"])
        if not t or r.get("gold_label") != "unsafe":
            continue
        rp = (r.get("prediction") or {}).get("label")
        tp = (t.get("prediction") or {}).get("label")
        if rp == "unsafe" and tp != "unsafe":
            wins.append({"id": r["id"], "source": r.get("source"),
                         "text": (r.get("text") or "")[:160]})
        elif tp == "unsafe" and rp != "unsafe":
            losses.append({"id": r["id"], "source": r.get("source"),
                           "text": (r.get("text") or "")[:160]})
    return wins, losses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("auto", "mock", "live"), default="auto",
                    help="auto: each arm live when its requirements are met, else mock")
    ap.add_argument("--data", action="append", default=None,
                    help="JSONL eval file (repeatable); default = the three data/vision files")
    ap.add_argument("--run", default=datetime.now(timezone.utc).strftime("vision-%Y%m%dT%H%M%SZ"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--text-concurrency", type=int, default=16,
                    help="parallel requests for the serverless text arm")
    ap.add_argument("--routed-concurrency", type=int, default=16,
                    help="parallel requests for the Super VL deployment arm")
    ap.add_argument("--max-text-usd", type=float, default=2.0,
                    help="abort if the estimated text-arm spend exceeds this cap")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows of the set")
    ap.add_argument("--retry-errors", action="store_true",
                    help="before running, drop error receipts from this --run dir so "
                         "failed rows are retried (successes stay cached)")
    ap.add_argument("--arms", choices=("both", "routed", "text", "report"), default="both",
                    help="which arms make calls; the others are read from this --run's "
                         "cached receipts (report = no calls at all)")
    ap.add_argument("--text-model", choices=("lightning", "ultra"), default="lightning",
                    help="text-only arm model: serverless Lightning 3.5 30B (default) or "
                         "Nemotron 3 Ultra serverless (fallback if Lightning is wedged)")
    ap.add_argument("--text-timeout", type=float, default=45.0,
                    help="per-request timeout (s) for the serverless text arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    _registry._load_env_file()  # FIREWORKS_* from `.env` at repo root (gitignored)

    data_paths = [(Path(p) if Path(p).is_absolute() else ROOT / p)
                  for p in (args.data or DEFAULT_DATA)]
    rows, skipped = [], 0
    for path in data_paths:
        file_rows, file_skipped = load_rows(path)
        rows += file_rows
        skipped += file_skipped
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        ap.error("no rows with resolved images")

    api_key = os.environ.get("FIREWORKS_API_KEY")
    vision_model = os.environ.get("FIREWORKS_VISION_MODEL", "")
    run_arms = {"both": (TEXT_ARM, ROUTED_ARM), "routed": (ROUTED_ARM,),
                "text": (TEXT_ARM,), "report": ()}[args.arms]
    if args.mode == "live" and run_arms and not api_key:
        ap.error("--mode live requires FIREWORKS_API_KEY")
    if args.mode == "live" and ROUTED_ARM in run_arms and not vision_model:
        ap.error("--mode live requires FIREWORKS_VISION_MODEL "
                 "(accounts/<acct>/deployments/<id>) — run scripts/vision_deploy.sh first")

    text_live = args.mode == "live" or (args.mode == "auto" and bool(api_key))
    routed_live = args.mode == "live" or (
        args.mode == "auto" and bool(api_key) and bool(vision_model))
    modes = {TEXT_ARM: "live" if text_live else "mock",
             ROUTED_ARM: "live" if routed_live else "mock"}
    text_model = vision_live.ULTRA_MODEL if args.text_model == "ultra" else vision_live.LIGHTNING_MODEL
    text_in, text_out = ((vision_live.ULTRA_IN_PER_1M, vision_live.ULTRA_OUT_PER_1M)
                         if args.text_model == "ultra"
                         else (vision_live.LIGHTNING_IN_PER_1M, vision_live.LIGHTNING_OUT_PER_1M))

    # Cost guard: serverless text arm is the only spend we can start without a
    # deployment. Rough upper estimate before spending anything.
    text_estimate = len(rows) * (600 * text_in + 32 * text_out) / 1e6  # ~600 in + 32 out tok
    if text_live and TEXT_ARM in run_arms and text_estimate > args.max_text_usd:
        ap.error(f"estimated text-arm spend ${text_estimate:.4f} exceeds "
                 f"--max-text-usd ${args.max_text_usd:.2f}; trim --data/--limit")
    rate_h = float(os.environ.get("FIREWORKS_VISION_RATE_USD_PER_HOUR",
                                  vision_live.RATE_USD_PER_HOUR))

    run_dir = Path(args.out) / args.run
    if args.retry_errors:
        for arm in run_arms:
            receipt = run_dir / f"{arm}.jsonl"
            if receipt.exists():
                lines = [l for l in receipt.read_text(encoding="utf-8").splitlines()
                         if l.strip()]
                kept = [l for l in lines if not json.loads(l).get("error")]
                dropped = len(lines) - len(kept)
                receipt.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
                if dropped:
                    print(f"{arm}: dropped {dropped} error receipts for retry")
    specs = {
        TEXT_ARM: ("live:vision_live.NemotronLightningTextProvider:"
                   + text_model + ":reasoning_effort=none"
                   if text_live else "mock:text-sim"),
        ROUTED_ARM: ("live:vision_live.SuperVLVisionProvider:" + vision_model
                     if routed_live else "mock:vision-sim"),
    }
    concurrency = {TEXT_ARM: args.text_concurrency, ROUTED_ARM: args.routed_concurrency}

    # Wall seconds persist across partial (--arms) invocations of the same run.
    wall_path = run_dir / "wall_seconds.json"
    wall: dict[str, float] = (json.loads(wall_path.read_text(encoding="utf-8"))
                              if wall_path.exists() else {})
    receipts: dict[str, list[dict]] = {}
    for name in (TEXT_ARM, ROUTED_ARM):
        receipt_path = run_dir / f"{name}.jsonl"
        if name not in run_arms:
            receipts[name] = load_receipts(receipt_path, rows)
            spec = next((r.get("system_spec", "") for r in receipts[name]
                         if r.get("system_spec")), "")
            modes[name] = "live" if spec.startswith("live:") else "mock" if spec else "missing"
            specs[name] = spec
            print(f"{name} [{modes[name]}]: loaded cached receipts (no calls)")
            continue
        system = (_text_arm_system(text_live, args.text_model, args.text_timeout)
                  if name == TEXT_ARM else _routed_arm_system(routed_live))
        started = time.perf_counter()
        receipts[name] = run_system(
            system, rows, receipt_path, system_name=name,
            system_spec=specs[name], concurrency=concurrency[name])
        wall[name] = time.perf_counter() - started
        n_err = sum(1 for r in receipts[name] if r.get("error"))
        print(f"{name} [{modes[name]}]: {len(receipts[name])} rows in "
              f"{wall[name]:.1f}s, {n_err} errors -> {receipt_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    wall_path.write_text(json.dumps(wall, indent=1), encoding="utf-8")

    # Per-verdict cost_usd for the routed arm is a wall-clock upper-bound
    # allocation; the deployment is billed for hot wall time regardless.
    deployment_billed = {
        name: wall[name] * rate_h / 3600.0 for name in (ROUTED_ARM,)
        if modes[name] == "live" and name in wall}

    summaries = {name: evaluate(rs) for name, rs in receipts.items()}
    comparison = paired_comparison(
        receipts[TEXT_ARM], receipts[ROUTED_ARM],
        metric="unsafe_recall", seed=args.seed)
    comparison_f1 = paired_comparison(
        receipts[TEXT_ARM], receipts[ROUTED_ARM],
        metric="unsafe_f1", seed=args.seed)
    wins, losses = _wins_losses(receipts[ROUTED_ARM], receipts[TEXT_ARM])

    tag = "LIVE" if all(m == "live" for m in modes.values()) else \
        "MIXED" if "live" in modes.values() else "MOCK"
    banner = (f"{tag} vision A/B — arms: "
              + ", ".join(f"{k}={v}" for k, v in modes.items()))

    fmt = lambda x: "n/a" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
    lines = [f"> {banner}", "",
             f"Rows: {len(rows)} image-resolved ({skipped} skipped, unresolved), "
             f"{sum(r['gold_label'] == 'unsafe' for r in rows)} unsafe / "
             f"{sum(r['gold_label'] == 'safe' for r in rows)} safe.", "",
             "| Arm | Mode | n | Errors | Harmful caught (unsafe recall) | Unsafe precision | "
             "Safe wrongly blocked (over-block) | Macro F1 | p50 ms | p95 ms | $ total |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in summaries.items():
        n_err = sum(1 for r in receipts[name] if r.get("error"))
        lines.append(
            f"| {name} | {modes[name]} | {m['n']} | {n_err} | {fmt(m['unsafe_recall'])} | "
            f"{fmt(m['unsafe_precision'])} | {fmt(m['over_block_rate'])} | "
            f"{fmt(m['macro_f1'])} | {fmt(m['latency_p50_ms'])} | "
            f"{fmt(m['latency_p95_ms'])} | {fmt(m['cost_total_usd'])} |")
    if any(r.get("error") for rs in receipts.values() for r in rs):
        lines += ["", "> Note: rows with transport/model errors keep `label: \"\"` "
                  "and count as wrong predictions — see `error` in the receipts "
                  "and the Errors column. Rerun the same --run name with "
                  "--retry-errors to retry just the failed rows."]
    lines += ["", "### Per source (unsafe recall = harmful caught; over-block = safe wrongly blocked)", "",
              "| Arm | Source | n | unsafe n | Harmful caught | Safe wrongly blocked |",
              "|---|---|---:|---:|---:|---:|"]
    for name, m in summaries.items():
        for src, g in m.get("by_source", {}).items():
            lines.append(f"| {name} | {src} | {g['n']} | {g['n'] - g['safe_rows']} | "
                         f"{fmt(g['unsafe_recall'])} | {fmt(g['over_block_rate'])} |")
    if ROUTED_ARM in deployment_billed:
        billed = deployment_billed[ROUTED_ARM]
        lines += ["", f"Super VL deployment billed wall time for this arm: {wall[ROUTED_ARM]:.0f}s "
                  f"x ${rate_h:.2f}/h = **${billed:.3f}** "
                  f"(4x B200, per {vision_live.PRICING_URL}). "
                  "Per-verdict cost_usd sums are a higher serial upper bound; "
                  "this wall-time figure and the Fireworks meter are authoritative."]
    lines += ["", f"**Catches text-only misses:** routed arm caught {len(wins)} unsafe rows the "
              f"text path called safe/missed; text path caught {len(losses)} the routed arm missed.",
              "", "Paired comparison, vision-routed − vision-text-only (id-matched):",
              f"- unsafe recall Δ={comparison['difference']:.4f}, 95% CI "
              f"[{comparison['ci95'][0]:.4f}, {comparison['ci95'][1]:.4f}], "
              f"McNemar p={comparison['mcnemar_p']:.4g}, n={comparison['n_paired']}.",
              f"- unsafe F1 Δ={comparison_f1['difference']:.4f}, 95% CI "
              f"[{comparison_f1['ci95'][0]:.4f}, {comparison_f1['ci95'][1]:.4f}].", "",
              "Sample of unsafe rows caught only by the vision arm:"]
    for w in wins[:15]:
        lines.append(f"- `{w['id']}` ({w['source']}): {w['text']!r}")
    if not wins:
        lines.append("- (none in this run)")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "RUN_INFO.json").write_text(json.dumps({
        "banner": banner, "modes": modes, "specs": specs,
        "data": [str(p) for p in data_paths],
        "rows": len(rows), "rows_skipped_unresolved": skipped,
        "unsafe_rows": sum(r["gold_label"] == "unsafe" for r in rows),
        "safe_rows": sum(r["gold_label"] == "safe" for r in rows),
        "concurrency": {TEXT_ARM: args.text_concurrency,
                        ROUTED_ARM: args.routed_concurrency},
        "wall_seconds": wall,
        "deployment_billed_usd": deployment_billed,
        "deployment_rate_usd_per_hour": rate_h,
        "pricing_source": vision_live.PRICING_URL,
        "text_arm_spend_usd": summaries[TEXT_ARM]["cost_total_usd"],
        "text_arm_estimate_usd_preflight": text_estimate,
        "max_text_usd_cap": args.max_text_usd,
        "label_policy": ("MOCK rows are synthetic plumbing checks. LIVE rows are real "
                         "model calls. Per-verdict cost_usd on the vision arm is a "
                         "serial wall-clock upper bound; deployment_billed_usd is the "
                         "actual hot-time charge."),
    }, indent=1), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({
        "banner": banner, "modes": modes, "systems": summaries,
        "paired": {"unsafe_recall": comparison, "unsafe_f1": comparison_f1},
        "vision_only_catches": wins, "text_only_catches": losses,
        "deployment_billed_usd": deployment_billed, "wall_seconds": wall,
    }, indent=1, default=str), encoding="utf-8")

    print(banner)
    print(f"report: {run_dir / 'table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
