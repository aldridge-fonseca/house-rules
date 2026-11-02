#!/usr/bin/env python3
"""Run resumable safety evaluation. From safety-one/: python scripts/run_eval.py"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.eval.report import build_report
from safety_one.eval.runner import run_system


ABLATIONS = ("specialist-only", "+rules", "+router", "+escalation", "full")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            required = ("id", "text", "image_path", "lang", "gold_label", "gold_categories", "source", "split")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing fields {', '.join(missing)}")
            if row["gold_label"] not in ("safe", "unsafe"):
                raise ValueError(f"{path}:{lineno}: gold_label must be safe or unsafe")
            rows.append(row)
    return rows


def _mock_factory(kind: str):
    from safety_one.pipeline import SafetyPipeline, load_config
    from safety_one.providers.mock import MockProvider, default_mock_providers

    cfg = load_config()
    providers = default_mock_providers(cfg)
    if kind in ABLATIONS:
        pcfg = json.loads(json.dumps(cfg))
        pcfg.setdefault("router", {})
        pcfg.setdefault("confidence_gate", {})
        if kind == "specialist-only":
            pcfg["router"].update(route_order=[], default_route="specialist")
            pcfg["confidence_gate"]["enabled"] = False
        elif kind == "+rules":
            pcfg["router"].update(route_order=["rule"], default_route="specialist")
            pcfg["confidence_gate"]["enabled"] = False
        elif kind == "+router":
            pcfg["confidence_gate"]["enabled"] = False
        # +escalation and full run the complete config (router + gate).
        return SafetyPipeline(providers, pcfg)
    if kind == "frontier":
        return MockProvider(name="frontier", model="mock-frontier", seed=20261110,
                            base_latency_ms=250, cost_per_call_usd=0.002, sharpness=1.05)
    if kind in ("open-guard", "open_guard"):
        return MockProvider(name="open-guard", model="mock-open-guard", seed=20261111,
                            base_latency_ms=120, cost_per_call_usd=0.0002, sharpness=1.2)
    if kind in providers:
        return providers[kind]
    raise ValueError(f"unknown built-in mock system {kind!r}")


def _load_system(spec: str):
    """mock:<kind> built-ins; otherwise module:factory, imported only on request."""
    if spec.startswith("mock:"):
        return _mock_factory(spec.partition(":")[2])
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"system spec {spec!r} must be mock:<kind> or module:factory")
    from importlib import import_module
    obj = getattr(import_module(module_name), attr)
    return obj() if isinstance(obj, type) or callable(obj) else obj


def _system_mode(spec: str, system) -> str:
    """'mock' | 'live' | 'mixed' | 'unknown' — registry systems carry .mode."""
    if spec.startswith("mock:"):
        return "mock"
    mode = getattr(system, "mode", None)
    return str(mode) if mode else "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", action="append", help="JSONL eval input; repeat for datasets")
    ap.add_argument("--system", action="append", metavar="NAME=SPEC",
                    help="repeatable; built-ins: mock:specialist-only, mock:+rules, mock:+router, mock:+escalation, mock:full, mock:frontier, mock:open-guard")
    ap.add_argument("--run", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--concurrency", "--workers", dest="concurrency", type=int, default=4,
                    help="max in-flight classify calls per system; receipts resume by id+row hash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-per-file", type=int, default=0,
                    help="deterministic seeded sample of N rows per --data file (0 = all rows)")
    ap.add_argument("--label", default=None,
                    help="banner for table.md/summary.json; default derives MOCK/LIVE from system modes")
    args = ap.parse_args(argv)
    datasets = [Path(p) for p in (args.data or ["tests/fixtures/eval.jsonl"])]
    rows = []
    for path in datasets:
        file_rows = load_jsonl(path)
        if args.limit_per_file and len(file_rows) > args.limit_per_file:
            import random
            file_rows = random.Random(f"{args.seed}:{path.name}").sample(file_rows, args.limit_per_file)
        rows += file_rows
    specs = args.system or [f"{name}=mock:{name}" for name in (*ABLATIONS, "frontier", "open-guard")]
    systems = []
    for item in specs:
        name, sep, spec = item.partition("=")
        if not sep or not name or not spec:
            ap.error(f"invalid --system {item!r}; use NAME=SPEC")
        systems.append((name, spec, _load_system(spec)))
    run_dir = Path(args.out) / args.run
    modes = {name: _system_mode(spec, system) for name, spec, system in systems}
    non_live = [(name, spec, modes[name]) for name, spec, _ in systems if modes[name] != "live"]
    if non_live:
        warn = ["!" * 78,
                "!! MOCK RESULTS — synthetic scores/latency/cost, NOT model measurements",
                "!! These systems resolved to non-live providers (missing API keys in .env?):"]
        for name, spec, mode in non_live:
            hint = "explicit mock spec" if spec.startswith("mock:") \
                else "live-capable spec fell back to mock"
            warn.append(f"!!   {name} = {spec} -> mode={mode} ({hint})")
        warn += ["!! Set provider keys in .env (see .env.example) for a LIVE run.", "!" * 78]
        print("\n".join(warn), file=sys.stderr)
    receipts = {}
    for name, spec, system in systems:
        receipts[name] = run_system(system, rows, run_dir / f"{name}.jsonl", system_name=name,
                                    system_spec=spec, concurrency=args.concurrency)
        print(f"{name}: {len(receipts[name])} rows -> {run_dir / (name + '.jsonl')}")
    banner = args.label
    if banner is None:
        tag = "MOCK" if all(m == "mock" for m in modes.values()) else "LIVE" if all(m == "live" for m in modes.values()) else "MIXED"
        banner = (f"{tag} run — system modes: " + ", ".join(f"{k}={v}" for k, v in modes.items())
                  + (". Mock scores/latency/cost are synthetic plumbing checks, not model measurements." if tag != "LIVE" else ""))
    (run_dir / "RUN_INFO.json").write_text(json.dumps(
        {"banner": banner, "modes": modes, "data": [str(p) for p in datasets], "rows": len(rows),
         "limit_per_file": args.limit_per_file, "seed": args.seed,
         "specs": {name: spec for name, spec, _ in systems}}, indent=1), encoding="utf-8")
    baseline = next((n for n in receipts if n == "frontier" or n.startswith("frontier")), next(iter(receipts)))
    build_report(receipts, run_dir, baseline=baseline, seed=args.seed, banner=banner)
    print(banner)
    print(f"report: {run_dir / 'table.md'} (charts are SVG; receipts resume by id and row hash)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
