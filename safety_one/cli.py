"""Command line for the named-system registry.

From repo root:

    python -m safety_one.cli list-systems
    python -m safety_one.cli list-systems --json
    python -m safety_one.cli classify "some text" --system full
    python -m safety_one.cli classify "..." --system frontier-gpt --mode mock --json

Mode "auto" (default) uses the real provider only when its API key is in the
environment (or `.env`); otherwise a deterministic mock answers, so
everything here works offline.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .registry import MODES, build_system, list_systems
from .types import Message, Verdict


def _verdict_dict(v: Verdict) -> dict:
    d = dataclasses.asdict(v) if dataclasses.is_dataclass(v) else dict(v)
    return d


def _print_verdict(v: Verdict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_verdict_dict(v), indent=2, sort_keys=True))
        return
    rows = [
        ("label", v.label),
        ("route", v.route),
        ("model", v.model),
        ("categories", ", ".join(v.categories) or "-"),
        ("probs", "  ".join(f"{k}={p}" for k, p in sorted(v.probs.items())) or "-"),
        ("rule_hits", ", ".join(v.rule_hits) or "-"),
        ("latency_ms", f"{v.latency_ms:.1f}"),
        ("cost_usd", f"{v.cost_usd:.6f}"),
    ]
    if v.raw.get("escalated"):
        rows.append(("escalated", "true"))
    width = max(len(k) for k, _ in rows)
    for k, val in rows:
        print(f"{k:<{width}}  {val}")


def _cmd_list(args) -> int:
    systems = list_systems()
    if args.json:
        print(json.dumps(systems, indent=2))
        return 0
    width = max(len(s["name"]) for s in systems)
    for s in systems:
        if not s["live_supported"]:
            avail = "mock-only"
        elif s["env_key"]:
            avail = f"live:{s['env_key']}{' (set)' if s['key_present'] else ''}"
        else:
            avail = "live:no-key-needed"
        print(f"{s['name']:<{width}}  {s['kind']:<8}  {avail:<32}  {s['description']}")
    return 0


def _cmd_classify(args) -> int:
    try:
        system = build_system(args.system, mode=args.mode, seed=args.seed)
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    msg = Message(id=args.id, text=args.text,
                  image_path=args.image, lang=args.lang)
    try:
        verdict = system.classify(msg)
    except Exception as exc:
        print(f"error: classify failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    if not args.json:
        mode = getattr(system, "mode", args.mode)
        print(f"system: {args.system} ({mode})")
    _print_verdict(verdict, args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="safety_one.cli", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list-systems", help="list named systems from config/systems.yaml")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=_cmd_list)

    cl = sub.add_parser("classify", help="classify one message")
    cl.add_argument("text")
    cl.add_argument("--system", default="full",
                    help="named system from config/systems.yaml (default: full)")
    cl.add_argument("--mode", choices=MODES, default="auto")
    cl.add_argument("--image", dest="image", default=None,
                    help="optional image path (routes to the vision lane)")
    cl.add_argument("--lang", default=None, help="optional language tag")
    cl.add_argument("--id", default="cli-1", help="message id (seeds the mock)")
    cl.add_argument("--seed", type=int, default=None, help="mock seed override")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=_cmd_classify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
