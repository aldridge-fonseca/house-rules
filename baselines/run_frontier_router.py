#!/usr/bin/env python3
"""Frontier-router eval driver and dry-run cost plan.

Dry-run (default): prints the exact rows per file, estimated tokens, and the
estimated USD per arm, then writes the full plan (every row id) to
``results/<run>/frontier-router.plan.json``. No network, no spend.

``--execute``: builds each arm through the registry in mode="live" (needs
``ROUTER_API_KEY``, ``ROUTER_BASE_URL`` and the arm's ``FRONTIER_MODEL*`` env)
and appends receipts to ``results/<run>/<system>.jsonl``. A later combined
``scripts/run_eval.py`` invocation reuses them by id + row hash + spec
(``safety_one.registry:<name>``), so no paid call is repeated.

Rows: ALL of ``data/frozen/disguised_v1.jsonl`` plus a seeded 300-row sample of
each public file — the identical sampler ``run_eval.py --limit-per-file`` uses
(``random.Random(f"{seed}:{path.name}").sample``), so the frontier sample is a
subset of every system's full-file rows and pairs by receipt id. Stock
``run_eval.py`` cannot express "full file + sampled files" in one invocation
(--limit-per-file applies to every --data); this driver is the workaround —
no run_eval change needed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DISGUISED = "data/frozen/disguised_v1.jsonl"
PUBLIC = ("data/public/openai_moderation.jsonl",
          "data/public/wildguard_test.jsonl",
          "data/public/xstest.jsonl")
DEFAULT_SYSTEMS = ("frontier-router-a", "frontier-router-b")
# Planning estimates only — the 800-in/30-out per-row assumption comes from
# baselines/README.md. They are never reported as measured cost.
CHARS_PER_TOKEN = 4.0
OUT_TOKENS_PER_ROW = 30
IMAGE_TOKEN_ESTIMATE = 1120      # matches RouterFrontierProvider._RESERVE_IMAGE_TOKENS


def _load_jsonl(path: Path):
    from scripts.run_eval import load_jsonl
    return load_jsonl(path)


def sampled_subset(path: Path, limit: int, seed: int):
    """Byte-for-byte the same sampler as run_eval.py --limit-per-file."""
    rows = _load_jsonl(path)
    if limit and len(rows) > limit:
        rows = random.Random(f"{seed}:{path.name}").sample(rows, limit)
    return rows


def build_plan(root: Path, seed: int, public_limit: int) -> list[dict]:
    """[{'path', 'rows', 'sampled'}] — disguised in full + seeded public slices."""
    plan = [{"path": DISGUISED, "rows": _load_jsonl(root / DISGUISED), "sampled": False}]
    for rel in PUBLIC:
        plan.append({"path": rel, "rows": sampled_subset(root / rel, public_limit, seed),
                     "sampled": bool(public_limit)})
    return plan


def estimate_tokens(rows: list[dict], prompt_chars: int) -> dict:
    in_tokens = sum(math.ceil((prompt_chars + len(r.get("text") or "")) / CHARS_PER_TOKEN)
                    + (IMAGE_TOKEN_ESTIMATE if r.get("image_path") else 0)
                    for r in rows)
    return {"rows": len(rows), "est_input_tokens": in_tokens,
            "est_output_tokens": OUT_TOKENS_PER_ROW * len(rows)}


def cost_for(est: dict, price: dict) -> float:
    return (est["est_input_tokens"] * price["input_per_million"]
            + est["est_output_tokens"] * price["output_per_million"]) / 1_000_000


def verify_subset(plan: list[dict], root: Path) -> list[str]:
    """Assert every planned row (id + input hash) exists in its full source
    file — i.e. it is a subset of what any full-file run covers."""
    from safety_one.eval.runner import _hash
    problems = []
    for item in plan:
        full = _load_jsonl(root / item["path"])
        covered = {(str(r["id"]), _hash(r)) for r in full}
        missing = [str(r["id"]) for r in item["rows"]
                   if (str(r["id"]), _hash(r)) not in covered]
        if missing:
            problems.append(f"{item['path']}: {len(missing)} planned rows absent "
                            f"from the full file (first: {missing[:3]})")
    return problems


def receipt_coverage(run_dir: Path) -> dict[str, set]:
    """id+hash coverage of existing receipts in the run dir, for the
    'subset of run frozen-v1' check once sibling systems have run."""
    coverage = {}
    if not run_dir.is_dir():
        return coverage
    for receipt in sorted(run_dir.glob("*.jsonl")):
        if receipt.name.endswith(".plan.json"):
            continue
        ids = set()
        with receipt.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ids.add((str(rec.get("id")), str(rec.get("input_hash"))))
        coverage[receipt.name] = ids
    return coverage


def _model_envs(systems: list[str]) -> dict[str, str]:
    from safety_one.registry import load_systems
    specs = (load_systems().get("systems") or {})
    return {name: (specs.get(name) or {}).get("model_env", "FRONTIER_MODEL")
            for name in systems}


def render_plan(systems: list[str], plan: list[dict], est: dict, seed: int,
                run_dir: Path, root: Path, show_ids: bool = False) -> str:
    from safety_one.budget import FRONTIER_HARD_CAP_USD
    from safety_one.providers.frontier import PRICES, ROUTER_PRICES

    envs = _model_envs(systems)
    cap = min(float(os.environ.get("SAFETY_ONE_BUDGET_FRONTIER_USD", "40")),
              FRONTIER_HARD_CAP_USD)
    lines = ["DRY-RUN PLAN — frontier router baseline (no requests sent)", "",
             f"run id: {run_dir.name}   seed: {seed}   arms: {', '.join(systems)}", "",
             "rows:"]
    for item in plan:
        tag = "all rows" if not item["sampled"] else f"seeded sample ({len(item['rows'])})"
        lines.append(f"  {item['path']:<46} {len(item['rows']):>5}  {tag}")
    total = sum(len(i["rows"]) for i in plan)
    lines += [f"  {'TOTAL':<46} {total:>5}", "",
              f"estimated tokens per arm (planning estimate, ~{CHARS_PER_TOKEN:g} chars/token "
              f"in, {OUT_TOKENS_PER_ROW} out/row):",
              f"  input  ~{est['est_input_tokens']:,}",
              f"  output ~{est['est_output_tokens']:,}", "",
              "estimated USD per arm (whole row set above):"]
    what_if = []
    for arm in systems:
        model = os.environ.get(envs.get(arm, ""), "")
        price = ROUTER_PRICES.get(model) if model else None
        if price:
            lines.append(f"  {arm:<20} model={model}  router price "
                         f"-> ~${cost_for(est, price):.2f} (source: {price.get('source') or 'prices.yaml'})")
        else:
            where = f"model={model}" if model else f"{envs.get(arm)} unset"
            lines.append(f"  {arm:<20} {where}  router price UNKNOWN — fill "
                         f"baselines/prices.yaml router.models at key time")
    for key, p in PRICES.items():
        what_if.append((key, p, cost_for(est, p)))
    lines += ["", "  what-if at already-verified DIRECT provider rates "
              "(assumption for planning only, not the router's price):"]
    for key, p, c in what_if:
        lines.append(f"    {key:<10} ${p['input_per_million']:g}/${p['output_per_million']:g} per 1M"
                     f" -> ~${c:.2f}/arm, ~${c * len(systems):.2f} for {len(systems)} arms")
    lines += ["",
              f"frontier group cap: ${cap:.2f} (SAFETY_ONE_BUDGET_FRONTIER_USD, "
              f"hard max ${FRONTIER_HARD_CAP_USD:.0f}) — paid calls reserve+settle "
              "per row in results/spend.json and stop before crossing it."]
    worst = max(c for _, _, c in what_if) * len(systems) if what_if else 0.0
    if worst > cap:
        lines.append(f"  WARNING: the most expensive what-if (${worst:.2f} for "
                     f"{len(systems)} arms) exceeds the ${cap:.2f} cap — the ledger "
                     "would halt mid-run. Raise the env cap (<= $150) or choose a "
                     "cheaper second model.")
    coverage = receipt_coverage(run_dir)
    lines.append("")
    if coverage:
        from safety_one.eval.runner import _hash
        planned = {(str(r["id"]), _hash(r)) for i in plan for r in i["rows"]}
        lines.append("pairing check vs existing receipts in the run dir:")
        for name, ids in coverage.items():
            inter = len(planned & ids)
            lines.append(f"  {name:<36} covers {inter}/{len(planned)} planned rows")
        uncovered = [n for n, ids in coverage.items() if len(planned & ids) < len(planned)]
        if uncovered:
            lines.append("  (partial coverage is expected if those systems ran "
                         "--limit-per-file on the public files only)")
    else:
        lines.append(f"pairing check: {run_dir} has no receipts yet — every planned "
                     "row is a subset of the full source files by construction "
                     "(same id + row hash), so it pairs with any full-file run "
                     "of the other systems.")
    if show_ids:
        lines.append("")
        for item in plan:
            lines.append(f"ids — {item['path']}:")
            lines += [f"  {r['id']}" for r in item["rows"]]
    return "\n".join(lines)


def write_plan_json(path: Path, systems: list[str], plan: list[dict], est: dict,
                    seed: int, public_limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"run": path.parent.name, "seed": seed, "public_limit_per_file": public_limit,
           "systems": systems, "total_rows": sum(len(i["rows"]) for i in plan),
           "estimate": est,
           "files": [{"path": i["path"], "sampled": i["sampled"], "rows": len(i["rows"]),
                      "ids": [str(r["id"]) for r in i["rows"]]} for i in plan]}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def execute(systems: list[str], rows: list[dict], run_dir: Path, concurrency: int) -> int:
    from safety_one.budget import BudgetTracker
    from safety_one.eval.runner import run_system
    from safety_one.registry import build_system

    run_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in systems:
        system = build_system(name, mode="live")   # raises without keys/models; never mock
        receipts = run_system(system, rows, run_dir / f"{name}.jsonl", system_name=name,
                              system_spec=f"safety_one.registry:{name}",
                              concurrency=concurrency)
        errors = sum(1 for r in receipts if r.get("error"))
        cost = sum(float((r.get("prediction") or {}).get("cost_usd") or 0.0) for r in receipts)
        unknown = sum(1 for r in receipts
                      if ((r.get("prediction") or {}).get("raw") or {}).get("cost_basis") == "unknown")
        note = f" ({unknown} rows cost_basis=unknown — fill prices.yaml)" if unknown else ""
        print(f"{name}: {len(receipts)} rows, {errors} errors, "
              f"measured cost ${cost:.4f}{note} -> {run_dir / (name + '.jsonl')}")
        failed = failed or bool(errors)
    totals = BudgetTracker().totals()
    frontier = sum(v for k, v in totals["providers"].items() if k.startswith("frontier")
                   or k in ("openai", "anthropic", "gemini"))
    print(f"spend ledger: frontier group ${frontier:.4f}, global ${totals['global']:.4f} "
          f"(results/spend.json)")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="frozen-v1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--public-limit", type=int, default=300,
                    help="seeded sample per public file; disguised file is always full")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--execute", action="store_true", help="paid live run; default is plan only")
    ap.add_argument("--show-ids", action="store_true", help="list every planned row id on stdout")
    args = ap.parse_args(argv)

    from safety_one.providers.frontier import moderation_prompt
    from safety_one.registry import _load_env_file
    _load_env_file()  # .env supplies ROUTER_* / FRONTIER_MODEL* if present

    run_dir = Path(args.out) / args.run
    plan = build_plan(ROOT, args.seed, args.public_limit)
    problems = verify_subset(plan, ROOT)
    if problems:
        for p in problems:
            print(f"SUBSET CHECK FAILED: {p}", file=sys.stderr)
        return 2
    rows = [r for item in plan for r in item["rows"]]
    est = estimate_tokens(rows, len(moderation_prompt()))
    write_plan_json(run_dir / "frontier-router.plan.json", args.systems, plan, est,
                    args.seed, args.public_limit)
    print(render_plan(args.systems, plan, est, args.seed, run_dir, ROOT, args.show_ids))
    print(f"\nplan file: {run_dir / 'frontier-router.plan.json'}")
    if not args.execute:
        print("\nno requests sent. Live run once ROUTER_BASE_URL, ROUTER_API_KEY and "
              "the arm FRONTIER_MODEL* vars are set:")
        print(f"  .venv/bin/python baselines/run_frontier_router.py --run {args.run} "
              f"--systems {' '.join(args.systems)} --execute")
        return 0
    return execute(args.systems, rows, run_dir, args.concurrency)


if __name__ == "__main__":
    raise SystemExit(main())
