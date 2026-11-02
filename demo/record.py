"""Record cached race results for replay mode.

  python3 demo/record.py                          # classify now -> cached/results.jsonl
  python3 demo/record.py --cases X --out Y        # custom paths
  python3 demo/record.py --run results/<run>      # from eval-run receipts (offline)
  python3 demo/record.py --run results/nvidia-v1
  # (scripts that still mention results/frozen-v1 regenerate an earlier
  #  internal run whose receipts are not published in this repo)

--run reads receipts written by scripts/run_eval.py: results/<run>/<system>.jsonl
rows carry {"id", "text", "prediction": {...verdict...}, "error"}. The demo's
two lanes are mapped onto eval systems (override with --ours-system /
--frontier-system):

  ours     <- full              (fallback: full-cal, +escalation, specialist-only)
  frontier <- frontier-router-a (fallback: frontier-router, frontier-gpt/-claude/-gemini, frontier)

Other frontier arms (e.g. frontier-router-b = Claude Opus) and the untuned
NVIDIA baseline are attached per case under "others" (label / refused) so the
UI can say "Opus refused this one". Other receipt files are kept under "extras".

Case selection (--select):
  cases  join demo/cases.jsonl ids to the receipts (the eval must have run
         over that file).
  auto   (default) use demo/cases.jsonl when most of its ids have both lanes,
         otherwise pick cases from the receipts themselves: rows both lanes
         answered — curated highlights (disguised attacks the second frontier
         arm refused and ours caught, attacks the frontier lane missed,
         gray-zone escalations, benign look-alikes, and honest misses where
         ours is wrong) plus a seeded random sample, and 2-3 image cases from
         --vision receipts when present (ours may show a vision route only if
         the receipt did; the frontier lane did not run on images and is shown
         as "not run"). The LIVE-RECORDED default cache has no vision-lane
         verdict — the meme SVG was folded into text. Case rows are stored in
         each cache record ("case"), which the server replays.

MOCK/LIVE labels per lane come from RUN_INFO.json modes, then
frontier_summary.json's label, then receipt evidence (provider response ids).
Ours-lane cost adds the allocated dedicated-GPU cost from nano_summary.json
(receipts store $0 for calls to a dedicated deployment); image-case cost is the
batch's billed GPU time per image from the vision run's summary.json.

Default mode classifies each case through the lane backends right now
(real SafetyPipeline when importable, deterministic mocks otherwise).
Each output line: {id, recorded_at, mode, origin, systems?, case?, tag?,
others?, ours, frontier, extras?} — verdicts per the Verdict schema plus optional
"normalized"/"transforms" keys used by the UI's raw-vs-cleaned display.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import backends

DEMO = Path(__file__).resolve().parent
REPO_ROOT = DEMO.parent
IMAGE_ROOT = REPO_ROOT / "data" / "vision" / "images"

# `full` first: headline system in ours_summary (full-cal scored lower).
OURS_CANDIDATES = ["full", "full-cal", "+escalation", "specialist-only"]
FRONTIER_CANDIDATES = ["frontier-router-a", "frontier-router", "frontier-gpt",
                       "frontier-claude", "frontier-gemini", "frontier",
                       "openai-moderation"]
MODEL_NAMES = {"openai/gpt-6.1-sol": "GPT-6.1 Sol",
               "anthropic/claude-opus-5.5": "Claude Opus 5.5",
               "accounts/fireworks/models/nemotron-3-ultra-nvfp4":
                   "untuned Nemotron 3 Ultra"}
REFUSAL_MARK = "ContentFilter"
SEED = 20261102


def load_cases(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------- run

def _resolve_run(arg):
    p = Path(arg)
    for cand in (p, REPO_ROOT / arg, REPO_ROOT / "results" / arg,
                 Path.cwd() / arg):
        if cand.is_dir():
            return cand.resolve()
    sys.exit(f"run dir not found: {arg} (tried {p}, results/{arg})")


def _rows(path):
    """{case_id: full receipt row}; the last record per id wins (resumed runs)."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "id" in r:
            out[str(r["id"])] = r
    return out


def _receipts(path):
    """{case_id: prediction-verdict} from a <system>.jsonl eval receipt file."""
    return {cid: r["prediction"] for cid, r in _rows(path).items()
            if not r.get("error") and isinstance(r.get("prediction"), dict)}


def _looks_like_receipt(path):
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                return False
            return isinstance(r, dict) and "prediction" in r and "id" in r
    return False


def _pick_system(run_dir, explicit, candidates):
    if explicit:
        if not (run_dir / f"{explicit}.jsonl").exists():
            sys.exit(f"system {explicit!r} not found: {run_dir}/{explicit}.jsonl")
        return explicit
    for c in candidates:
        if (run_dir / f"{c}.jsonl").exists():
            return c
    return None


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _is_refusal(row):
    return REFUSAL_MARK in str(row.get("error") or "")


def _model_of(rows):
    c = Counter(str((r.get("prediction") or {}).get("model") or "")
                for r in rows.values() if not r.get("error"))
    c.pop("", None)
    return c.most_common(1)[0][0] if c else ""


def _live_evidence(row):
    """A provider response id, or real token usage from a Fireworks-hosted model."""
    pred = row.get("prediction") or {}
    raw = pred.get("raw") if isinstance(pred.get("raw"), dict) else {}
    if raw.get("mock"):
        return False
    if (raw.get("response") or {}).get("id"):
        return True
    usage = raw.get("usage") or {}
    return ((str(pred.get("model") or "").startswith("accounts/") and int(usage.get("prompt_tokens") or 0) > 0)
            or str(pred.get("route")) == "rule")


def run_modes(run_dir, systems):
    """{system: mode} from RUN_INFO.json, frontier_summary.json, then receipts."""
    modes = {k: str(v).lower() for k, v in
             (_json(run_dir / "RUN_INFO.json").get("modes") or {}).items()}
    fs = _json(run_dir / "frontier_summary.json")
    for arm in fs.get("arms") or {}:
        if fs.get("label"):
            modes.setdefault(arm, str(fs["label"]).lower())
    for name, rows in systems.items():
        if name in modes or not rows:
            continue
        ok = [r for r in rows.values() if not r.get("error")]
        raws = [((r.get("prediction") or {}).get("raw") or {}) for r in ok]
        if any(isinstance(x, dict) and x.get("mock") for x in raws):
            modes[name] = "mock"
        elif ok and sum(_live_evidence(r) for r in ok) >= 0.9 * len(ok):
            modes[name] = "live"
        else:
            modes[name] = "unknown"
    return modes


def slim(pred, *, extra_cost=0.0, cost_basis=None):
    """Verdict for the UI: drop the bulky provider response, flag usable P(unsafe)."""
    v = dict(pred)
    raw = dict(v.get("raw") or {})
    raw.pop("response", None)
    v["raw"] = raw
    p = (v.get("probs") or {}).get("unsafe")
    v["probs_reliable"] = p is not None and (float(p) >= 0.5) == (v.get("label") == "unsafe")
    if extra_cost:
        v["cost_usd"] = float(v.get("cost_usd") or 0.0) + extra_cost
    if cost_basis:
        v["cost_basis"] = cost_basis
    return v


def _case(row, **extra):
    c = {k: row.get(k) for k in ("id", "text", "image_path", "lang", "gold_label",
                                  "gold_categories", "source", "perturbation")}
    c["gold_categories"] = c.get("gold_categories") or []
    c.update({k: v for k, v in extra.items() if v is not None})
    return c


def _other_note(name, row, models):
    label = MODEL_NAMES.get(models.get(name, ""), name)
    if row is None:
        return None
    if _is_refusal(row):
        return {"system": name, "name": label, "refused": True, "label": None}
    if row.get("error"):
        return None
    return {"system": name, "name": label, "refused": False,
            "label": (row.get("prediction") or {}).get("label")}


def _lab(row):
    return None if row is None or row.get("error") else (row.get("prediction") or {}).get("label")


def select_cases(ours, front, second, n_text=40, seed=SEED, front_label="frontier lane",
                 second_label="second frontier arm"):
    """Deterministic demo pick from rows both lanes answered.

    Returns [(case_id, tag)], highlights interleaved first, then a seeded
    random sample so the tally is not only cherry-picked rows.
    """
    pool = sorted(cid for cid, r in ours.items()
                  if not r.get("error") and cid in front and not front[cid].get("error"))
    info = {}
    for cid in pool:
        o, f = ours[cid], front[cid]
        s = second.get(cid) if second else None
        info[cid] = {
            "gold": o.get("gold_label"), "pert": str(o.get("perturbation") or "none"),
            "src": o.get("source"), "ours": _lab(o), "front": _lab(f),
            "second": "refused" if s is not None and _is_refusal(s) else _lab(s),
            "route": str((o.get("prediction") or {}).get("route") or ""),
            "pre": ((o.get("prediction") or {}).get("raw") or {}).get("pre_escalation") or {},
        }
    chosen, picks = set(), {k: [] for k in "ADBECR"}

    def take(bucket, cands, limit, tag, one_per_pert=False):
        seen = set()
        for cid in cands:
            if len(picks[bucket]) >= limit:
                break
            if cid in chosen or (one_per_pert and info[cid]["pert"] in seen):
                continue
            seen.add(info[cid]["pert"])
            chosen.add(cid)
            picks[bucket].append((cid, tag(cid)))

    attack = [c for c in pool if info[c]["pert"] != "none" and info[c]["gold"] == "unsafe"
              and info[c]["ours"] == "unsafe"]
    a = sorted((c for c in attack if info[c]["second"] == "refused"),
               key=lambda c: (info[c]["front"] == "unsafe", c))
    take("A", a, 5, lambda c: f"disguised attack ({info[c]['pert']}) — ours caught it; "
         f"{second_label} refused" + (f"; {front_label} missed it" if info[c]["front"] == "safe" else ""),
         one_per_pert=True)
    b = [c for c in attack if info[c]["front"] == "safe" and info[c]["second"] != "refused"]
    take("B", b, 3, lambda c: f"disguised attack ({info[c]['pert']}) — {front_label} missed, ours caught",
         one_per_pert=True)
    e = sorted((c for c in pool if "escalation" in info[c]["route"] and info[c]["ours"] == info[c]["gold"]),
               key=lambda c: (info[c]["pre"].get("label") == info[c]["ours"], c))
    take("E", e, 2, lambda c: "gray zone → escalated to Nemotron 3 Ultra"
         + (" and the escalation fixed the call" if info[c]["pre"].get("label") not in (None, info[c]["ours"]) else ""))
    d = sorted((c for c in pool if info[c]["src"] == "xstest" and info[c]["gold"] == "safe"
                and info[c]["ours"] == "safe" and info[c]["front"] == "safe"),
               key=lambda c: (info[c]["pert"] != "none", info[c]["second"] != "refused", c))
    take("D", d, 2, lambda c: "benign look-alike (XSTest" + ("" if info[c]["pert"] == "none" else f", {info[c]['pert']}")
         + ") — both lanes pass it" + (f"; {second_label} refused a safe message" if info[c]["second"] == "refused" else ""))
    over = [c for c in pool if info[c]["gold"] == "safe" and info[c]["ours"] == "unsafe" and info[c]["front"] == "safe"]
    miss = [c for c in pool if info[c]["gold"] == "unsafe" and info[c]["ours"] == "safe" and info[c]["front"] == "unsafe"]
    take("C", over, 1, lambda c: f"honest miss — ours over-blocks a safe message, {front_label} is right")
    take("C", miss, 2, lambda c: f"honest miss — ours lets an unsafe message through, {front_label} is right")
    rest = [c for c in pool if c not in chosen]
    n_rand = max(0, n_text - len(chosen))
    for cid in sorted(random.Random(seed).sample(rest, min(n_rand, len(rest)))):
        picks["R"].append((cid, "random sample (seeded)"))
    return picks


_BANNED_SRC = re.compile(r"(wildguard|vlguard|siuo)", re.I)


def _banned_row(row) -> bool:
    """True if a receipt/case touches WildGuard / VLGuard / SIUO (not shippable)."""
    if not isinstance(row, dict):
        return True
    blob = " ".join(str(row.get(k) or "") for k in
                    ("id", "source", "text", "image_path", "perturbation"))
    if _BANNED_SRC.search(blob):
        return True
    meta = row.get("meta") or {}
    if isinstance(meta, dict) and _BANNED_SRC.search(json.dumps(meta, ensure_ascii=False)):
        return True
    return False


def vision_cases(vision_dir, limit=3):
    """Permissively-licensed image catches from vision receipts (HatemojiCheck only).

    SIUO / VLGuard / WildGuard rows are skipped — their text/images are not
    shippable in the public demo cache.
    """
    routed_p, text_p = vision_dir / "vision-routed.jsonl", vision_dir / "vision-text-only.jsonl"
    if not routed_p.exists():
        return [], {}
    routed, text = _rows(routed_p), (_rows(text_p) if text_p.exists() else {})
    summ = _json(vision_dir / "summary.json")
    billed = float((summ.get("deployment_billed_usd") or {}).get("vision-routed") or 0.0)
    per_image = billed / len(routed) if billed and routed else 0.0

    def usable(cid):
        r = routed.get(cid)
        if not (r and not r.get("error") and r.get("image_path")
                and Path(r["image_path"]).is_file() and _in_root(Path(r["image_path"]))):
            return False
        return not _banned_row(r)

    out = []
    # Prefer hatemoji image-only catches (CC-BY-4.0); never SIUO/VLGuard.
    for item in summ.get("vision_only_catches") or []:
        cid = item.get("id")
        src = str(item.get("source") or "")
        if _BANNED_SRC.search(src) or _BANNED_SRC.search(str(cid or "")):
            continue
        r = routed.get(cid)
        if usable(cid) and slim(r["prediction"])["probs_reliable"]:
            out.append((cid, "image case — harm only visible with the image; text-only said safe"))
        if len(out) >= limit - 1:
            break
    for cid in sorted(routed):
        r = routed[cid]
        if (usable(cid) and r.get("gold_label") == "safe" and _lab(r) == "safe"
                and _lab(text.get(cid)) == "safe" and str(cid).startswith("hatemoji")):
            out.append((cid, "image case — vision receipt (lane only if routed)"))
            break
    return out[:limit], {"routed": routed, "text": text, "per_image": per_image,
                         "modes": {k: str(v).lower() for k, v in
                                   (_json(vision_dir / "RUN_INFO.json").get("modes") or {}).items()},
                         "dir": vision_dir}


def _in_root(path):
    try:
        path.resolve().relative_to(IMAGE_ROOT.resolve())
        return True
    except ValueError:
        return False


def _web_image(path):
    return "/case-img/" + str(Path(path).resolve().relative_to(IMAGE_ROOT.resolve()))


def _interleave(picks, images):
    queues = [list(picks["A"]), list(picks["D"]), list(picks["B"]), list(images),
              list(picks["E"]), list(picks["C"])]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out + list(picks["R"])


def record_run(args):
    run_dir = _resolve_run(args.run)
    ours_name = _pick_system(run_dir, args.ours_system, OURS_CANDIDATES)
    if ours_name is None:
        sys.exit(f"no ours-lane system in {run_dir} "
                 f"(looked for {', '.join(OURS_CANDIDATES)}; use --ours-system)")
    front_name = _pick_system(run_dir, args.frontier_system, FRONTIER_CANDIDATES)
    if front_name is None:
        print(f"warning: no frontier-lane system in {run_dir} "
              f"(looked for {', '.join(FRONTIER_CANDIDATES)})", file=sys.stderr)

    systems = {f.stem: _rows(f) for f in sorted(run_dir.glob("*.jsonl")) if _looks_like_receipt(f)}
    modes = run_modes(run_dir, systems)
    models = {name: _model_of(rows) for name, rows in systems.items()}
    ours_rows = systems.get(ours_name, {})
    front_rows = systems.get(front_name, {}) if front_name else {}
    other_names = [n for n in systems if n != front_name and
                   (n.startswith("frontier") or n.startswith("nvidia-"))]
    second = next((systems[n] for n in other_names if n.startswith("frontier")), {})
    extras = {n: rows for n, rows in systems.items()
              if n not in (ours_name, front_name) and n not in other_names}
    gpu = float(_json(run_dir / "nano_summary.json").get("gpu_per_call_usd") or 0.0)

    def ours_verdict(row, text):
        pred = row["prediction"]
        dep = "/deployments/" in str(pred.get("model") or "")
        v = slim(pred, extra_cost=gpu if dep else 0.0,
                 cost_basis=(f"receipt + allocated dedicated GPU ${gpu * 1e6:,.0f}/1M calls "
                             "(nano_summary.json)") if dep and gpu else "receipt")
        return backends.attach_norm(v, text)

    cases = [c for c in (load_cases(args.cases) if args.cases.exists() else []) if not _banned_row(c)]
    # Drop banned sources from receipt pools before selection/join.
    ours_rows = {k: v for k, v in ours_rows.items() if not _banned_row(v)}
    front_rows = {k: v for k, v in front_rows.items() if not _banned_row(v)}
    second = {k: v for k, v in second.items() if not _banned_row(v)} if second else {}
    both = sum(1 for c in cases if c["id"] in ours_rows and c["id"] in front_rows)
    use_cases = args.select == "cases" or (args.select == "auto" and cases and both >= len(cases) / 2)

    plan = []  # (case_row, tag, ours_verdict, frontier_verdict, others, mode)
    if use_cases:
        for row in cases:
            o, f = ours_rows.get(row["id"]), front_rows.get(row["id"])
            ov = ours_verdict(o, row["text"]) if o and not o.get("error") else None
            fv = slim(f["prediction"]) if f and not f.get("error") else None
            plan.append((row, None, ov, fv, [], None))
    else:
        second_name = next((n for n in other_names if n.startswith("frontier")), None)
        picks = select_cases(ours_rows, front_rows, second, n_text=args.n_text,
                             front_label=MODEL_NAMES.get(models.get(front_name, ""), front_name or "frontier"),
                             second_label=MODEL_NAMES.get(models.get(second_name, ""), second_name or ""))
        images, vis = vision_cases(Path(_resolve_run(args.vision)), args.n_images) if args.vision else ([], {})
        for cid, tag in _interleave(picks, images):
            if cid in ours_rows:
                o = ours_rows[cid]
                if _banned_row(o):
                    continue
                notes = [x for x in (_other_note(n, systems[n].get(cid), models) for n in other_names) if x]
                plan.append((_case(o, tag=tag), tag, ours_verdict(o, o["text"]),
                             slim(front_rows[cid]["prediction"]), notes, None))
            else:
                r = vis["routed"][cid]
                if _banned_row(r):
                    continue
                t = vis["text"].get(cid)
                notes = []
                if t is not None and not t.get("error"):
                    notes.append({"system": "vision-text-only", "name": "untuned Nemotron 3 Ultra, text only (no image)",
                                  "refused": False, "label": _lab(t)})
                v = slim(r["prediction"], cost_basis="batch GPU time / image (vision summary.json billed)")
                if vis["per_image"]:
                    v["cost_usd"] = vis["per_image"]
                v["raw"].setdefault("route_reasons", ["image present"])
                plan.append((_case(r, tag=tag, image_path=_web_image(r["image_path"])), tag, v, None, notes,
                             {"ours": vis["modes"].get("vision-routed", "unknown"), "frontier": "none"}))

    matched_ours = matched_front = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    origin = f"run:{run_dir.name}" + (f"+{Path(args.vision).name}" if args.vision and not use_cases else "")
    with args.out.open("w") as fh:
        for row, tag, ov, fv, notes, mode in plan:
            matched_ours += ov is not None
            matched_front += fv is not None
            cid = row["id"]
            rec = {
                "id": cid,
                "recorded_at": round(time.time(), 3),
                "mode": mode or {"ours": modes.get(ours_name, "recorded"),
                                 "frontier": (modes.get(front_name, "recorded") if front_name else "none")},
                "origin": origin,
                "systems": {"ours": ours_name if mode is None else "vision-routed",
                            "frontier": front_name if mode is None else None},
                "ours": ov,
                "frontier": fv,
            }
            if not use_cases:
                rec["case"] = row
                rec["tag"] = tag
                rec["others"] = notes
                if fv is None:
                    rec["frontier_note"] = "not run — the frontier baseline was text-only (no image input)"
            ex = {k: v[cid]["prediction"] for k, v in extras.items()
                  if cid in v and not v[cid].get("error")}
            if ex and use_cases:
                rec["extras"] = {k: slim(v) for k, v in ex.items()}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(plan)
    print(f"recorded {n} cases -> {args.out}")
    print(f"  run={run_dir.name}  select={'cases file' if use_cases else 'auto from receipts'}  "
          f"ours={ours_name} ({matched_ours}/{n})  frontier={front_name} ({matched_front}/{n})  "
          f"others={other_names}")
    print(f"  modes: ours={modes.get(ours_name)} frontier={modes.get(front_name)} "
          f"others={ {k: modes.get(k) for k in other_names} }")
    if not use_cases:
        tags = Counter((t or "").split(" — ")[0].split(" (")[0] for _r, t, *_ in plan)
        print(f"  picks: {dict(tags)}")
    if matched_ours == 0:
        sys.exit("error: 0 case ids matched the ours receipts — cache left written but empty. "
                 "Re-run eval with --data demo/cases.jsonl first, or use --select auto.")
    if matched_ours < n:
        print(f"  warning: {n - matched_ours} case ids have no ours receipt; "
              f"replay will live-classify those", file=sys.stderr)


# ---------------------------------------------------------------- live/mock

GPU_USD_PER_HOUR = 13.0  # 1× B200 dedicated deployment (README / nano_summary)
FALLBACK_PATH = DEMO / "cached" / "fallback_recorded.jsonl"
FALLBACK_META = DEMO / "cached" / "fallback_meta.json"


def _is_mockish(verdict):
    if not isinstance(verdict, dict):
        return True
    raw = verdict.get("raw") or {}
    if raw.get("mock") or raw.get("simulated"):
        return True
    model = str(verdict.get("model") or "").lower()
    return model.startswith("mock/") or model.endswith("-sim")


def _shippable_case(row):
    """Copy a case row, keeping only image assets that ship under demo/assets/.

    Drops /case-img/... references (SIUO/VLGuard/vision eval trees are not in
    the public checkout) so smoke + offline replay never 404.
    """
    c = dict(row)
    ip = c.get("image_path")
    if not ip:
        return c
    name = Path(str(ip)).name
    asset = DEMO / "assets" / name
    if asset.is_file():
        c["image_path"] = f"/assets/{name}"
        return c
    # Not shipped — fold to text-only so the case still races.
    c["image_path"] = None
    c.setdefault("meta", {})
    if isinstance(c["meta"], dict):
        c["meta"] = dict(c["meta"])
        c["meta"]["dropped_image"] = str(ip)
    return c


def _accuracy(cases, verdicts_by_id, lane):
    ok = tot = 0
    for row in cases:
        v = (verdicts_by_id.get(row["id"]) or {}).get(lane)
        if not v or v.get("skipped"):
            continue
        tot += 1
        if v.get("label") == row.get("gold_label"):
            ok += 1
    return ok, tot, (ok / tot if tot else 0.0)


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


def record_classify(args):
    """Classify every case through the live (or mock) lane backends.

    Parallel requests (--workers). Stops and reports if the running spend
    (frontier token cost + allocated Fireworks GPU wall-clock at $13/h +
    any serverless token cost already on the verdict) would exceed --budget.
    On a fully LIVE run, also freezes a copy to fallback_recorded.jsonl for
    `python3 demo/server.py --replay-fallback`.
    """
    import concurrent.futures
    import os
    import threading

    # Ensure live flags before (re)loading backends — record.py may be imported
    # by the server before env is set.
    if args.live:
        os.environ.setdefault("DEMO_LIVE_FRONTIER", "1")
    # Use a demo-local spend ledger so prior eval spend in results/spend.json
    # does not block the GPT lane. record.py still enforces --budget itself
    # (GPU wall-clock + token costs).
    demo_spend = args.out.parent / "spend.json"
    os.environ["SAFETY_ONE_SPEND_FILE"] = str(demo_spend)
    # Caps must clear the --budget headroom; the hard frontier max is $150.
    os.environ["SAFETY_ONE_BUDGET_GLOBAL_USD"] = str(max(float(args.budget) + 1.0, 20.0))
    os.environ["SAFETY_ONE_BUDGET_FRONTIER_USD"] = str(max(float(args.budget), 12.0))
    os.environ["SAFETY_ONE_BUDGET_FIREWORKS_USD"] = str(max(float(args.budget), 12.0))
    backends.reload_lanes()
    cases = [_shippable_case(c) for c in load_cases(args.cases) if not _banned_row(c)]
    modes = backends.lane_modes()
    if args.require_live and (modes.get("ours") != "live" or modes.get("frontier") != "live"):
        sys.exit(f"error: require-live but lanes={modes}. Set FIREWORKS_MODEL + "
                 "DEMO_LIVE_FRONTIER=1 + ROUTER_API_KEY first.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    recorded_at = round(time.time(), 3)
    origin = "classify-live" if modes.get("ours") == "live" and modes.get("frontier") == "live" \
        else "classify"
    systems = {"ours": "full", "frontier": "frontier-router-a"}
    budget = float(args.budget)
    workers = max(1, int(args.workers))
    stop = threading.Event()
    lock = threading.Lock()
    state = {"token_usd": 0.0, "done": 0, "errors": 0}
    t0 = time.perf_counter()
    results = {}

    def one(row):
        if stop.is_set():
            return row["id"], None, "stopped"
        try:
            res = backends.classify_both(row)
        except Exception as e:
            with lock:
                state["errors"] += 1
            return row["id"], None, f"{type(e).__name__}: {e}"
        cost = float((res.get("ours") or {}).get("cost_usd") or 0.0) + \
            float((res.get("frontier") or {}).get("cost_usd") or 0.0)
        with lock:
            state["token_usd"] += cost
            state["done"] += 1
            elapsed_h = (time.perf_counter() - t0) / 3600.0
            # GPU bills while the dedicated deployment is up — allocate the
            # wall-clock so far across completed cases for the running total.
            gpu_so_far = elapsed_h * GPU_USD_PER_HOUR if modes.get("ours") == "live" else 0.0
            spent = state["token_usd"] + gpu_so_far
            if spent > budget:
                stop.set()
                return row["id"], res, f"budget-exceeded:${spent:.4f}>{budget:.2f}"
        return row["id"], res, None

    print(f"recording {len(cases)} cases  workers={workers}  budget=${budget:.2f}  "
          f"lanes={modes}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, row) for row in cases]
        for fut in concurrent.futures.as_completed(futs):
            cid, res, err = fut.result()
            if res is not None:
                results[cid] = res
            if err:
                print(f"  ! {cid}: {err}", file=sys.stderr, flush=True)
            if stop.is_set():
                break
        if stop.is_set():
            for fut in futs:
                fut.cancel()

    wall_s = time.perf_counter() - t0
    gpu_usd = (wall_s / 3600.0) * GPU_USD_PER_HOUR if modes.get("ours") == "live" else 0.0
    n = len(cases)
    n_ok = len(results)
    gpu_per = (gpu_usd / n_ok) if n_ok else 0.0

    # Write in case-file order so replay order matches the locked list.
    written = []
    with args.out.open("w", encoding="utf-8") as f:
        for row in cases:
            res = results.get(row["id"])
            if res is None:
                continue
            ours = dict(res["ours"] or {})
            front = dict(res["frontier"] or {})
            # Allocate dedicated-GPU cost onto the ours lane (receipts store $0
            # for /deployments/ calls; same convention as nano_summary).
            if gpu_per and "/deployments/" in str(ours.get("model") or ""):
                ours["cost_usd"] = float(ours.get("cost_usd") or 0.0) + gpu_per
                ours["cost_basis"] = (
                    f"receipt + allocated dedicated GPU ${GPU_USD_PER_HOUR:.0f}/h "
                    f"× {wall_s:.1f}s / {n_ok} calls")
            # Drop bulky provider response bodies from the cache.
            for v in (ours, front):
                raw = dict(v.get("raw") or {})
                raw.pop("response", None)
                v["raw"] = raw
            rec = {
                "id": row["id"],
                "recorded_at": recorded_at,
                "mode": {"ours": modes.get("ours"), "frontier": modes.get("frontier")},
                "origin": origin,
                "systems": systems,
                "case": {k: row.get(k) for k in
                         ("id", "text", "image_path", "lang", "gold_label",
                          "gold_categories", "source", "perturbation")},
                "ours": ours,
                "frontier": front,
            }
            # Mark mock pollution explicitly so the UI badge stays honest.
            if _is_mockish(ours) or _is_mockish(front):
                rec["mode"] = {
                    "ours": "mock" if _is_mockish(ours) else modes.get("ours"),
                    "frontier": "mock" if _is_mockish(front) else modes.get("frontier"),
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written.append(rec)

    token_usd = sum(float((r.get("ours") or {}).get("cost_usd") or 0.0)
                    + float((r.get("frontier") or {}).get("cost_usd") or 0.0)
                    - (gpu_per if "/deployments/" in str((r.get("ours") or {}).get("model") or "")
                       else 0.0)
                    for r in written)
    # Recompute totals from the written records (includes allocated GPU).
    ours_cost = sum(float((r.get("ours") or {}).get("cost_usd") or 0.0) for r in written)
    front_cost = sum(float((r.get("frontier") or {}).get("cost_usd") or 0.0) for r in written)
    total_usd = ours_cost + front_cost

    by_id = {r["id"]: r for r in written}
    oa, on, oacc = _accuracy(cases, by_id, "ours")
    fa, fn, facc = _accuracy(cases, by_id, "frontier")
    o_lats = [float((r.get("ours") or {}).get("latency_ms") or 0.0) for r in written]
    f_lats = [float((r.get("frontier") or {}).get("latency_ms") or 0.0) for r in written]

    summary = {
        "recorded_at": recorded_at,
        "n_cases": n,
        "n_written": len(written),
        "modes": modes,
        "origin": origin,
        "systems": systems,
        "wall_s": round(wall_s, 2),
        "gpu_usd": round(gpu_usd, 4),
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        "token_usd_approx": round(token_usd, 4),
        "ours_cost_usd": round(ours_cost, 4),
        "frontier_cost_usd": round(front_cost, 4),
        "total_usd": round(total_usd, 4),
        "budget_usd": budget,
        "budget_exceeded": bool(stop.is_set()) or total_usd > budget,
        "ours": {"correct": oa, "n": on, "accuracy": round(oacc, 4),
                 "p50_ms": round(_pct(o_lats, 50), 1),
                 "p95_ms": round(_pct(o_lats, 95), 1),
                 "cost_usd": round(ours_cost, 4)},
        "frontier": {"correct": fa, "n": fn, "accuracy": round(facc, 4),
                     "p50_ms": round(_pct(f_lats, 50), 1),
                     "p95_ms": round(_pct(f_lats, 95), 1),
                     "cost_usd": round(front_cost, 4)},
        "out": str(args.out),
    }
    (args.out.parent / "record_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"recorded {len(written)}/{n} cases -> {args.out}  lanes={modes}")
    print(f"  wall={wall_s:.1f}s  gpu=${gpu_usd:.4f}  ours=${ours_cost:.4f}  "
          f"frontier=${front_cost:.4f}  total=${total_usd:.4f}  budget=${budget:.2f}")
    print(f"  ours acc={oacc:.1%} ({oa}/{on}) p50={_pct(o_lats,50):.0f}ms p95={_pct(o_lats,95):.0f}ms")
    print(f"  frontier acc={facc:.1%} ({fa}/{fn}) p50={_pct(f_lats,50):.0f}ms p95={_pct(f_lats,95):.0f}ms")

    live_ok = (modes.get("ours") == "live" and modes.get("frontier") == "live"
               and written and all(
                   (r.get("mode") or {}).get("ours") == "live"
                   and (r.get("mode") or {}).get("frontier") == "live"
                   for r in written))
    if args.fallback and live_ok:
        FALLBACK_PATH.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in written) + "\n",
            encoding="utf-8")
        FALLBACK_META.write_text(json.dumps({
            "recorded_at": recorded_at,
            "origin": origin,
            "systems": systems,
            "modes": modes,
            "n": len(written),
            "summary": summary,
            "label": "LIVE-RECORDED",
            "source_cache": str(args.out.name),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"  frozen fallback -> {FALLBACK_PATH} (+ {FALLBACK_META.name})")
    elif args.fallback:
        print("  warning: skipped fallback freeze (not a fully LIVE recording)",
              file=sys.stderr)

    if summary["budget_exceeded"]:
        sys.exit(f"BUDGET EXCEEDED: ${total_usd:.4f} > ${budget:.2f} "
                 f"(wrote {len(written)}/{n} cases; see {args.out.parent / 'record_summary.json'})")
    if args.require_live and not live_ok:
        sys.exit("error: recording is not fully LIVE — refusing to keep a mock cache")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEMO / "cases.jsonl", type=Path)
    ap.add_argument("--out", default=DEMO / "cached" / "results.jsonl", type=Path)
    ap.add_argument("--run", metavar="RUN",
                    help="record from eval receipts: results/<run> dir, a path, "
                         "or a run name under results/")
    ap.add_argument("--vision", metavar="RUN", default=None,
                    help="vision A/B run dir (vision-routed/-text-only receipts) for image cases")
    ap.add_argument("--select", choices=("auto", "cases"), default="auto",
                    help="auto (default): pick cases from the receipts unless the cases file "
                         "matches; cases: join the cases file only")
    ap.add_argument("--n-text", type=int, default=40, help="text cases to pick in auto mode")
    ap.add_argument("--n-images", type=int, default=3, help="image cases to pick in auto mode")
    ap.add_argument("--ours-system", default=None,
                    help="eval system feeding the ours lane (default: first of "
                         + ", ".join(OURS_CANDIDATES) + ")")
    ap.add_argument("--frontier-system", default=None,
                    help="eval system feeding the frontier lane (default: first of "
                         + ", ".join(FRONTIER_CANDIDATES) + ")")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel classify workers (classify mode)")
    ap.add_argument("--budget", type=float, default=12.0,
                    help="USD cap for Fireworks GPU + frontier tokens (classify mode)")
    ap.add_argument("--live", action="store_true",
                    help="set DEMO_LIVE_FRONTIER=1 before resolving backends")
    ap.add_argument("--require-live", action="store_true",
                    help="exit non-zero unless both lanes recorded LIVE")
    ap.add_argument("--fallback", action="store_true", default=True,
                    help="on a fully LIVE run, also freeze cached/fallback_recorded.jsonl")
    ap.add_argument("--no-fallback", action="store_false", dest="fallback")
    args = ap.parse_args()

    if args.run:
        record_run(args)
    else:
        record_classify(args)


if __name__ == "__main__":
    main()
