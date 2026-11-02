"""Moderation race demo server — bound to 127.0.0.1, stdlib only.

  python3 demo/server.py                     # replay cached/results.jsonl
  python3 demo/server.py --replay-fallback   # replay frozen LIVE-RECORDED copy
  python3 demo/server.py --live              # classify each case now
  python3 demo/server.py --record PATH       # record live results to PATH, exit
  python3 demo/server.py --port 8787

Open http://127.0.0.1:8787/ — single page, no build step, works offline.

Endpoints:
  GET /            index.html
  GET /assets/...  static assets (e.g. the meme image)
  GET /api/meta    {"mode","lanes","n_cases","cache_live",...}
  GET /case-img/.. images of image cases (data/vision/images only)
  GET /api/race?n= server-sent events: meta -> per-case {case, verdict x2}
                    -> receipt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
sys.path.insert(0, str(DEMO))
import backends                      # noqa: E402
from record import load_cases        # noqa: E402

HOST = "127.0.0.1"
CASES_PATH = DEMO / "cases.jsonl"
CACHE_PATH = DEMO / "cached" / "results.jsonl"
FALLBACK_CACHE = DEMO / "cached" / "fallback_recorded.jsonl"
FALLBACK_META = DEMO / "cached" / "fallback_meta.json"
PROD_BASIS_PATH = DEMO / "cached" / "production_basis.json"
PAGE = DEMO / "index.html"
IMAGE_ROOT = ROOT / "data" / "vision" / "images"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".gif": "image/gif"}
OURS_SUMMARY = ROOT / "results" / "nvidia-v1" / "ours_summary.json"
FINAL_HEADLINE = ROOT / "results" / "nvidia-v1" / "final_headline.json"
VS_FRONTIER = ROOT / "results" / "stats" / "vs_frontier.json"

# Active replay cache path — flipped by --replay-fallback.
ACTIVE_CACHE = CACHE_PATH


def _escalation_stats(cache: dict) -> dict | None:
    """Count Ultra escalations + latencies from a replay cache (ours lane)."""
    esc, non_max = [], 0.0
    n = 0
    for rec in cache.values():
        ours = rec.get("ours") or {}
        if ours.get("skipped"):
            continue
        n += 1
        lat = float(ours.get("latency_ms") or 0)
        route = str(ours.get("route") or "")
        if "escalat" in route.lower():
            esc.append({"id": rec.get("id"), "latency_ms": round(lat, 1),
                        "route": route})
        else:
            non_max = max(non_max, lat)
    if not n:
        return None
    esc.sort(key=lambda x: -x["latency_ms"])
    return {
        "n": len(esc),
        "of": n,
        "latency_ms_min": round(min(e["latency_ms"] for e in esc)) if esc else None,
        "latency_ms_max": round(max(e["latency_ms"] for e in esc)) if esc else None,
        "ids": [e["id"] for e in esc],
        "latencies_ms": [e["latency_ms"] for e in esc],
        "non_escalated_max_ms": round(non_max) if n > len(esc) else None,
    }


def load_production_basis(cache: dict | None = None) -> dict:
    """Production $/1M + p95 from results/, demo-run notes from cached JSON.

    Prefers live reads of results/nvidia-v1/* and results/stats/vs_frontier.json
    so the UI cannot disagree with receipts; falls back to the shipped
    demo/cached/production_basis.json. Escalation counts always come from the
    active replay cache when available.
    """
    basis: dict = {}
    if PROD_BASIS_PATH.exists():
        try:
            basis = json.loads(PROD_BASIS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            basis = {}

    prod = dict(basis.get("production") or {})
    ours_prod = dict(prod.get("ours") or {})
    front_prod = dict(prod.get("frontier") or {})
    # Refresh from receipts when present (never invent / never favor-round).
    try:
        if FINAL_HEADLINE.exists():
            fh = json.loads(FINAL_HEADLINE.read_text(encoding="utf-8"))
            nt = ((fh.get("final") or {}).get("nvidia_test") or {})
            if nt.get("usd_per_1m_msgs") is not None:
                ours_prod["usd_per_1m_msgs"] = round(float(nt["usd_per_1m_msgs"]), 1)
            if nt.get("p50_ms") is not None:
                ours_prod["p50_ms"] = round(float(nt["p50_ms"]))
            if nt.get("p95_ms") is not None:
                ours_prod["p95_ms"] = round(float(nt["p95_ms"]))
            if nt.get("n") is not None:
                ours_prod["n"] = int(nt["n"])
            ours_prod.setdefault("set", "nvidia_test")
        elif OURS_SUMMARY.exists():
            os_ = json.loads(OURS_SUMMARY.read_text(encoding="utf-8"))
            nt = (((os_.get("systems") or {}).get("final") or {})
                  .get("nvidia_test") or {})
            if nt.get("usd_per_1m_msgs") is not None:
                ours_prod["usd_per_1m_msgs"] = round(float(nt["usd_per_1m_msgs"]), 1)
            if nt.get("p95_ms") is not None:
                ours_prod["p95_ms"] = round(float(nt["p95_ms"]))
            if nt.get("p50_ms") is not None:
                ours_prod["p50_ms"] = round(float(nt["p50_ms"]))
            if nt.get("n") is not None:
                ours_prod["n"] = int(nt["n"])
            ours_prod.setdefault("set", "nvidia_test")
        if VS_FRONTIER.exists():
            vf = json.loads(VS_FRONTIER.read_text(encoding="utf-8"))
            base = (((vf.get("comparisons") or {}).get("ours_vs_gpt") or {})
                    .get("baseline") or {})
            if base.get("usd_per_1m_msgs") is not None:
                front_prod["usd_per_1m_msgs"] = round(float(base["usd_per_1m_msgs"]))
            if base.get("latency_p50_ms") is not None:
                front_prod["p50_ms"] = round(float(base["latency_p50_ms"]))
            if base.get("latency_p95_ms") is not None:
                front_prod["p95_ms"] = round(float(base["latency_p95_ms"]))
            if base.get("n") is not None:
                front_prod["n"] = int(base["n"])
            front_prod.setdefault("label", "GPT-6.1 Sol")
            front_prod.setdefault("set", "nvidia_test")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    prod["ours"] = ours_prod
    prod["frontier"] = front_prod
    prod.setdefault("label", "production basis (from results/)")
    prod.setdefault(
        "note",
        "GPU rental amortized at full utilization over measured run wall-clock "
        "÷ model calls (concurrency 48), plus Ultra escalation tokens; "
        "spin-up/idle excluded.",
    )
    basis["production"] = prod

    demo_run = dict(basis.get("demo_run") or {})
    demo_run.setdefault(
        "label",
        "this 100-message demo run (low utilization, GPU time ÷ 100 msgs)",
    )
    if cache:
        esc = _escalation_stats(cache)
        if esc:
            demo_run["escalations"] = esc
            demo_run["n"] = esc["of"]
    basis["demo_run"] = demo_run
    return basis


def _json(handler, obj, status=200):
    body = json.dumps(obj, ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _file(handler, path, ctype):
    try:
        body = path.read_bytes()
    except OSError:
        handler.send_error(404)
        return
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def sse(handler, event, obj):
    data = json.dumps(obj, ensure_ascii=False)
    handler.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
    handler.wfile.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HouseRulesDemo/1.0"

    def log_message(self, fmt, *args):  # quiet; keep one line per request
        sys.stderr.write("  %s %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return _file(self, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/meta":
            first = (next(iter(self.server.cache.values()), {})
                     if self.server.mode == "replay" else {})
            cache_path = Path(self.server.cache_path) if self.server.cache_path else None
            return _json(self, {
                "mode": self.server.mode,
                "lanes": self.server.lane_modes,
                "n_cases": len(self.server.cases),
                "case_source": (cache_path.name if self.server.cases_from_cache and cache_path
                                else str(CASES_PATH.name)),
                "cache": cache_path.name if cache_path and cache_path.exists() else None,
                "replay_fallback": bool(self.server.replay_fallback),
                # provenance of the replay cache: per-lane mode (mock/live),
                # where it came from ("classify" or "run:<name>"), which eval
                # systems fed the lanes. Used to label MOCK vs LIVE results.
                "cache_mode": first.get("mode"),
                "cache_origin": first.get("origin"),
                "cache_systems": first.get("systems"),
                "cache_recorded_at": first.get("recorded_at") or self.server.fallback_meta.get(
                    "recorded_at"),
                "cache_label": self.server.fallback_meta.get("label") or (
                    "LIVE-RECORDED" if self.server.cache_live else None),
                # True only when every lane of every cached record is LIVE
                # (lanes marked "none" = not run). The UI's MOCK badge keys off it.
                "cache_live": self.server.cache_live,
                "cases_from_cache": self.server.cases_from_cache,
                # Cost/latency context: production-basis $/1M from results/,
                # demo-run spend + Ultra escalation tail from the cache.
                "production_basis": self.server.production_basis,
            })
        if u.path == "/api/race":
            return self._race(u)
        if u.path.startswith("/case-img/"):
            root = IMAGE_ROOT.resolve()
            target = (root / unquote(u.path[len("/case-img/"):])).resolve()
            if root not in target.parents or target.suffix.lower() not in IMAGE_TYPES:
                return self.send_error(403)
            return _file(self, target, IMAGE_TYPES[target.suffix.lower()])
        if u.path.startswith("/assets/"):
            name = u.path[len("/assets/"):]
            target = (DEMO / "assets" / name).resolve()
            if target.parent != (DEMO / "assets").resolve():
                return self.send_error(403)
            return _file(self, target, "image/svg+xml" if name.endswith(".svg")
                         else "application/octet-stream")
        self.send_error(404)

    # ------------------------------------------------------------------ SSE
    def _race(self, u):
        n = int(parse_qs(u.query).get("n", ["0"])[0] or 0)
        cases = self.server.cases[:n] if n else self.server.cases
        mode = self.server.mode
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            sse(self, "meta", {"mode": mode, "lanes": self.server.lane_modes,
                               "n": len(cases), "ts": round(time.time(), 3)})
            cache = self.server.cache if mode == "replay" else {}
            for row in cases:
                rec = cache.get(row["id"], {}) if mode == "replay" else {}
                if mode == "replay":
                    res = {"ours": rec.get("ours"), "frontier": rec.get("frontier")}
                    if res["ours"] is None:
                        res = backends.classify_both(row)
                    elif res["frontier"] is None:
                        # lane not run for this case (e.g. image rows: the
                        # frontier baseline was text-only) — say so, no metrics
                        res["frontier"] = {
                            "skipped": True, "id": row["id"], "label": "",
                            "route": "not-run", "model": "", "latency_ms": 0.0,
                            "cost_usd": 0.0, "categories": [], "probs": {},
                            "note": rec.get("frontier_note") or "not run for this case"}
                    time.sleep(0.07)          # pacing so the feed is readable
                else:
                    res = backends.classify_both(row)
                sse(self, "case", {
                    "id": row["id"], "text": row["text"],
                    "image_path": row.get("image_path"), "lang": row.get("lang"),
                    "gold_label": row["gold_label"],
                    "gold_categories": row.get("gold_categories", []),
                    "source": row.get("source", ""),
                    "perturbation": row.get("perturbation"),
                    "tag": rec.get("tag") or row.get("tag"),
                    "others": rec.get("others") or []})
                # emit verdicts in finish order so the faster lane lands first
                for lane in sorted(("ours", "frontier"),
                                   key=lambda l: (bool(res[l].get("skipped")),
                                                  res[l]["latency_ms"])):
                    sse(self, "verdict", dict(res[lane], lane=lane, id=row["id"]))
            sse(self, "receipt", {"n": len(cases), "ts": round(time.time(), 3)})
        except (BrokenPipeError, ConnectionResetError):
            pass                                # client closed the stream


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--live", action="store_true",
                    help="classify now instead of replaying the cache")
    ap.add_argument("--replay-fallback", action="store_true",
                    help="replay cached/fallback_recorded.jsonl (frozen LIVE "
                         "recording for stage use when live calls aren't available)")
    ap.add_argument("--cache", metavar="PATH", type=Path, default=None,
                    help="explicit replay cache JSONL (overrides default / fallback)")
    ap.add_argument("--record", metavar="PATH",
                    help="record live results to PATH (JSONL) and exit")
    args = ap.parse_args()

    if args.record:
        import record
        sys.argv = ["record.py", "--cases", str(CASES_PATH), "--out", args.record,
                    "--live", "--require-live", "--fallback"]
        record.main()
        return

    if args.live and args.replay_fallback:
        sys.exit("error: --live and --replay-fallback are mutually exclusive")

    cache_path = args.cache
    if cache_path is None and args.replay_fallback:
        cache_path = FALLBACK_CACHE
        if not cache_path.exists():
            sys.exit(f"error: fallback cache missing: {cache_path} "
                     "(run demo/record.py --live --require-live --fallback first)")
    srv = make_server(args.port, live=args.live, cache_path=cache_path,
                      replay_fallback=args.replay_fallback)
    print(f"house-rules demo  http://{HOST}:{args.port}/  mode={srv.mode} "
          f"lanes={srv.lane_modes}  cases={len(srv.cases)}  "
          f"cache={srv.cache_path}  (ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def make_server(port, live=False, cache_path=None, replay_fallback=False):
    """Configured (not yet serving) ThreadingHTTPServer on 127.0.0.1."""
    cases = load_cases(CASES_PATH)
    path = Path(cache_path) if cache_path else CACHE_PATH
    mode = "live" if live or not path.exists() else "replay"
    cache = {}
    fallback_meta = {}
    if mode == "replay":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["id"]] = r
        if replay_fallback or path == FALLBACK_CACHE:
            try:
                fallback_meta = json.loads(FALLBACK_META.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                fallback_meta = {}
    # Receipt-recorded / live-classify caches carry their own case rows; replay
    # those instead of demo/cases.jsonl so fallback stays self-contained.
    from_cache = bool(cache) and all("case" in r for r in cache.values())
    if from_cache:
        # Preserve file order (dict insertion order from the JSONL walk).
        cases = [r["case"] for r in cache.values()]
    lanes = [str(v).lower() for r in cache.values()
             for v in (r.get("mode") or {}).values() if str(v).lower() != "none"]
    srv = ThreadingHTTPServer((HOST, port), Handler)
    srv.cases, srv.cache, srv.mode = cases, cache, mode
    srv.cache_path = str(path) if mode == "replay" else None
    srv.replay_fallback = bool(replay_fallback or (path == FALLBACK_CACHE))
    srv.fallback_meta = fallback_meta
    srv.cache_live = bool(lanes) and all(v == "live" for v in lanes)
    srv.cases_from_cache = from_cache
    srv.lane_modes = backends.lane_modes()
    srv.production_basis = load_production_basis(cache if mode == "replay" else None)
    return srv


if __name__ == "__main__":
    main()
