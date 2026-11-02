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
sys.path.insert(0, str(DEMO))
import backends                      # noqa: E402
from record import load_cases        # noqa: E402

HOST = "127.0.0.1"
CASES_PATH = DEMO / "cases.jsonl"
CACHE_PATH = DEMO / "cached" / "results.jsonl"
FALLBACK_CACHE = DEMO / "cached" / "fallback_recorded.jsonl"
FALLBACK_META = DEMO / "cached" / "fallback_meta.json"
PAGE = DEMO / "index.html"
IMAGE_ROOT = DEMO.parent / "data" / "vision" / "images"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".gif": "image/gif"}

# Active replay cache path — flipped by --replay-fallback.
ACTIVE_CACHE = CACHE_PATH


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
    server_version = "SafetyOneDemo/1.0"

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
    print(f"safety-one demo  http://{HOST}:{args.port}/  mode={srv.mode} "
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
    return srv


if __name__ == "__main__":
    main()
