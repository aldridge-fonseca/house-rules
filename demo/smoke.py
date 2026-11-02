"""Smoke test: bring the server up on an ephemeral port, hit every endpoint,
consume one SSE race (n=6) through the receipt, tear down. Leaves nothing
running.

  python3 demo/smoke.py                   # replay cached/results.jsonl
  python3 demo/smoke.py --replay-fallback # replay cached/fallback_recorded.jsonl
  python3 demo/smoke.py --live            # exercise the live classify path
"""
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402

DEMO = Path(__file__).resolve().parent
BANNED = ("wildguard", "vlguard", "siuo")


def _assert_shipped_images(cases):
    """Every image_path must resolve to a file under demo/assets/ (shareable)."""
    for c in cases:
        ip = str(c.get("image_path") or "")
        if not ip:
            continue
        low = ip.lower()
        for term in BANNED:
            assert term not in low, f"banned image ref in case {c.get('id')}: {ip}"
        # Only /assets/<file> is allowed in the shippable demo.
        assert ip.startswith("/assets/"), (
            f"case {c.get('id')} image_path {ip!r} is not a shipped /assets/ URL "
            "(no /case-img/ SIUO/VLGuard paths in the public demo)")
        name = Path(ip).name
        assert ".." not in ip and "/" not in name
        path = DEMO / "assets" / name
        assert path.is_file(), f"missing shipped asset for {c.get('id')}: {path}"


def main():
    live = "--live" in sys.argv
    fallback = "--replay-fallback" in sys.argv
    cache = server.FALLBACK_CACHE if fallback else None
    if fallback and not server.FALLBACK_CACHE.exists():
        sys.exit(f"fallback cache missing: {server.FALLBACK_CACHE}")
    if not live and cache is None and not server.CACHE_PATH.exists():
        sys.exit(f"replay cache missing: {server.CACHE_PATH} "
                 "(run demo/record.py --live --require-live --fallback first)")
    srv = server.make_server(0, live=live, cache_path=cache, replay_fallback=fallback)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        html = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        assert ("HOUSE RULES" in html or "House Rules" in html) and "/api/race" in html
        assert "Built on Fireworks AI" in html and "NVIDIA Nemotron" in html
        assert "firectl" not in html.lower()
        assert "Super VL" not in html
        meta = json.loads(urllib.request.urlopen(base + "/api/meta",
                                                 timeout=10).read())
        assert meta["n_cases"] > 0
        pb = meta.get("production_basis") or {}
        prod = pb.get("production") or {}
        ours_p = (prod.get("ours") or {}).get("usd_per_1m_msgs")
        gpt_p = (prod.get("frontier") or {}).get("usd_per_1m_msgs")
        assert ours_p is not None and abs(float(ours_p) - 56.2) < 0.05, ours_p
        assert gpt_p is not None and abs(float(gpt_p) - 1820) < 1, gpt_p
        esc = ((pb.get("demo_run") or {}).get("escalations") or {})
        if meta["mode"] == "replay" and esc:
            assert esc.get("n") == 7 and esc.get("of") == 100, esc
            assert esc.get("non_escalated_max_ms") == 425, esc
        svg = urllib.request.urlopen(base + "/assets/meme1.svg",
                                     timeout=10).read()
        assert b"<svg" in svg

        _assert_shipped_images(srv.cases)
        # Also scan the raw cache for banned ids / paths (fresh-clone safety).
        for rec in srv.cache.values():
            blob = json.dumps(rec, ensure_ascii=False).lower()
            for term in BANNED:
                assert term not in blob, f"banned {term!r} in cache record {rec.get('id')}"

        if meta["mode"] == "replay":
            lanes = [str(v).lower() for r in srv.cache.values()
                     for v in (r.get("mode") or {}).values() if str(v).lower() != "none"]
            expect_live = bool(lanes) and all(v == "live" for v in lanes)
            assert meta["cache_live"] is expect_live, (meta["cache_live"], expect_live)
            # Fetch every shipped case image (≤3) over HTTP.
            imgs = [c["image_path"] for c in srv.cases if c.get("image_path")]
            for path in imgs[:3]:
                with urllib.request.urlopen(base + urllib.parse.quote(path), timeout=10) as r:
                    body = r.read(16)
                    ctype = r.headers.get("Content-Type", "")
                    assert body and (ctype.startswith("image/") or "svg" in ctype), (path, ctype)
            try:
                urllib.request.urlopen(base + "/case-img/../../.env", timeout=10)
                raise AssertionError("path traversal not blocked")
            except urllib.error.HTTPError as e:
                assert e.code in (403, 404), e.code

        got = {"case": 0, "verdict": 0, "receipt": False}
        with urllib.request.urlopen(base + "/api/race?n=6", timeout=30) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line == "event: case":
                    got["case"] += 1
                elif line == "event: verdict":
                    got["verdict"] += 1
                elif line == "event: receipt":
                    got["receipt"] = True
                    break
        assert got["case"] == 6 and got["verdict"] == 12 and got["receipt"], got
        print(f"SMOKE OK  mode={meta['mode']} lanes={meta['lanes']} "
              f"cache_live={meta.get('cache_live')} origin={meta.get('cache_origin')} "
              f"streamed {got['case']} cases / {got['verdict']} verdicts")
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
