#!/usr/bin/env python3
"""Build the image-route eval sets for Safety One.

Sources (all fetched without sign-in):

  nvidia   data/nvidia/image_rows.jsonl. Wikimedia-URL train rows
           get thumbnails fetched via Special:FilePath and every usable image is
           copied + downscaled into data/vision/images/nvidia/. test/validation
           rows reference the 3.9 GB images.tar.gz bundle — left as
           image_resolved=false unless --nvidia-bundle is given (streams the
           tarball, extracts ONLY the referenced members).
  siuo     sinwang/SIUO on HF (ungated, CC-BY-NC-4.0). 167 cross-modal unsafe
           items; images are plain repo files fetched one by one.
  vlguard  ys-zong/VLGuard test.zip — GATED on HF. --vlguard is ready to run
           the moment HF_TOKEN exists; without it the step is skipped with
           instructions. (data/public/vlguard.jsonl rows already exist; this
           rewrites image_path to extracted files and emits an eval JSONL.)

Emits into data/vision/:
  nvidia_image_eval.jsonl   300 rows (subset resolvable today)
  siuo_eval.jsonl           167 rows, all unsafe
  vlguard_eval.jsonl        only when --vlguard succeeds
  manifest.json             counts, sha256, sources, licenses

Eval-row schema: {"id","text","image_path","lang","gold_label",
                            "gold_categories","source","split"} + free "meta".

Usage:
  python3 scripts/prep_vision.py                 # nvidia + siuo (+vlguard if token)
  python3 scripts/prep_vision.py --nvidia --siuo # same as default
  python3 scripts/prep_vision.py --nvidia-bundle # also stream images.tar.gz (~3.9GB)
  HF_TOKEN=… python3 scripts/prep_vision.py --vlguard

No paid calls. Reads HF_TOKEN from env or `.env` at repo root only for the gated
VLGuard zip. Disk target for data/vision/ is well under the 300 MB slice.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

SAFETY_ONE = Path(__file__).resolve().parent.parent
VISION = SAFETY_ONE / "data" / "vision"
RAW = VISION / "raw"
IMG_DIR = VISION / "images"

UA = {"User-Agent": "house-rules-data/1.0 (eval data preparation)"}
TIMEOUT = 40

SIUO_BASE = "https://huggingface.co/datasets/sinwang/SIUO/resolve/main"
SIUO_LICENSE = "CC-BY-NC-4.0"
SIUO_URL = "https://huggingface.co/datasets/sinwang/SIUO"
SIUO_PAPER = "https://arxiv.org/abs/2406.17795"

NVIDIA_LICENSE = "CC-BY-4.0"
NVIDIA_URL = "https://huggingface.co/datasets/nvidia/Nemotron-3.5-Content-Safety-Dataset"
NVIDIA_BUNDLE = NVIDIA_URL + "/resolve/main/images.tar.gz"

VLGUARD_ZIP_URL = "https://huggingface.co/datasets/ys-zong/VLGuard/resolve/main/test.zip"
VLGUARD_LICENSE = "MIT (partial — author-added per HF discussion; see data/BENCHMARKS.md)"
VLGUARD_URL = "https://huggingface.co/datasets/ys-zong/VLGuard"

MAX_SIDE = 512  # downscale target for fetched/copied eval images

WIKI_FILE = re.compile(r"commons\.wikimedia\.org/wiki/File:([^#?]+)", re.IGNORECASE)

# SIUO category -> project taxonomy slug (data/TAXONOMY.md). Best-effort;
# original SIUO category kept in meta.category.
SIUO_CAT2SLUG = {
    "self-harm": "self_harm",
    "privacy violation": "pii",
    "illegal activities & crime": "illegal_activity",
    "discrimination & stereotyping": "discrimination",
    "religion beliefs": "hate_identity",
    "morality": "immoral_unethical",
    "dangerous behavior": "other",
    "information misinterpretation": "other",
    "controversial topics, politics": "other",
}


def log(msg: str) -> None:
    print(f"[prep_vision] {msg}", flush=True)


def env_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = SAFETY_ONE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def fetch(url: str, dst: Path, token: str | None = None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    headers = dict(UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dst


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def downscale(src: Path, dst: Path, max_side: int = MAX_SIDE) -> bool:
    """Copy src -> dst downscaled to max_side, saved as JPEG. Returns success."""
    try:
        from PIL import Image
    except ImportError:
        log("Pillow not installed — copying without downscale")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return True
    try:
        img = Image.open(src)
        img.load()
        img.thumbnail((max_side, max_side))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=85)
        return True
    except Exception as e:
        log(f"  downscale failed {src.name}: {e}")
        return False


def row(rid, text, image_path, lang, label, cats, source, split, meta=None):
    r = {
        "id": rid,
        "text": text,
        "image_path": image_path,
        "lang": lang,
        "gold_label": label,
        "gold_categories": cats,
        "source": source,
        "split": split,
    }
    if meta:
        r["meta"] = meta
    return r


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# NVIDIA image rows
# --------------------------------------------------------------------------- #

def _short(rid: str) -> str:
    return rid.rsplit("-", 1)[-1][:12]


def _fetch_wikimedia_thumb(ref: str, dst: Path, width: int = 384) -> bool:
    m = WIKI_FILE.search(ref or "")
    if not m:
        return False
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{m.group(1)}?width={width}"
    tmp = dst.with_suffix(".part")
    try:
        fetch(url, tmp)
        ok = tmp.stat().st_size > 800  # tiny responses are usually error pages
        if ok:
            ok = downscale(tmp, dst)
        tmp.unlink(missing_ok=True)
        return ok
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def prep_nvidia(do_bundle: bool = False) -> dict:
    """Resolve data/nvidia/image_rows.jsonl into local images under data/vision/."""
    src = SAFETY_ONE / "data" / "nvidia" / "image_rows.jsonl"
    rows_in = [json.loads(l) for l in open(src)]
    out_dir = IMG_DIR / "nvidia"
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: what needs fetching vs copying.
    pending_wiki: list[tuple[dict, Path]] = []
    pending_bundle: list[dict] = []
    copied = 0
    for r in rows_in:
        meta = dict(r.get("meta") or {})
        ip = r.get("image_path") or ""
        dst = out_dir / f"{_short(r['id'])}.jpg"
        local = SAFETY_ONE / ip if ip and not ip.startswith(("http", "images/")) else None
        if dst.exists():
            meta["image_source"] = meta.get("image_source") or "cached"
            r["image_path"] = str(dst.relative_to(SAFETY_ONE))
            r["meta"] = meta
            copied += 1
        elif local and local.exists():
            if downscale(local, dst):
                meta["image_source"] = "nvidia_local_copy"
                r["image_path"] = str(dst.relative_to(SAFETY_ONE))
                copied += 1
            else:
                meta["image_resolved"] = False
            r["meta"] = meta
        elif ip.startswith("http"):
            pending_wiki.append((r, dst))
        else:
            meta["image_resolved"] = False
            meta["image_source"] = "nvidia_bundle_pending"
            r["image_path"] = None
            r["meta"] = meta
            pending_bundle.append(r)

    # Fetch wikimedia thumbnails concurrently (modest parallelism).
    fetched = 0
    if pending_wiki:
        def one(pair):
            rr, ddst = pair
            ref = (rr.get("meta") or {}).get("image_ref") or rr.get("image_path") or ""
            return rr, _fetch_wikimedia_thumb(ref, ddst)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for rr, ok in ex.map(one, pending_wiki):
                if ok:
                    rr["image_path"] = str((IMG_DIR / "nvidia" / f"{_short(rr['id'])}.jpg").relative_to(SAFETY_ONE))
                    rr["meta"]["image_source"] = "wikimedia_thumb"
                    rr["meta"]["image_resolved"] = True
                    fetched += 1
                else:
                    rr["image_path"] = None
                    rr["meta"]["image_resolved"] = False
                    rr["meta"]["image_source"] = "wikimedia_failed"
    log(f"nvidia: {copied} copied/cached, {fetched} wikimedia fetched, "
        f"{len(pending_bundle)} bundle-pending")

    # Optional: stream the 3.9 GB tarball, extracting only referenced members.
    bundle_done = 0
    if do_bundle and pending_bundle:
        bundle_done = _stream_extract_bundle(pending_bundle)
        log(f"nvidia: bundle extraction resolved {bundle_done}/{len(pending_bundle)}")

    out_path = VISION / "nvidia_image_eval.jsonl"
    write_jsonl(out_path, rows_in)
    return {
        "file": out_path.name, "rows": len(rows_in),
        "resolved": sum(1 for r in rows_in if r.get("image_path")),
        "bundle_pending": sum(1 for r in rows_in
                              if (r.get("meta") or {}).get("image_source") == "nvidia_bundle_pending"),
    }


def _stream_extract_bundle(pending: list[dict]) -> int:
    """Stream images.tar.gz (~3.9 GB); write only the refs we need."""
    wanted: dict[str, dict] = {}
    for r in pending:
        ref = (r.get("meta") or {}).get("image_ref") or ""
        split = r.get("split", "")
        wanted[f"images/{split.replace('-img', '')}/{ref}"] = r
        wanted[f"{split.replace('-img', '')}/{ref}"] = r
    if not wanted:
        return 0
    log(f"nvidia-bundle: streaming {NVIDIA_BUNDLE} for {len(wanted)} refs "
        "(downloads ~3.9 GB; writes only matched members)")
    out_dir = IMG_DIR / "nvidia"
    done = 0
    req = urllib.request.Request(NVIDIA_BUNDLE, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for member in tf:
                name = member.name.lstrip("./")
                hit = wanted.get(name)
                if hit is None or not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                dst = out_dir / f"{_short(hit['id'])}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(f.read())
                if downscale(tmp, dst):
                    hit["image_path"] = str(dst.relative_to(SAFETY_ONE))
                    hit["meta"]["image_resolved"] = True
                    hit["meta"]["image_source"] = "nvidia_bundle_extract"
                    done += 1
                tmp.unlink(missing_ok=True)
                if done >= len({k for k in wanted}):
                    break
    return done


# --------------------------------------------------------------------------- #
# SIUO
# --------------------------------------------------------------------------- #

def prep_siuo() -> dict:
    raw_dir = RAW / "siuo"
    gen = fetch(f"{SIUO_BASE}/siuo_gen.json", raw_dir / "siuo_gen.json")
    items = json.load(open(gen))

    out_dir = IMG_DIR / "siuo"
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(it):
        name = it["image"]
        dst = out_dir / (Path(name).stem + ".jpg")
        if not dst.exists():
            raw_img = fetch(f"{SIUO_BASE}/images/{name}", raw_dir / "images" / name)
            downscale(raw_img, dst)
            # raw PNGs are regenerable; do not keep them against the disk cap
            raw_img.unlink(missing_ok=True)
        return dst if dst.exists() else None

    rows, missing = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        dsts = list(ex.map(one, items))
    for it, dst in zip(items, dsts):
        if dst is None:
            missing += 1
            image_path = None
        else:
            image_path = str(dst.relative_to(SAFETY_ONE))
        cat = it.get("category", "")
        rows.append(row(
            f"siuo-{it['question_id']}", it["question"], image_path, "en",
            "unsafe", [SIUO_CAT2SLUG.get(cat, "other")],
            "siuo", "test",
            meta={
                "category": cat,
                "safety_warning": it.get("safety_warning", ""),
                "image_source": "siuo_hf" if dst else "siuo_missing",
                "image_resolved": dst is not None,
            },
        ))
    out_path = VISION / "siuo_eval.jsonl"
    write_jsonl(out_path, rows)
    log(f"siuo: {len(rows)} rows, {missing} images missing")
    return {"file": out_path.name, "rows": len(rows), "missing_images": missing}


# --------------------------------------------------------------------------- #
# VLGuard (gated — ready once HF_TOKEN exists)
# --------------------------------------------------------------------------- #

def prep_vlguard() -> dict:
    tok = env_token()
    if not tok:
        log("vlguard: SKIPPED — gated on HF. Re-run with HF_TOKEN set "
            "(or accept terms at https://huggingface.co/datasets/ys-zong/VLGuard). "
            "Rows already exist in data/public/vlguard.jsonl.")
        return {"file": "vlguard_eval.jsonl", "rows": 0, "status": "gated-skip"}
    zpath = fetch(VLGUARD_ZIP_URL, RAW / "vlguard" / "test.zip", token=tok)
    out_dir = IMG_DIR / "vlguard"
    extracted = 0
    by_stem: dict[str, Path] = {}
    by_rel: dict[str, Path] = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.endswith("/") or name.startswith("__MACOSX"):
                continue
            rel = Path(name)
            if rel.parts and rel.parts[0] in ("test", "images"):
                rel = Path(*rel.parts[1:])
            dst = out_dir / rel.parent / (rel.stem + ".jpg")
            if not dst.exists():
                tmp = dst.with_suffix(".part")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(z.read(name))
                downscale(tmp, dst)
                tmp.unlink(missing_ok=True)
            by_stem.setdefault(rel.stem, dst)
            by_rel.setdefault(str(rel), dst)
            extracted += 1
    zpath.unlink(missing_ok=True)  # zip is regenerable; keep the disk budget
    # Rewrite the public vlguard rows to point at extracted files.
    src = SAFETY_ONE / "data" / "public" / "vlguard.jsonl"
    rows = [json.loads(l) for l in open(src)]
    resolved = 0
    for r in rows:
        ip = r.get("image_path") or ""
        rel = ip.split("images/", 1)[-1] if "images/" in ip else ""
        cand = by_rel.get(rel) or by_rel.get(str(Path(rel))) \
            or by_stem.get(Path(ip).stem)
        if cand and cand.exists():
            r["image_path"] = str(cand.relative_to(SAFETY_ONE))
            r.setdefault("meta", {})["image_resolved"] = True
            resolved += 1
        else:
            r["image_path"] = None
            r.setdefault("meta", {})["image_resolved"] = False
    out_path = VISION / "vlguard_eval.jsonl"
    write_jsonl(out_path, rows)
    log(f"vlguard: extracted {extracted} files; {resolved}/{len(rows)} rows resolved")
    return {"file": out_path.name, "rows": len(rows), "resolved": resolved}


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nvidia", action="store_true")
    ap.add_argument("--nvidia-bundle", action="store_true",
                    help="stream images.tar.gz (~3.9 GB) and extract test/val refs")
    ap.add_argument("--siuo", action="store_true")
    ap.add_argument("--vlguard", action="store_true",
                    help="needs HF_TOKEN; pulls the gated test.zip")
    args = ap.parse_args()

    explicit = args.nvidia or args.siuo or args.vlguard
    do_nvidia = args.nvidia or not explicit
    do_siuo = args.siuo or not explicit
    do_vlguard = args.vlguard or not explicit

    started = time.time()
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "scripts/prep_vision.py",
        "sources": {
            "nvidia": {"url": NVIDIA_URL, "license": NVIDIA_LICENSE,
                       "notes": "image rows via data/nvidia/image_rows.jsonl; "
                                "wikimedia thumbs via Special:FilePath; test/val "
                                "refs live in images.tar.gz (3.9 GB, CC-BY-4.0)"},
            "siuo": {"url": SIUO_URL, "license": SIUO_LICENSE,
                     "paper": SIUO_PAPER,
                     "notes": "siuo_gen.json + images/*.png fetched individually"},
            "vlguard": {"url": VLGUARD_URL, "license": VLGUARD_LICENSE,
                        "notes": "test.zip GATED — needs HF_TOKEN + accepted terms"},
        },
        "params": {"max_side": MAX_SIDE, "nvidia_bundle": args.nvidia_bundle},
        "outputs": {},
    }

    if do_nvidia:
        manifest["outputs"]["nvidia"] = prep_nvidia(do_bundle=args.nvidia_bundle)
    if do_siuo:
        manifest["outputs"]["siuo"] = prep_siuo()
    if do_vlguard:
        manifest["outputs"]["vlguard"] = prep_vlguard()

    for f in sorted(VISION.glob("*_eval.jsonl")):
        key = f.name
        manifest["outputs"].setdefault("files", {})[key] = {
            "bytes": f.stat().st_size, "sha256": sha256(f),
        }
    manifest["elapsed_s"] = round(time.time() - started, 1)
    (VISION / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log("manifest.json written")


if __name__ == "__main__":
    main()
