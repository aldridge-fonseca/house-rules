#!/usr/bin/env python3
"""Prepare the NVIDIA Nemotron-3.5-Content-Safety-Dataset for Safety One.

Downloads the three parquet splits (~67 MB total) into data/nvidia/raw/ (gitignored)
and emits, into data/nvidia/:

  train_sft.jsonl    Fireworks chat-SFT rows; assistant = strict JSON label
  train_kway.jsonl   decision_loop rows (format validated by the r2 converter)
  dev_calib.jsonl    held-out-from-train calibration rows (eval schema)
  validation.jsonl   NVIDIA validation split, text rows (eval schema, extra holdout)
  test.jsonl         NVIDIA test split, text rows (eval schema)
  image_rows.jsonl   capped image rows (URLs / small local files / bundle refs)
  manifest.json      counts, sha256, sources, license, leakage report

Eval-row schema: {"id","text","image_path","lang","gold_label",
                            "gold_categories","source","split"}

Usage:
  python3 scripts/prep_nvidia.py                 # default caps
  python3 scripts/prep_nvidia.py --limit 5000    # smaller train subsets
  python3 scripts/prep_nvidia.py --limit 0       # all available train rows

No paid/API calls. Reads HF_TOKEN / `.env` only if the dataset ever goes
gated (it is public, CC-BY-4.0).
"""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

REPO = "nvidia/Nemotron-3.5-Content-Safety-Dataset"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
LICENSE = "CC-BY-4.0"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO}"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nvidia"
RAW = DATA_DIR / "raw"

SYSTEM_PROMPT = (
    "You are a content-safety classifier. Reply only with a JSON object "
    '{"label": "safe"|"unsafe", "categories": [slug, ...]} using the project taxonomy.'
)

CHOICE_OPTIONS = {
    "safe": "The user prompt does not request disallowed or harmful assistance.",
    "unsafe": "The user prompt requests disallowed or harmful assistance.",
}
CHOICE_INSTRUCTIONS = (
    "Is this user request safe or unsafe under the dataset's content-safety label?"
)

# NVIDIA violated_categories string -> project slug (see data/TAXONOMY.md)
CAT2SLUG = {
    "Criminal Planning/Confessions": "criminal_planning",
    "Violence": "violence",
    "Hate/Identity Hate": "hate_identity",
    "Harassment": "harassment",
    "Needs Caution": "needs_caution",
    "Controlled/Regulated Substances": "controlled_substances",
    "Profanity": "profanity",
    "Sexual": "sexual",
    "PII/Privacy": "pii",
    "Guns and Illegal Weapons": "weapons",
    "Suicide and Self Harm": "self_harm",
    "Fraud/Deception": "fraud_deception",
    "Illegal Activity": "illegal_activity",
    "Sexual (minor)": "sexual_minor",
    "Other": "other",
    "Immoral/Unethical": "immoral_unethical",
    "Political/Misinformation/Conspiracy": "misinformation",
    "Threat": "threat",
    "Unauthorized Advice": "unauthorized_advice",
    "Manipulation": "manipulation",
    "Malware": "malware",
    "High Risk Gov Decision Making": "high_risk_gov",
    "Copyright/Trademark/Plagiarism": "copyright",
    "Economic Harm": "economic_harm",
    "Discrimination": "discrimination",
    "Environmental Harm": "environmental_harm",
}

WIKI_FILE = re.compile(r"commons\.wikimedia\.org/wiki/File:([^#?]+)", re.IGNORECASE)
MASK64 = (1 << 64) - 1


def log(msg: str) -> None:
    print(f"[prep_nvidia {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_env_token() -> str | None:
    """HF_TOKEN from env or `.env` at repo root (never printed)."""
    import os

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env_file = DATA_DIR.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def download(url: str, dst: Path, token: str | None = None) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "house-rules-data/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    log(f"download {url} -> {dst.name}")
    with urllib.request.urlopen(req, timeout=120) as r, dst.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def normalize(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text).lower().split()
    )


def shingles(text: str) -> set[int]:
    """Hashed word-3-gram set for Jaccard near-dup checks."""
    w = normalize(text).split()
    out = set()
    for i in range(len(w)):
        if i + 3 <= len(w):
            g = " ".join(w[i : i + 3])
        else:
            g = " ".join(w[i:])  # short docs: tail unigram/bigram
        out.add(int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8,
                                               person=b"s1shg").digest(), "little"))
    return out


def minhash_sig(sh: set[int], k: int = 6) -> tuple[int, ...]:
    """Cheap k-min signature via h_i = h1 + i*h2 (mod 2^64)."""
    mins = [MASK64] * k
    for h in sh:
        h2 = ((h * 0x9E3779B97F4A7C15) ^ (h >> 29)) & MASK64 or 1
        for i in range(k):
            v = (h + i * h2) & MASK64
            if v < mins[i]:
                mins[i] = v
    return tuple(mins)


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_dup_clusters(docs: list[set[int]], thresh: float = 0.8):
    """MinHash(6) banded LSH -> verified Jaccard clusters. Returns (dsu, candidates)."""
    sigs = [minhash_sig(s) for s in docs]
    dsu = DSU(len(docs))
    bands = {}
    for i, s in enumerate(sigs):
        for key in (("a", s[0], s[1], s[2]), ("b", s[3], s[4], s[5])):
            bands.setdefault(key, []).append(i)
    checked = set()
    for idxs in bands.values():
        if len(idxs) < 2:
            continue
        if len(idxs) > 2000:  # pathological bucket: union all, don't O(n^2) verify
            for j in idxs[1:]:
                dsu.union(idxs[0], j)
            continue
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                a, b = idxs[x], idxs[y]
                pair = (a, b) if a < b else (b, a)
                if pair in checked:
                    continue
                checked.add(pair)
                sa, sb = docs[a], docs[b]
                j = len(sa & sb) / max(1, len(sa | sb))
                if j >= thresh:
                    dsu.union(a, b)
    clusters = {}
    for i in range(len(docs)):
        clusters.setdefault(dsu.find(i), []).append(i)
    return [v for v in clusters.values() if len(v) > 1]


def cats_to_slugs(vc: str | None) -> list[str]:
    if not vc:
        return []
    out = []
    for c in vc.split(","):
        slug = CAT2SLUG.get(c.strip())
        out.append(slug if slug else "other")
    return out


def eval_row(split: str, r: dict) -> dict:
    return {
        "id": f"nv-{split}-{r['row_id']}",
        "text": r["prompt"],
        "image_path": r.get("image_local") or None,
        "lang": r["language"],
        "gold_label": r["input_label"],
        "gold_categories": cats_to_slugs(r["violated_categories"]),
        "source": "nvidia-nemotron-cs-3.5",
        "split": split,
    }


def sft_row(r: dict) -> dict:
    label = {"label": r["input_label"], "categories": cats_to_slugs(r["violated_categories"])}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False,
                                                       separators=(",", ":"))},
        ]
    }


def kway_row(r: dict) -> dict:
    return {
        "state": r["prompt"],
        "questions": {
            "input_safety": {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": CHOICE_OPTIONS,
                "label": r["input_label"],
            }
        },
    }


def write_jsonl(path: Path, rows) -> dict:
    h = hashlib.sha256()
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            h.update(line.encode())
            f.write(line)
            n += 1
    return {"rows": n, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def fetch_wikimedia(rows: list[dict], out_dir: Path, max_n: int, width: int = 384) -> int:
    """Fetch thumbnails for wikimedia-URL rows; sets r['image_local'] on success."""
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for r in rows:
        m = WIKI_FILE.search(r["image_path"] or "")
        if m:
            targets.append((r, m.group(1)))
        if len(targets) >= max_n:
            break
    done = 0

    def one(t):
        r, name = t
        url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{name}?width={width}"
        dst = out_dir / name.replace("/", "_")
        if dst.exists() and dst.stat().st_size > 500:
            r["image_local"] = str(dst.relative_to(DATA_DIR.parent.parent))
            return 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "house-rules-data/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read(2_000_000)
            if len(data) > 500:
                dst.write_bytes(data)
                r["image_local"] = str(dst.relative_to(DATA_DIR.parent.parent))
                return 1
        except Exception:
            pass
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        done = sum(ex.map(one, targets))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=20000,
                    help="max train rows (0 = all). default 20000")
    ap.add_argument("--dev-size", type=int, default=2500,
                    help="dev/calibration rows held out of train pool. default 2500")
    ap.add_argument("--image-cap", type=int, default=300,
                    help="max rows in image_rows.jsonl. default 300")
    ap.add_argument("--image-download", type=int, default=60,
                    help="wikimedia thumbnails to fetch locally. default 60")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--skip-download", action="store_true",
                    help="assume parquet files already in raw/")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    RAW.mkdir(parents=True, exist_ok=True)
    token = load_env_token()

    if not args.skip_download:
        for split in ("train", "validation", "test"):
            download(f"{BASE}/data/{split}-00000-of-00001.parquet",
                     RAW / f"{split}.parquet", token)

    import pyarrow.parquet as pq

    raw_info = {f"{s}.parquet": {
        "bytes": (RAW / f"{s}.parquet").stat().st_size,
        "sha256": hashlib.sha256((RAW / f"{s}.parquet").read_bytes()).hexdigest(),
    } for s in ("train", "validation", "test")}

    cols = ["prompt", "input_label", "image_path", "violated_categories",
            "language", "task_type", "row_id", "dataset_source", "provenance"]
    splits = {}
    stats = {"rows_total": {}, "topic_rows_excluded": {}, "image_rows": {},
             "text_safety_rows": {}}
    for s in ("train", "validation", "test"):
        t = pq.read_table(RAW / f"{s}.parquet", columns=cols).to_pydict()
        rows = [dict(zip(cols, v)) for v in zip(*(t[c] for c in cols))]
        stats["rows_total"][s] = len(rows)
        keep, img, topic = [], [], 0
        for r in rows:
            if r["task_type"] != "safety" or r["input_label"] not in ("safe", "unsafe"):
                topic += 1
                continue
            r["image_path"] = r["image_path"] or ""
            (img if r["image_path"] else keep).append(r)
        stats["topic_rows_excluded"][s] = topic
        stats["image_rows"][s] = len(img)
        stats["text_safety_rows"][s] = len(keep)
        splits[s] = {"text": keep, "image": img}
        log(f"{s}: {len(rows)} rows -> {len(keep)} text-safety, "
            f"{len(img)} image, {topic} topic/other excluded")

    # ---- leakage + dedupe ---------------------------------------------------
    log("building shingle sets for leakage check")
    pool = splits["train"]["text"] + splits["validation"]["text"] + splits["test"]["text"]
    norm = [normalize(r["prompt"]) for r in pool]
    exact = {}
    exact_dups = 0
    for i, nt in enumerate(norm):
        key = hashlib.sha256(nt.encode()).hexdigest()
        if key in exact:
            exact_dups += 1
        exact.setdefault(key, []).append(i)
    docs = [shingles(r["prompt"]) for r in pool]
    clusters = near_dup_clusters(docs)

    n_train = len(splits["train"]["text"])
    n_val = len(splits["validation"]["text"])
    is_eval = [i >= n_train for i in range(len(pool))]  # val+test are eval-side
    drop_train, train_near_cluster = set(), set()
    for cl in clusters:
        if any(is_eval[i] for i in cl):
            drop_train.update(i for i in cl if i < n_train)
        else:
            train_near_cluster.update(cl)

    # exact-dup groups that straddle train/eval are also leaks
    for idxs in exact.values():
        if len(idxs) > 1 and any(is_eval[i] for i in idxs):
            drop_train.update(i for i in idxs if i < n_train)
    log(f"leakage: {len(drop_train)} train rows collide (exact/near) with eval rows; "
        f"{exact_dups} exact-duplicate rows total; "
        f"{len(train_near_cluster)} train rows in train-internal near-dup clusters (kept)")

    train_pool = [r for i, r in enumerate(splits["train"]["text"])
                  if i not in drop_train]
    # within-train exact dedupe (keep first)
    seen, deduped = set(), []
    for r in train_pool:
        k = hashlib.sha256(normalize(r["prompt"]).encode()).hexdigest()
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    stats["train_exact_dups_dropped"] = len(train_pool) - len(deduped)
    train_pool = deduped
    rng.shuffle(train_pool)

    dev = train_pool[: args.dev_size]
    train_rows = train_pool[args.dev_size:]
    if args.limit:
        train_rows = train_rows[: args.limit]
    log(f"train pool {len(train_pool)} -> dev {len(dev)} + train {len(train_rows)}")

    # ---- images -------------------------------------------------------------
    wiki_train = [r for r in splits["train"]["image"] if r["image_path"].startswith("http")]
    test_img = splits["test"]["image"]
    val_img = splits["validation"]["image"]
    n_wiki = min(len(wiki_train), args.image_cap // 2)
    n_test = min(len(test_img), (args.image_cap - n_wiki) // 2)
    n_val = min(len(val_img), args.image_cap - n_wiki - n_test)
    picked_img = wiki_train[:n_wiki] + test_img[:n_test] + val_img[:n_val]
    got = fetch_wikimedia(picked_img, RAW / "images", args.image_download)
    log(f"image rows picked {len(picked_img)}; wikimedia thumbnails fetched: {got}")

    wiki_ids = {id(r) for r in wiki_train}
    test_img_ids = {id(r) for r in test_img}

    def img_eval_row(r: dict) -> dict:
        src_split = "train" if id(r) in wiki_ids else (
            "test" if id(r) in test_img_ids else "validation")
        row = eval_row(f"{src_split}-img", r)
        p = r["image_path"]
        if r.get("image_local"):
            row["image_path"] = r["image_local"]
        elif p.startswith("http"):
            row["image_path"] = p  # usable URL
        else:
            row["image_path"] = f"images/{src_split}/{p}"  # canonical post-extract path
        row["meta"] = {
            "image_ref": p,
            "image_resolved": bool(r.get("image_local")) or p.startswith("http"),
        }
        return row

    # ---- write --------------------------------------------------------------
    files = {
        "train_sft.jsonl": (sft_row(r) for r in train_rows),
        "train_kway.jsonl": (kway_row(r) for r in train_rows),
        "dev_calib.jsonl": (eval_row("dev", r) for r in dev),
        "validation.jsonl": (eval_row("validation", r) for r in splits["validation"]["text"]),
        "test.jsonl": (eval_row("test", r) for r in splits["test"]["text"]),
        "image_rows.jsonl": (img_eval_row(r) for r in picked_img),
    }
    out_info = {}
    for name, rows in files.items():
        out_info[name] = write_jsonl(DATA_DIR / name, rows)
        log(f"wrote {name}: {out_info[name]['rows']} rows")

    # eval-side exact/near dedupe report (between validation and test)
    ev_base = n_train
    val_test_near = 0
    for cl in clusters:
        has_val = any(ev_base <= i < ev_base + n_val for i in cl)
        has_test = any(i >= ev_base + n_val for i in cl)
        if has_val and has_test:
            val_test_near += 1

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "scripts/prep_nvidia.py",
        "source": {
            "dataset": REPO,
            "url": SOURCE_URL,
            "license": LICENSE,
            "card": SOURCE_URL,
            "raw_files": raw_info,
        },
        "params": vars(args),
        "outputs": out_info,
        "stats": stats,
        "leakage": {
            "method": "sha256 exact on NFKC-lowered text + MinHash(6,2 bands) "
                      "verified Jaccard>=0.8 on word-3gram shingles",
            "train_rows_dropped_for_eval_collision": len(drop_train),
            "exact_duplicate_rows_seen": exact_dups,
            "train_exact_dups_dropped": stats["train_exact_dups_dropped"],
            "train_internal_neardup_rows_kept": len(train_near_cluster),
            "val_test_neardup_clusters_reported_not_dropped": val_test_near,
            "policy": "eval rows are sacred; colliding train rows dropped",
        },
        "taxonomy": "data/TAXONOMY.md (binary safe/unsafe headline; 26 slug categories)",
        "system_prompt": SYSTEM_PROMPT,
        "notes": [
            "image bundle (images.tar.gz, ~3.9GB) intentionally not downloaded: disk cap",
            "bundle image_path values resolve to images/<split>/<ref> after extraction",
            "wikimedia-URL image rows exist only in the train split",
            "topic_following rows (on-topic/off-topic) excluded from safety outputs",
            "violated_categories is the dataset's row-level violation label; it can "
            "reflect the response side on input-safe/response-unsafe rows",
        ],
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("wrote manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
