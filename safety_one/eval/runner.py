"""Concurrent, resumable evaluation. Receipts are appended after each prediction."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _hash(row: dict[str, Any]) -> str:
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _field(v: Any, key: str, default: Any = None) -> Any:
    return v.get(key, default) if isinstance(v, dict) else getattr(v, key, default)


def _predict(system: Any, row: dict[str, Any]) -> dict[str, Any]:
    # Keep these fields aligned with the shared interface types.
    try:
        from safety_one.types import Message
        msg = Message(id=str(row["id"]), text=row.get("text", ""), image_path=row.get("image_path"),
                      lang=row.get("lang"), meta={})
    except (ImportError, AttributeError):
        from types import SimpleNamespace
        msg = SimpleNamespace(id=str(row["id"]), text=row.get("text", ""), image_path=row.get("image_path"),
                              lang=row.get("lang"), meta={})
    if hasattr(system, "classify"):
        verdict = system.classify(msg)
    elif hasattr(system, "run"):
        verdict = system.run(msg)
    elif callable(system):
        verdict = system(msg)
    else:
        raise TypeError("system must implement classify(Message), run(Message), or be callable")
    return {key: _field(verdict, key, default) for key, default in (
        ("label", ""), ("categories", []), ("probs", {}), ("route", "unknown"),
        ("rule_hits", []), ("model", "unknown"), ("latency_ms", 0.0),
        ("cost_usd", 0.0), ("raw", {}))}


def _retryable(rec: dict[str, Any]) -> bool:
    """Transient failures are re-run on resume; a provider content-filter refusal is a final outcome."""
    err = rec.get("error") or ""
    return bool(err) and "ContentFilter" not in err


def _read_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    cache[(str(rec.get("id")), str(rec.get("input_hash")))] = rec
                except (json.JSONDecodeError, TypeError):
                    continue
    return cache


def run_system(system: Any, rows: list[dict[str, Any]], receipt: str | Path, *,
               system_name: str, system_spec: str = "", concurrency: int = 4) -> list[dict[str, Any]]:
    """Run on identical rows, reusing only matching id + full-row hash receipts."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    path = Path(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = _read_cache(path)
    done: dict[tuple[str, str], dict[str, Any]] = {}
    todo = []
    for row in rows:
        row_hash = _hash(row)
        old = cache.get((str(row["id"]), row_hash))
        if old and old.get("system_spec", "") == system_spec and not _retryable(old):
            done[(str(row["id"]), row_hash)] = old
        else:
            todo.append((row, row_hash))

    def work(item):
        row, row_hash = item
        try:
            pred = _predict(system, row)
            error = None
        except Exception as exc:  # keep failures in the denominator and receipt
            pred = {"label": "", "categories": [], "probs": {}, "route": "error", "rule_hits": [],
                    "model": system_name, "latency_ms": 0.0, "cost_usd": 0.0, "raw": {}}
            error = f"{type(exc).__name__}: {exc}"
        return {"id": str(row["id"]), "input_hash": row_hash, "system": system_name,
                "system_spec": system_spec, "text": row.get("text", ""), "image_path": row.get("image_path"),
                "lang": row.get("lang"), "gold_label": row.get("gold_label"),
                "gold_categories": row.get("gold_categories") or [], "source": row.get("source"),
                "split": row.get("split"), "perturbation": row.get("perturbation") or "none",
                "prediction": pred, "error": error}

    # Main thread owns writes; at most `concurrency` prediction calls are active.
    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {pool.submit(work, item): item for item in todo}
        for future in as_completed(future_map):
            rec = future.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[(rec["id"], rec["input_hash"])] = rec
    return [done[(str(row["id"]), _hash(row))] for row in rows]
