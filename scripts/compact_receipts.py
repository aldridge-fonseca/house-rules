#!/usr/bin/env python3
"""Copy run_eval receipts without raw provider response bodies ($0).

    .venv/bin/python scripts/compact_receipts.py /tmp/raw/full-cal.jsonl results/frozen-v1/full-cal.jsonl

Keeps every top-level field (id, input_hash, gold, text, ...) and the full
prediction (label, categories, probs, route trail, latency, cost) plus
prediction.raw metadata (usage/token counts, probability_source,
pre_escalation, escalation_usage, decision_threshold, route reasons); drops
only prediction.raw.response (the provider's JSON body incl. logprobs).
The last receipt per (id, input_hash) wins, as in the resumable runner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def compact(src: Path, dst: Path) -> int:
    latest: dict[tuple, dict] = {}
    for line in src.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            raw = (rec.get("prediction") or {}).get("raw")
            if isinstance(raw, dict):
                raw.pop("response", None)
            latest[(rec.get("id"), rec.get("input_hash"))] = rec
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for rec in latest.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(latest)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(compact(Path(sys.argv[1]), Path(sys.argv[2])), "rows")
