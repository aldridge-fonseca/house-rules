"""Upload a capped dataset and launch managed Fireworks LoRA SFT.

This is managed SFT via firectl, not the Training API decision_loop. It is dry-run
unless --execute is supplied. --max-examples is applied before upload.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _selected_dataset(source: Path, limit: int):
    if not limit:
        return source, None
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl",
                                         prefix="safety-one-sft-", dir="/tmp", delete=False)
    count = 0
    with source.open(encoding="utf-8") as inp:
        for line in inp:
            if line.strip():
                handle.write(line)
                count += 1
                if count >= limit:
                    break
    handle.close()
    if not count:
        Path(handle.name).unlink(missing_ok=True)
        raise ValueError("dataset has no JSONL records")
    print(f"Dataset cap: {count} examples.")
    return Path(handle.name), Path(handle.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="accounts/fireworks/models/nemotron-nano-3-30b-a3b")
    ap.add_argument("--dataset-file", required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--output-model", required=True)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=20000)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    if args.epochs < 1 or args.lora_rank < 1 or args.max_examples < 0:
        ap.error("epochs/rank must be positive and max-examples non-negative")
    source = Path(args.dataset_file)
    if not source.is_file():
        ap.error(f"dataset file not found: {source}")
    try:
        upload_path, temp_path = _selected_dataset(source, args.max_examples)
    except ValueError as exc:
        ap.error(str(exc))

    upload = ["firectl", "dataset", "create", args.dataset_id, str(upload_path)]
    launch = ["firectl", "sftj", "create", "--base-model", args.base_model,
              "--dataset", args.dataset_id, "--output-model", args.output_model,
              "--lora-rank", str(args.lora_rank), "--epochs", str(args.epochs)]
    if args.job_id:
        launch += ["--job-id", args.job_id]
    try:
        print("Dry run; no Fireworks calls made:" if not args.execute else "Executing managed SFT workflow:")
        print(" ".join(upload))
        print(" ".join(launch))
        if not args.execute:
            return 0
        if not os.environ.get("FIREWORKS_API_KEY") or not os.environ.get("FIREWORKS_ACCOUNT_ID"):
            print("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID are required with --execute.",
                  file=sys.stderr)
            return 2
        for command in (upload, launch):
            result = subprocess.run(command, check=False)
            if result.returncode:
                return result.returncode
        print("SFT submitted. Save the job ID from firectl output, then run train_poll.py.")
        return 0
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
