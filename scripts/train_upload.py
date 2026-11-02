"""Upload a JSONL chat dataset to Fireworks. Dry-run unless --execute is set."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-file", required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--max-examples", type=int, default=0,
                    help="truncate to this many JSONL records; 0 keeps all")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    source = Path(args.dataset_file)
    if not source.is_file():
        ap.error(f"dataset file not found: {source}")
    if args.max_examples < 0:
        ap.error("--max-examples must be >= 0")

    upload_path = source
    tmp = None
    if args.max_examples:
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl",
                                          prefix="safety-one-", dir="/tmp", delete=False)
        count = 0
        with source.open(encoding="utf-8") as inp:
            for line in inp:
                if not line.strip():
                    continue
                tmp.write(line)
                count += 1
                if count >= args.max_examples:
                    break
        tmp.close()
        upload_path = Path(tmp.name)
        if count == 0:
            upload_path.unlink(missing_ok=True)
            ap.error("dataset has no JSONL records")
        print(f"Selected {count} examples for upload.")

    command = ["firectl", "dataset", "create", args.dataset_id, str(upload_path)]
    print("Dry run; command not executed:" if not args.execute else "Executing dataset upload:")
    print(" ".join(command))
    if not args.execute:
        if tmp:
            upload_path.unlink(missing_ok=True)
        return 0
    if not os.environ.get("FIREWORKS_API_KEY") or not os.environ.get("FIREWORKS_ACCOUNT_ID"):
        print("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID are required with --execute.",
              file=sys.stderr)
        if tmp:
            upload_path.unlink(missing_ok=True)
        return 2
    try:
        result = subprocess.run(command, check=False)
        return result.returncode
    finally:
        if tmp:
            upload_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
