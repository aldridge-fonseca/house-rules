"""Read managed SFT job status. Polling is dry-run unless --execute is set."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--watch", action="store_true", help="poll until terminal state")
    ap.add_argument("--interval-seconds", type=int, default=60)
    ap.add_argument("--timeout-minutes", type=int, default=120)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    command = ["firectl", "sftj", "get", args.job_id]
    print(" ".join(command))
    if not args.execute:
        print("Dry run; no Fireworks call made.")
        return 0
    if not os.environ.get("FIREWORKS_API_KEY") or not os.environ.get("FIREWORKS_ACCOUNT_ID"):
        print("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID are required with --execute.",
              file=sys.stderr)
        return 2
    if args.interval_seconds < 1 or args.timeout_minutes < 1:
        ap.error("poll interval and timeout must be positive")
    deadline = time.monotonic() + args.timeout_minutes * 60
    terminal = ("SUCCEEDED", "SUCCESS", "COMPLETED", "FAILED", "CANCELLED", "CANCELED")
    while True:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            return result.returncode
        state = result.stdout.upper()
        if any(word in state for word in terminal):
            return 0
        if not args.watch:
            return 0
        if time.monotonic() >= deadline:
            print("Polling deadline reached; inspect the job in the Fireworks console.",
                  file=sys.stderr)
            return 3
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
