"""Delete a Fireworks deployment, optionally after a timer. Dry-run by default."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deployment-id", required=True,
                    help="deployment ID, not API key or model ID")
    ap.add_argument("--after-minutes", type=int, default=0)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    if args.after_minutes < 0:
        ap.error("--after-minutes must be non-negative")
    # --ignore-checks: without it the API refuses (FailedPrecondition) to delete a
    # deployment that served requests in the last hour — i.e. every timer teardown.
    command = ["firectl", "deployment", "delete", args.deployment_id, "--ignore-checks"]
    print((f"Scheduled teardown in {args.after_minutes} minutes: " if args.after_minutes
           else "Teardown command: ") + " ".join(command))
    if not args.execute:
        print("Dry run; deployment was not deleted.")
        return 0
    if not os.environ.get("FIREWORKS_API_KEY") or not os.environ.get("FIREWORKS_ACCOUNT_ID"):
        print("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID are required with --execute.",
              file=sys.stderr)
        return 2
    if args.after_minutes:
        time.sleep(args.after_minutes * 60)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
