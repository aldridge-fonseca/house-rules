"""Create an on-demand deployment for a trained model and schedule teardown."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="accounts/<account>/models/<tuned-model>")
    ap.add_argument("--auto-stop-minutes", type=int, default=45)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    if args.auto_stop_minutes < 1:
        ap.error("--auto-stop-minutes must be positive")
    command = ["firectl", "deployment", "create", args.model,
               "--deployment-shape", "default", "--min-replica-count", "1"]
    print("Dry run; no Fireworks call made:" if not args.execute else "Creating deployment:")
    print(" ".join(command))
    print(f"Auto-stop target: {args.auto_stop_minutes} minutes after create completes.")
    if not args.execute:
        print("After creation, run deploy_teardown.py --deployment-id <ID> "
              f"--after-minutes {args.auto_stop_minutes} --execute.")
        return 0
    if not os.environ.get("FIREWORKS_API_KEY") or not os.environ.get("FIREWORKS_ACCOUNT_ID"):
        print("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID are required with --execute.",
              file=sys.stderr)
        return 2
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        return result.returncode
    resource = re.search(r"accounts/[^\s]+/deployments/([A-Za-z0-9_-]+)", result.stdout)
    if not resource:
        resource = re.search(r"(?:deployment(?:\s+id)?|id)\s*[:=]\s*([A-Za-z0-9_-]+)",
                             result.stdout, re.IGNORECASE)
    if not resource:
        print("Could not parse deployment ID; schedule teardown manually with "
              "deploy_teardown.py. Copy the deployment ID from the output above.",
              file=sys.stderr)
        return 4
    deployment_id = resource.group(1)
    print(f"Deployment ID: {deployment_id}")
    timer = [sys.executable, str(__file__).replace("deploy_create.py", "deploy_teardown.py"),
             "--deployment-id", deployment_id,
             "--after-minutes", str(args.auto_stop_minutes), "--execute"]
    detached = subprocess.Popen(timer, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True,
                                env=os.environ.copy())
    print(f"Auto-stop timer scheduled (local timer process {detached.pid}).")
    print("Confirm readiness with firectl deployment get and record billable time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
