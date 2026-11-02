#!/usr/bin/env python3
"""List Fireworks deployments; dry-run cleanup of old or over-budget instances.

REST paths follow Fireworks docs:
https://docs.fireworks.ai/api-reference/list-deployments
https://docs.fireworks.ai/api-reference/delete-deployment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safety_one.budget import BudgetTracker

API_ROOT = "https://api.fireworks.ai/v1"
DOC_LIST = "https://docs.fireworks.ai/api-reference/list-deployments"
DOC_DELETE = "https://docs.fireworks.ai/api-reference/delete-deployment"


def _request(url: str, key: str, *, method: str = "GET", opener=None) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"}, method=method)
    try:
        with (opener or urllib.request.urlopen)(request, timeout=30) as response:
            body = response.read()
            return json.loads(body.decode("utf-8")) if body else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Fireworks API returned HTTP {exc.code}") from None


def list_deployments(account_id: str, key: str, *, opener=None) -> list[dict]:
    url = f"{API_ROOT}/accounts/{urllib.parse.quote(account_id, safe='')}/deployments"
    result: list[dict] = []
    while url:
        page = _request(url, key, opener=opener)
        items = page.get("deployments", page.get("items", []))
        if not isinstance(items, list):
            raise ValueError("Fireworks list response has no deployments array")
        result.extend(item for item in items if isinstance(item, dict))
        token = page.get("nextPageToken") or page.get("next_page_token")
        url = (f"{API_ROOT}/accounts/{urllib.parse.quote(account_id, safe='')}/deployments?"
               + urllib.parse.urlencode({"pageToken": token})) if token else ""
    return result


def _created_epoch(deployment: dict) -> float | None:
    value = next((deployment.get(k) for k in
                  ("createTime", "create_time", "createdAt", "created_at", "timeCreated")
                  if deployment.get(k)), None)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return None


def _deployment_id(deployment: dict) -> str:
    raw = deployment.get("name") or deployment.get("id") or deployment.get("deploymentId") or ""
    return str(raw).rstrip("/").rsplit("/", 1)[-1]


def _fireworks_total(tracker: BudgetTracker) -> float:
    providers = tracker.totals()["providers"]
    return sum(value for provider, value in providers.items()
               if provider.startswith("fireworks"))


def select_for_cleanup(deployments: list[dict], *, now: float, older_than_minutes: int,
                       over_budget: bool) -> list[tuple[dict, str]]:
    selected = []
    cutoff = now - older_than_minutes * 60
    for deployment in deployments:
        created = _created_epoch(deployment)
        if created is not None and created <= cutoff:
            selected.append((deployment, f"older than {older_than_minutes} minutes"))
        elif over_budget:
            selected.append((deployment, "Fireworks tracked spend is over its cap"))
    return selected


def main(argv=None, *, opener=None, now=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-than-minutes", type=int, default=30,
                        help="delete deployments at least this old (default: 30)")
    parser.add_argument("--execute", action="store_true", help="send DELETE requests; default is dry-run")
    args = parser.parse_args(argv)
    if args.older_than_minutes < 0:
        parser.error("--older-than-minutes must be non-negative")
    key, account = os.environ.get("FIREWORKS_API_KEY"), os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not key or not account:
        print("FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID are required", file=sys.stderr)
        return 2
    try:
        deployments = list_deployments(account, key, opener=opener)
        tracker = BudgetTracker()
        over = _fireworks_total(tracker) > tracker.cap_for("fireworks")
        selected = select_for_cleanup(deployments, now=now if now is not None else time.time(),
                                     older_than_minutes=args.older_than_minutes, over_budget=over)
        print(f"Found {len(deployments)} deployment(s); {len(selected)} cleanup candidate(s).")
        for deployment, reason in selected:
            name = deployment.get("name") or _deployment_id(deployment) or "(unknown id)"
            print(f"{'DELETE' if args.execute else 'WOULD DELETE'} {name}: {reason}")
            if args.execute:
                dep_id = _deployment_id(deployment)
                if not dep_id:
                    print("Skipping deployment without an ID", file=sys.stderr)
                    continue
                url = (f"{API_ROOT}/accounts/{urllib.parse.quote(account, safe='')}/deployments/"
                       f"{urllib.parse.quote(dep_id, safe='')}")
                _request(url, key, method="DELETE", opener=opener)
        if not args.execute:
            print(f"Dry run; no deployments deleted. See {DOC_DELETE}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        # Avoid displaying request URLs, headers, response bodies, or secrets.
        print(f"Watchdog failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
