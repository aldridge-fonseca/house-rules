#!/usr/bin/env python3
"""Make one small live request per configured provider; without keys, dry-run."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety_one.budget import BudgetExceeded, BudgetTracker
from safety_one.types import Message


def configured_providers() -> list[tuple[str, object]]:
    from safety_one.providers.fireworks import FireworksProvider
    from safety_one.providers.frontier import FrontierProvider, OpenAIModerationProvider

    providers: list[tuple[str, object]] = []
    fw_key = bool(os.environ.get("FIREWORKS_API_KEY"))
    serverless = os.environ.get("FIREWORKS_SERVERLESS_MODEL")
    deployment = os.environ.get("FIREWORKS_MODEL")
    if fw_key and serverless:
        providers.append(("fireworks-serverless", FireworksProvider(model=serverless)))
    if fw_key and deployment and "/deployments/" in deployment:
        providers.append(("fireworks-deployment", FireworksProvider(model=deployment)))
    for name, env in (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")):
        if os.environ.get(env):
            providers.append((name, FrontierProvider(name, execute=True)))
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(("openai-moderation", OpenAIModerationProvider(execute=True)))
    return providers


def main() -> int:
    tracker = BudgetTracker()
    configured = configured_providers()
    if not configured:
        print("DRY-RUN: no configured provider keys/models; no requests sent")
        return 0
    message = Message(id="smoke", text="Hello. Reply with a safe moderation label.")
    failed = False
    for name, provider in configured:
        try:
            # Conservative per-smoke call ceilings. Override per provider using
            # SAFETY_ONE_SMOKE_RESERVE_<PROVIDER>_USD if needed.
            key = "SAFETY_ONE_SMOKE_RESERVE_" + name.upper().replace("-", "_") + "_USD"
            default = 0.25 if name.startswith("fireworks") else 0.10
            reserve = float(os.environ.get(key, default))
            token = tracker.reserve(name, reserve, metadata={"purpose": "smoke"})
        except (BudgetExceeded, ValueError) as exc:
            failed = True
            print(f"{name}: FAIL latency_ms=0.0 cost_usd=0.000000 budget={exc}")
            continue
        started = time.perf_counter()
        try:
            verdict = provider.classify(message)
            latency = (time.perf_counter() - started) * 1000
            cost = max(0.0, float(verdict.cost_usd))
            estimated = cost == 0.0 and name != "openai-moderation"
            if estimated:
                # Some provider clients cannot price a call from response usage
                # (notably Fireworks deployments); retain the conservative cap.
                cost = reserve
            tracker.settle(token, cost)
            basis = "reserve-estimate" if estimated else "provider"
            print(f"{name}: OK latency_ms={latency:.1f} cost_usd={cost:.6f} cost_basis={basis}")
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            # A timeout or malformed reply can follow a billable request. Keep
            # the conservative reservation in the ledger rather than assuming $0.
            tracker.settle(token, reserve)
            failed = True
            # Exception text may include request or credential data. Only show
            # the exception type and HTTP status-free generic description.
            print(f"{name}: FAIL latency_ms={latency:.1f} cost_usd_reserved={reserve:.6f} error={type(exc).__name__}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
