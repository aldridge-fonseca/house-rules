"""Persistent, dependency-free USD spend guard for provider calls.

Call ``reserve`` with a conservative upper-bound estimate before a paid request,
then ``settle`` with its measured cost. Reservations count against both caps.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """Raised before a call when its reservation would cross a spend cap."""


# The user's absolute frontier-spend limit: no frontier-group provider may
# exceed this, however the configurable cap is set.
FRONTIER_HARD_CAP_USD = 150.0


def _is_frontier(name: str) -> bool:
    """True for every provider billed against the shared frontier allowance:
    the three native arms plus every frontier-* lane (token-router arms)."""
    n = name.lower().replace("_", "-")
    return n in {"openai", "anthropic", "gemini", "frontier"} or n.startswith("frontier-")


class BudgetTracker:
    def __init__(self, path: str | Path | None = None, *, global_cap: float | None = None,
                 provider_caps: dict[str, float] | None = None):
        root = Path(__file__).resolve().parents[1]
        self.path = Path(path or os.environ.get("SAFETY_ONE_SPEND_FILE", root / "results" / "spend.json"))
        self.global_cap = _cap(global_cap, "SAFETY_ONE_BUDGET_GLOBAL_USD", 200.0)
        self.provider_caps = {k.lower(): float(v) for k, v in (provider_caps or {}).items()}
        self._lock = threading.RLock()

    def _frontier_cap(self) -> float:
        # Configurable via SAFETY_ONE_BUDGET_FRONTIER_USD (default $40), never
        # above FRONTIER_HARD_CAP_USD ($150).
        return min(_cap(None, "SAFETY_ONE_BUDGET_FRONTIER_USD", 40.0),
                   FRONTIER_HARD_CAP_USD)

    def cap_for(self, provider: str) -> float:
        name = provider.lower().replace("_", "-")
        if name in self.provider_caps:
            cap = self.provider_caps[name]
            return min(cap, FRONTIER_HARD_CAP_USD) if _is_frontier(name) else cap
        if name.startswith("fireworks"):
            return _cap(None, "SAFETY_ONE_BUDGET_FIREWORKS_USD", 150.0)
        # Group all frontier providers under the shared frontier allowance.
        if _is_frontier(name):
            return self._frontier_cap()
        env_name = "SAFETY_ONE_BUDGET_" + name.upper().replace("-", "_") + "_USD"
        default = 150.0 if name.startswith("fireworks") else self.global_cap
        return _cap(None, env_name, default)

    def totals(self) -> dict:
        data = self._read()
        providers: dict[str, float] = {}
        reservations = data.get("reservations", {})
        for row in data.get("entries", []):
            providers[row["provider"]] = providers.get(row["provider"], 0.0) + float(row["cost_usd"])
        for row in reservations.values():
            providers[row["provider"]] = providers.get(row["provider"], 0.0) + float(row["amount_usd"])
        return {"global": sum(providers.values()), "providers": providers,
                "reservations": reservations}

    def reserve(self, provider: str, amount_usd: float, *, metadata: dict | None = None) -> str:
        amount = _money(amount_usd)
        if amount <= 0:
            raise ValueError("reservation must be greater than zero")
        provider = provider.lower()
        with self._lock:
            data = self._read()
            totals = self._totals(data)
            current_provider = totals["providers"].get(provider, 0.0)
            if totals["global"] + amount > self.global_cap + 1e-12:
                raise BudgetExceeded(f"global spend cap ${self.global_cap:.2f} would be exceeded")
            cap = self.cap_for(provider)
            if current_provider + amount > cap + 1e-12:
                raise BudgetExceeded(f"{provider} spend cap ${cap:.2f} would be exceeded")
            if _is_frontier(provider):
                frontier_total = sum(v for k, v in totals["providers"].items()
                                     if _is_frontier(k))
                if frontier_total + amount > self._frontier_cap() + 1e-12:
                    raise BudgetExceeded("frontier group spend cap would be exceeded")
            elif provider.startswith("fireworks"):
                fireworks_total = sum(v for k, v in totals["providers"].items() if k.startswith("fireworks"))
                if fireworks_total + amount > cap + 1e-12:
                    raise BudgetExceeded(f"Fireworks group spend cap ${cap:.2f} would be exceeded")
            token = uuid.uuid4().hex
            data.setdefault("reservations", {})[token] = {
                "provider": provider, "amount_usd": amount, "metadata": metadata or {}}
            self._write(data)
            return token

    def settle(self, token: str, actual_cost_usd: float) -> None:
        actual = _money(actual_cost_usd)
        with self._lock:
            data = self._read()
            reservation = data.setdefault("reservations", {}).pop(token, None)
            if reservation is None:
                raise KeyError("unknown or already settled budget reservation")
            # The estimate is a guard; reject a settlement that proves the cap
            # was crossed, but persist actual spend so future calls stay blocked.
            data.setdefault("entries", []).append({
                "provider": reservation["provider"], "cost_usd": actual,
                "metadata": reservation.get("metadata", {})})
            self._write(data)
            totals = self._totals(data)
            provider = reservation["provider"]
            frontier_total = sum(v for k, v in totals["providers"].items()
                                 if _is_frontier(k))
            fireworks_total = sum(v for k, v in totals["providers"].items() if k.startswith("fireworks"))
            over_group = (
                (_is_frontier(provider) and
                 frontier_total > self._frontier_cap() + 1e-12) or
                (provider.startswith("fireworks") and fireworks_total > self.cap_for(provider) + 1e-12)
            )
            if (totals["global"] > self.global_cap + 1e-12 or
                    totals["providers"].get(provider, 0) > self.cap_for(provider) + 1e-12 or over_group):
                raise BudgetExceeded("actual charge exceeded a configured cap; recorded in spend ledger")

    def cancel(self, token: str) -> None:
        with self._lock:
            data = self._read()
            if data.setdefault("reservations", {}).pop(token, None) is not None:
                self._write(data)

    def record(self, provider: str, cost_usd: float, *, metadata: dict | None = None) -> None:
        cost = _money(cost_usd)
        if cost == 0:
            return
        token = self.reserve(provider, cost, metadata=metadata)
        self.settle(token, cost)

    def _read(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "entries": [], "reservations": {}}
        if not isinstance(value, dict) or not isinstance(value.get("entries", []), list):
            raise ValueError(f"invalid spend ledger: {self.path}")
        value.setdefault("reservations", {})
        return value

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.path)

    @staticmethod
    def _totals(data: dict) -> dict:
        providers: dict[str, float] = {}
        for row in data.get("entries", []):
            providers[row["provider"]] = providers.get(row["provider"], 0.0) + float(row["cost_usd"])
        for row in data.get("reservations", {}).values():
            providers[row["provider"]] = providers.get(row["provider"], 0.0) + float(row["amount_usd"])
        return {"global": sum(providers.values()), "providers": providers}


def _money(value: float) -> float:
    amount = float(value)
    if amount < 0 or amount != amount or amount == float("inf"):
        raise ValueError("USD amount must be finite and non-negative")
    return amount


def _cap(value: float | None, env: str, default: float) -> float:
    raw = value if value is not None else os.environ.get(env, default)
    cap = _money(raw)
    if cap <= 0:
        raise ValueError(f"{env} cap must be greater than zero")
    return cap
