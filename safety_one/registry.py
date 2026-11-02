"""Named-system registry — one config shared by the eval, demo, and CLI.

    from safety_one.registry import build_system, list_systems
    system = build_system("full")              # auto: live if keys, else mock
    verdict = system.classify(Message(id="1", text="..."))

`config/systems.yaml` names every eval/demo arm — the ablation ladder
(specialist-only, +rules, +router, +escalation, full) and the single-provider
baselines (frontier-gpt/-claude/-gemini, openai-moderation, open-guard).
Pipeline systems deep-merge `overrides` into config/router.yaml; provider
systems map `backend` to a class in safety_one/providers/.

Modes: "mock" always uses deterministic MockProviders; "live" requires the
entry's env_key and raises RuntimeError if it is missing; "auto" (default)
goes live only when the key is present, else falls back to mocks — so the
eval and demo run unchanged offline and switch on when keys arrive.

run_eval.py compatibility: `--system name=safety_one.registry:<name>` works
because unknown module attributes resolve to a zero-arg factory via
__getattr__ below (e.g. `safety_one.registry:full` or `...:+rules`).
"""
from __future__ import annotations

import copy
import hashlib
import os
from typing import Any

import yaml

from .pipeline import SafetyPipeline, load_config
from .providers.mock import MockProvider, default_mock_providers
from .types import Message, Verdict  # noqa: F401  (re-exported for callers)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEMS_PATH = os.path.join(_ROOT, "config", "systems.yaml")
ENV_PATH = os.path.join(_ROOT, ".env")

MODES = ("auto", "mock", "live")
_ROUTES = ("specialist", "vision", "multilingual", "escalation")
_MOCK_KEYS = (
    "base_latency_ms", "per_char_ms", "cost_per_call_usd",
    "cost_per_1k_chars_usd", "image_cost_usd", "sharpness",
    "jitter_ms", "jitter_prob",
)


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

def _load_env_file() -> None:
    """Populate os.environ defaults from `.env` at repo root (gitignored).

    Never overrides a variable that is already set, and never prints values.
    """
    if os.environ.get("SAFETY_ONE_NO_DOTENV"):
        return
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_systems(path: str | None = None) -> dict:
    """Parse systems.yaml -> {"pipeline_roles": {...}, "systems": {...}}."""
    with open(path or SYSTEMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_systems(path: str | None = None) -> list[dict]:
    """One info dict per named system (name, kind, description, env_key...)."""
    _load_env_file()
    out = []
    for name, spec in (load_systems(path).get("systems") or {}).items():
        env_key = spec.get("env_key")
        live = bool(spec.get("live_supported", True))
        if spec.get("kind") == "pipeline":
            env_key = "FIREWORKS_API_KEY"
        out.append({
            "name": str(name),
            "kind": spec.get("kind", "provider"),
            "description": spec.get("description", ""),
            "env_key": env_key,
            "live_supported": live,
            "key_present": bool(env_key and os.environ.get(env_key)),
        })
    return out


def system_names(path: str | None = None) -> list[str]:
    return [s["name"] for s in list_systems(path)]


def _spec(name: str, path: str | None = None) -> dict:
    spec = (load_systems(path).get("systems") or {}).get(name)
    if spec is None:
        raise KeyError(
            f"unknown system {name!r}; available: {', '.join(system_names(path))}")
    return spec


def _merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of base (dicts only; lists replace)."""
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _name_seed(name: str) -> int:
    """Stable per-system mock seed (sha256; unlike hash(), not salted)."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


# --------------------------------------------------------------------------
# Mock fallback providers
# --------------------------------------------------------------------------

def _frontier_model(provider_key: str) -> str | None:
    try:
        from .providers.frontier import PRICES
        return (PRICES.get(provider_key) or {}).get("model")
    except Exception:
        return None


def _mock_model_name(name: str, spec: dict) -> str:
    if spec.get("mock", {}).get("model"):
        return spec["mock"]["model"]
    real = spec.get("model")
    if not real and spec.get("backend") == "frontier":
        real = _frontier_model(spec.get("provider", ""))
    if not real and spec.get("backend") == "frontier_router":
        real = os.environ.get(spec.get("model_env", "")) or None
    if not real and spec.get("backend") == "openai_moderation":
        try:
            from .providers.frontier import OpenAIModerationProvider
            real = OpenAIModerationProvider.model
        except Exception:
            pass
    return f"{real}-sim" if real else f"mock-{name}"


def _mock_provider(name: str, spec: dict, seed: int | None) -> MockProvider:
    mcfg = dict(spec.get("mock") or {})
    model = _mock_model_name(name, spec)
    kwargs = {k: mcfg[k] for k in _MOCK_KEYS if k in mcfg}
    return MockProvider(
        name=name, model=model,
        seed=_name_seed(name) if seed is None else seed, **kwargs)


# --------------------------------------------------------------------------
# Live providers (constructed only when keys are present / live requested)
# --------------------------------------------------------------------------

def _live_provider(spec: dict):
    backend = spec.get("backend")
    if backend == "frontier":
        from .providers.frontier import FrontierProvider
        return FrontierProvider(spec["provider"], execute=True)
    if backend == "openai_moderation":
        from .providers.frontier import OpenAIModerationProvider
        return OpenAIModerationProvider(execute=True)
    if backend == "frontier_router":
        from .providers.frontier import RouterFrontierProvider
        return RouterFrontierProvider(
            model_env=spec.get("model_env", "FRONTIER_MODEL"), execute=True)
    if backend == "open_guard":
        from .providers.open_guards import OpenGuardProvider
        return OpenGuardProvider(spec.get("guard", "llama_guard_3_8b"),
                                 execute=True)
    if backend in _FIREWORKS_BACKENDS:
        return _fireworks_provider(backend, spec.get("model"))
    raise ValueError(f"unknown provider backend {backend!r}")


_FIREWORKS_BACKENDS = ("fireworks_chat", "fireworks_vision",
                       "fireworks_systemone")


def _fireworks_provider(backend: str, model: str | None):
    from .providers import fireworks as fw
    cls = {
        "fireworks_chat": fw.FireworksProvider,
        "fireworks_vision": fw.FireworksVisionProvider,
        "fireworks_systemone": fw.SystemOneProvider,
    }[backend]
    return cls(model=model or None)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def _annotate(system: Any, name: str, mode: str, **extra) -> Any:
    system.system_name = name
    system.mode = mode
    for k, v in extra.items():
        setattr(system, k, v)
    return system


def _build_provider_system(name: str, spec: dict, mode: str,
                           seed: int | None) -> Any:
    env_key = spec.get("env_key")
    live_ok = bool(spec.get("live_supported", True))
    key = os.environ.get(env_key) if env_key else None

    if mode == "live":
        if not live_ok:
            raise RuntimeError(
                f"{name}: live mode unsupported — {spec.get('description', '')}")
        if env_key and not key:
            raise RuntimeError(f"{name}: live mode requires {env_key}")

    if live_ok and (mode == "live" or (mode == "auto" and key)):
        try:
            return _annotate(_live_provider(spec), name, "live")
        except Exception:
            if mode == "live":
                raise
    return _annotate(_mock_provider(name, spec, seed), name, "mock")


def _build_pipeline_system(name: str, spec: dict, mode: str,
                           seed: int | None, cfg: dict | None,
                           systems_cfg: dict) -> SafetyPipeline:
    base = cfg if cfg is not None else load_config()
    merged = _merge(base, spec.get("overrides") or {})

    key = os.environ.get("FIREWORKS_API_KEY")
    if mode == "live" and not key:
        raise RuntimeError("pipeline systems: live mode requires FIREWORKS_API_KEY")
    want_live = mode == "live" or (mode == "auto" and bool(key))

    roles = (systems_cfg.get("pipeline_roles") or {})
    mocks = default_mock_providers(merged, seed=seed)
    providers: dict = {}
    role_modes: dict[str, str] = {}
    for role in _ROUTES:
        recipe = dict(roles.get(role) or {"backend": "fireworks_chat"})
        backend = recipe.get("backend", "fireworks_chat")
        if want_live and backend != "mock":
            # Precedence: recipe model > recipe model_env (e.g. FIREWORKS_MODEL,
            # e.g. FIREWORKS_MODEL for a deployment) > router.yaml providers.<role>.model.
            model = recipe.get("model") or (
                os.environ.get(recipe["model_env"]) if recipe.get("model_env") else None) or (
                (merged.get("providers") or {}).get(role) or {}).get("model")
            # systemone carries its own default model; chat/vision need one.
            if not model and backend != "fireworks_systemone":
                if mode == "live":
                    raise RuntimeError(
                        f"pipeline role {role!r}: no model configured "
                        f"(set providers.{role}.model in config/router.yaml)")
            else:
                try:
                    providers[role] = _fireworks_provider(backend, model)
                    role_modes[role] = "live"
                    continue
                except Exception:
                    if mode == "live":
                        raise
        providers[role] = mocks[role]
        role_modes[role] = "mock"

    used = set(role_modes.values())
    resolved = "live" if used == {"live"} else "mock" if used == {"mock"} else "mixed"
    return _annotate(SafetyPipeline(providers, merged), name, resolved,
                     role_modes=role_modes)


def build_system(name: str, *, mode: str = "auto", seed: int | None = None,
                 cfg: dict | None = None,
                 systems_path: str | None = None) -> Any:
    """Build the named system -> Provider | SafetyPipeline.

    mode: "auto" (live when the entry's env_key is set, else mock),
          "mock" (always deterministic mocks), "live" (require keys).
    seed: override mock seed (also forwarded to default_mock_providers).
    cfg:  pre-parsed router.yaml dict; defaults to load_config().
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _load_env_file()
    systems_cfg = load_systems(systems_path)
    spec = (systems_cfg.get("systems") or {}).get(name)
    if spec is None:
        raise KeyError(
            f"unknown system {name!r}; available: {', '.join(system_names(systems_path))}")
    if spec.get("kind") == "pipeline":
        return _build_pipeline_system(name, spec, mode, seed, cfg, systems_cfg)
    return _build_provider_system(name, spec, mode, seed)


def factory(name: str, **kwargs):
    """Zero-arg builder, e.g. for run_eval `--system x=module:factory`."""
    return lambda: build_system(name, **kwargs)


def __getattr__(attr: str):
    """Expose each system as a zero-arg module factory, so
    `--system full=safety_one.registry:full` works in scripts/run_eval.py."""
    try:
        _spec(attr)
    except KeyError:
        raise AttributeError(attr) from None
    return factory(attr)
