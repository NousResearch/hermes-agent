"""Secret-free recovery provenance for Desktop/TUI agent construction boundaries.

A runtime observation is not a model pick. Live core fallbacks keep using the core
restore path; only agents BUILT on a fallback need deferred credential resolution.
All callers run inside the owning session's profile scope.
"""
from __future__ import annotations

import logging
import math
import time
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)
RETRY_SECONDS = 60.0


def route_fields(runtime: dict) -> dict:
    """Persist only route identity, never a client, pool, credential or raw snapshot."""
    route = {k: v for k in ("model", "provider", "base_url", "api_mode")
             if isinstance(v := runtime.get(k), str) and v}
    if route.get("provider") == "custom":
        from hermes_cli.runtime_provider import canonical_custom_identity
        route["provider"] = canonical_custom_identity(
            base_url=route.get("base_url"), model=route.get("model")) or "custom"
    return route


def sanitize_persisted_route(model: str, provider: str, base_url: str, api_mode: str) -> tuple[str, str, str]:
    """Shared row/recovery sanitation; never pair a provider with a foreign canonical URL."""
    from hermes_cli.runtime_provider import (
        canonical_custom_identity, is_foreign_provider_endpoint, is_routable_provider,
    )
    if is_foreign_provider_endpoint(provider, base_url):
        base_url = api_mode = ""
    if provider and not is_routable_provider(provider):
        healed = None
        try:
            healed = canonical_custom_identity(base_url=base_url or None, model=model or None)
        except Exception:
            logger.debug("custom provider identity recovery failed", exc_info=True)
        if healed:
            logger.info("healed stale session provider %r to %r", provider, healed)
            provider, base_url = healed, ""
        else:
            provider = ""
    return provider, base_url, api_mode


def profile_intent(config: dict) -> dict[str, str] | None:
    """Snapshot explicit user intent, not resolved provider defaults or credentials.

    Empty endpoint/wire values mean provider-selected. Keeping them in the snapshot
    makes a later deletion distinguishable from an unchanged provider default.
    """
    model = config.get("model")
    if isinstance(model, str):
        model = {"default": model}
    if not isinstance(model, dict):
        return None
    values = {key: model.get(source) for key, source in (
        ("model", "default"), ("provider", "provider"), ("base_url", "base_url"), ("api_mode", "api_mode"))}
    if any(value is not None and not isinstance(value, str) for value in values.values()):
        return None
    intent = {key: (value or "").strip() for key, value in values.items()}
    if intent["provider"].lower() == "auto":
        intent["provider"] = ""
    return intent if intent["model"] else None


def valid_recovery(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    primary, fallback, deadline = raw.get("primary"), raw.get("fallback"), raw.get("retry_at")
    if not isinstance(primary, dict) or not isinstance(fallback, dict):
        return None
    if not all(isinstance(route.get("model"), str) and route["model"] for route in (primary, fallback)):
        return None
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        return None
    try:
        if not math.isfinite(deadline):
            return None
    except OverflowError:  # JSON integers need not fit in a float.
        return None
    routes = []
    for raw_route in (primary, fallback):
        if any(raw_route.get(k) is not None and not isinstance(raw_route[k], str)
               for k in ("provider", "base_url", "api_mode")):
            return None
        route = {k: v.strip() for k in ("model", "provider", "base_url", "api_mode")
                 if isinstance(v := raw_route.get(k), str) and v.strip()}
        if not route.get("model"):
            return None
        provider, base_url, api_mode = sanitize_persisted_route(
            route["model"], route.get("provider", ""), route.get("base_url", ""), route.get("api_mode", ""))
        # An optional pair with an unresolvable identity must not replace the ordinary row route.
        if route.get("provider") and not provider:
            return None
        routes.append({k: v for k, v in dict(model=route["model"], provider=provider,
                                            base_url=base_url, api_mode=api_mode).items() if v})
    primary, fallback = routes
    if (primary.get("model"), primary.get("provider")) == (fallback.get("model"), fallback.get("provider")):
        return None
    state = {"primary": primary, "fallback": fallback, "retry_at": max(0, deadline)}
    intent = raw.get("profile_intent")
    fields = ("model", "provider", "base_url", "api_mode")
    # Legacy/incomplete intent cannot prove a canonical chat still follows this
    # route. Keep ordinary explicit-session recovery, but never infer history.
    if isinstance(intent, dict) and all(isinstance(intent.get(k), str) for k in fields) and intent["model"].strip():
        state["profile_intent"] = {k: intent[k].strip() for k in fields}
    return state


def recovery_matches_profile(state: dict | None, target: tuple[str, str], config: dict) -> bool:
    """Compare intent without resolving credentials, within the owning profile scope."""
    if state is None:
        return False
    primary = state["primary"]
    model, provider = target
    if primary["model"] != model or (provider and primary.get("provider") != provider):
        return False
    historical = state.get("profile_intent")
    return historical is not None and historical == profile_intent(config)


def recovery_state(agent: Any) -> dict | None:
    """Capture a portable wall-clock deadline, not a process-local monotonic timestamp."""
    from agent.voice_turn_route import session_runtime_view

    # Persistence and rebuild callers may hand us either the agent or its voice
    # view. Read the text route through attributes, never the proxy's __dict__.
    agent = session_runtime_view(agent)
    fallback = route_fields({key: getattr(agent, key, None)
                             for key in ("model", "provider", "base_url", "api_mode")})
    pending = valid_recovery(getattr(agent, "_tui_fallback_recovery", None))
    if pending:
        return {**pending, "fallback": fallback}
    voice_state = getattr(agent, "_voice_route_state", None) or {}
    if voice_state.get("_fallback_activated", getattr(agent, "_fallback_activated", False)) is not True:
        return None
    primary = getattr(agent, "_primary_runtime", None)
    if not isinstance(primary, dict):
        return None
    return valid_recovery({
        "primary": primary, "fallback": fallback,
        "profile_intent": getattr(agent, "_tui_profile_intent", None),
        "retry_at": time.time() + max(0, getattr(agent, "_rate_limited_until", 0) - time.monotonic()),
    })


def _pool_reset_blocks(primary: dict, agent: Any) -> bool:
    """Reuse the core's profile/model-aware, fail-open reset gate before resolving auth."""
    from agent.agent_runtime_helpers import (
        _primary_reset_gate_blocks, credential_pool_matches_provider, resolve_runtime_pool_key,
    )
    from agent.credential_pool import load_pool

    provider, base_url = primary.get("provider", ""), primary.get("base_url", "")
    matches = lambda pool: pool is not None and credential_pool_matches_provider(pool, provider, base_url=base_url)

    def load():
        key = resolve_runtime_pool_key(provider, base_url)
        pool = load_pool(key) if key else None
        return pool if matches(pool) else None

    return _primary_reset_gate_blocks(agent, primary, provider, base_url, matches, load)[0]


def build_override(override: Any) -> tuple[Any, dict | None]:
    state = valid_recovery(override.get("_fallback_recovery")) if isinstance(override, dict) else None
    if state is None:
        return override, None
    # No agent exists yet. Supply only the fallback identity needed for the core
    # gate's log; credentials still load from the caller's bound profile scope.
    fallback = SimpleNamespace(provider=state["fallback"].get("provider", ""), model=state["fallback"]["model"])
    if state["retry_at"] > time.time() or _pool_reset_blocks(state["primary"], fallback):
        return state["fallback"], state
    return state["primary"], None


def recover_pending_primary(agent: Any) -> bool:
    """Retry a construction-time fallback at a turn boundary, never probe the model.

    Runtime resolution may refresh credentials; AuthError gets a bounded retry delay.
    Existing persisted pool reset and entitlement gates remain authoritative. A live
    request-time fallback (with a real primary snapshot) is left to the core entirely.
    """
    state = valid_recovery(getattr(agent, "_tui_fallback_recovery", None))
    if state is None or state["retry_at"] > time.time():
        return False
    from agent.fallback_cooldown import _is_entitlement_rejected
    from tui_gateway import server

    primary = state["primary"]
    provider, model = primary.get("provider", ""), primary["model"]
    if _is_entitlement_rejected(agent, provider, model):
        return False
    if _pool_reset_blocks(primary, agent):
        return False
    # Throttle failures as well as repeated AuthErrors; do not mutate the prompt/client
    # before we have a usable destination. This marker is never a session model override.
    agent._tui_fallback_recovery = {**state, "retry_at": time.time() + RETRY_SECONDS}
    try:
        selected, runtime = server._resolve_agent_model_runtime(primary, provider or None)
        if runtime.get("_fallback_notice"):
            return False  # auth still unavailable; keep the existing fallback client
        chain = list(getattr(agent, "_fallback_chain", []) or [])
        agent.switch_model(
            new_model=selected, new_provider=runtime.get("provider"),
            api_key=runtime.get("api_key", ""), base_url=runtime.get("base_url", ""),
            api_mode=runtime.get("api_mode", ""), capabilities=runtime.get("capabilities"))
    except Exception:
        logger.warning("Deferred primary runtime recovery failed; retaining fallback", exc_info=True)
        return False
    # Recovery is NOT a deliberate provider rejection: switch_model prunes the old
    # provider from the chain for manual picks, but it must remain eligible here.
    agent._fallback_chain = chain
    agent._fallback_model = chain[0] if chain else None
    agent._tui_fallback_recovery = None
    agent._pending_fallback_notice = None
    # switch_model establishes the recovered route as the real primary snapshot.
    agent._provider_fallback_active = False
    agent._provider_fallback_route = None
    try:
        agent._emit_diagnostic_status(
            f"Primary model restored: {agent.model} via {agent.provider}; temporary fallback is no longer active.")
    except Exception:
        logger.debug("Primary recovery notice delivery failed", exc_info=True)
    return True
