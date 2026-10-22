"""Speed tier for ``delegate_task`` children.

A route's ``service_tier`` applies to that child only and is checked before any child is built: an unknown word,
or a tier the child's route or account does not offer, fails the spawn instead of the child silently running at
another tier.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

_TIER_KEYS = ("service_tier", "speed")


def without_service_tier(overrides: Any) -> dict[str, Any]:
    """Deep copy of *overrides* without a speed-tier key, at the top level or inside ``extra_body``."""
    cleaned = {k: v for k, v in copy.deepcopy(overrides if isinstance(overrides, dict) else {}).items()
               if k not in _TIER_KEYS}
    extra = cleaned.get("extra_body")
    if isinstance(extra, dict):
        extra = {k: v for k, v in extra.items() if k not in _TIER_KEYS}
        if extra:
            cleaned["extra_body"] = extra
        else:
            del cleaned["extra_body"]
    return cleaned


def _offered_tier_overrides(tier: str, model: Any, provider: Any, base_url: Any,
                            access_token: Any = None) -> Optional[dict[str, Any]]:
    """Request overrides for *tier* on this route, or None when the route (or, for ultrafast, the route's own Codex
    account, ``access_token``) does not offer it. The same gates as ``/fast``."""
    from hermes_cli.models import resolve_fast_mode_overrides
    fast = resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier=tier)
    if fast and tier == "ultrafast":
        from agent.codex_service_tiers import ultrafast_offered
        # The Codex backend accepts ultrafast from a plan without it and serves another tier instead.
        if not ultrafast_offered(model, provider=provider, base_url=base_url, access_token=access_token):
            return None
    return fast


def task_service_tier(word: str, creds: dict[str, Any], parent_agent) -> dict[str, Any]:
    """Creds keys for a per-task speed tier on the child's resolved route; a tier the route can't serve raises.
    ``task_tier_overrides`` is re-applied to the built child by :func:`pin_child_service_tier`."""
    from agent.fast_mode import STATIC_TIERS, parse_exact_service_tier
    try:
        tier = parse_exact_service_tier(word)
    except ValueError:
        tier = None
    if tier is None or (tier and tier not in STATIC_TIERS):
        raise ValueError(f"Unknown service_tier '{word}'. Use normal, fast, or ultrafast.")
    # Same bundle rule as _resolve_child_runtime: a pinned provider/endpoint owns the route, else the parent's.
    pinned = bool(creds.get("provider") or creds.get("base_url"))
    provider = creds.get("provider") or getattr(parent_agent, "provider", None)
    model = creds.get("model") or getattr(parent_agent, "model", None)
    base_url = creds.get("base_url") if pinned else getattr(parent_agent, "base_url", None)
    api_key = creds.get("api_key") if pinned else getattr(parent_agent, "api_key", None)
    base = creds.get("request_overrides")
    if base is None and not creds.get("provider"):
        base = getattr(parent_agent, "request_overrides", None)
    fast: dict[str, Any] = {}
    if tier:
        fast = _offered_tier_overrides(tier, model, provider, base_url, api_key) or {}
        if not fast:
            raise ValueError(
                f"service_tier '{word}' is not available for model '{model}' on provider '{provider}'. "
                f"Omit service_tier or choose a model/provider that offers it."
            )
    return {"request_overrides": {**without_service_tier(base), **fast}, "service_tier": tier or None,
            "task_tier_overrides": fast}


def pin_child_service_tier(child, creds: dict[str, Any]) -> None:
    """Re-apply a task's speed tier to the built child: construction can merge a custom provider's ``extra_body``
    (and its tier) back into ``request_overrides``. No-op for tasks that set no tier."""
    fast = creds.get("task_tier_overrides")
    if fast is None:
        return
    child.request_overrides = {**without_service_tier(getattr(child, "request_overrides", None)), **fast}
