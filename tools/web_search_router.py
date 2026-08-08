#!/usr/bin/env python3
"""
Search Provider Router — Layer B, web_search side (R1 §4 / §10).

Deterministic single-provider selection for web_search_tool. Runs only when
``web.router.enabled`` is true. When the flag is false, web_search_tool never
imports this module and the legacy backend resolution path is preserved
exactly.

B1 boundaries (R1 §15.2 / §10):
  - ONE provider per request, selected at decision-construction time;
  - a substitute provider is chosen deterministically ONLY when the
    intent-preferred provider is unavailable (availability is filtered, not
    retried);
  - NO runtime fallback execution, NO verification, NO retry, NO Browser
    invocation, NO query-content telemetry.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from tools.web_router import (
    INTENT_TO_PROVIDER,
    PROVIDER_PREFERENCES,
    SearchIntent,
    SearchRouterDecision,
    classify_search_intent,
    normalize_intent_hint,
    provider_runtime_ready,
    select_substitute_provider,
)


def select_search_provider(
    query: str,
    intent_hint: Optional[str] = None,
    provider_override: Optional[str] = None,
    registry_getter: Optional[Callable[[str], Any]] = None,
    env_has: Optional[Callable[[str], bool]] = None,
    enabled_names: Optional[list] = None,
) -> SearchRouterDecision:
    """Construct ONE SearchRouterDecision for a web_search call.

    Selection order (R1 §10):
      1. explicit provider_override (honored only if registered, enabled,
         credential-present and search-capable);
      2. intent classification (caller hint normalized, else local
         classification of the query — the classifier never depends on the
         Agent emitting a hint);
      3. deterministic substitute when the preferred provider is unavailable
         (decision-construction only — never executed as a fallback).

    Never executes a search. Pure decision construction.
    """
    decision = SearchRouterDecision()
    decision.provider_override = provider_override or None

    # 1) Explicit override — validated before intent selection.
    if provider_override:
        override_name = str(provider_override).strip().lower()
        if override_name not in PROVIDER_PREFERENCES:
            decision.selection_reason = "override_rejected_not_in_registry"
        elif not _provider_usable(override_name, registry_getter, env_has, enabled_names):
            decision.selection_reason = "override_rejected_unavailable"
        else:
            decision.selected_provider = override_name
            decision.selection_reason = "explicit_override"
            return decision

    # 2) Intent classification (hint normalized, else local).
    intent = normalize_intent_hint(intent_hint) or classify_search_intent(query)
    decision.selection_reason = (
        f"{decision.selection_reason};" if decision.selection_reason else ""
    ) + f"intent={intent.value}"

    preferred = INTENT_TO_PROVIDER.get(intent)
    if preferred and _provider_usable(preferred, registry_getter, env_has, enabled_names):
        decision.selected_provider = preferred
        decision.fallback_provider_advisory = select_substitute_provider(
            "search",
            exclude=preferred,
            registry_getter=registry_getter,
            env_has=env_has,
            enabled_names=enabled_names,
        )
        return decision

    # 3) Preferred unavailable -> ONE deterministic substitute (advisory in B1).
    substitute = select_substitute_provider(
        "search",
        exclude=preferred,
        registry_getter=registry_getter,
        env_has=env_has,
        enabled_names=enabled_names,
    )
    if substitute:
        decision.selected_provider = substitute
        decision.selection_reason = (
            f"{decision.selection_reason};" if decision.selection_reason else ""
        ) + f"preferred_unavailable_substitute={substitute}"
        decision.fallback_provider_advisory = preferred
        return decision

    # Nothing usable in this request — leave selected_provider None so the
    # caller's legacy "no provider" error path renders (B1 preserves output).
    decision.selection_reason = "no_provider_available"
    return decision


def _provider_usable(
    name: str,
    registry_getter: Optional[Callable[[str], Any]],
    env_has: Optional[Callable[[str], bool]],
    enabled_names: Optional[list],
) -> bool:
    """Registered + enabled + credential-present + dependency-importable +
    search-capable (R1 §10, post-B1 readiness correction §5)."""
    if enabled_names is not None and name not in set(enabled_names):
        return False
    return provider_runtime_ready(name, "search", registry_getter, env_has)
