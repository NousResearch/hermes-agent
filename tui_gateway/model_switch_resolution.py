"""TUI application model resolution over shared canonical model and provider domains."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from application_model_aliases import alias_api_key, model_aliases_from_config, provider_reference_context
from application_model_facts import configured_model_matches, provider_facts, static_detection
from application_model_switch_enrichment import enrich_model_switch
from models.selection import ExplicitAlias, select_default_model, select_explicit_model
from models.catalog_static import static_provider_default_preference, static_provider_model_ids
from providers import get_provider_profile, match_configured_provider, normalize_provider
from providers.routing import InvocationRequest, resolve_invocation_route


@dataclass(slots=True)
class TuiModelSwitchResult:
    success: bool
    new_model: str = ""
    target_provider: str = ""
    api_key: Any = ""
    base_url: str = ""
    api_mode: str = ""
    runtime_kind: str = ""
    provider_changed: bool = False
    request_overrides: dict = field(default_factory=dict)
    warning_message: str = ""
    provider_label: str = ""
    resolved_via_alias: str = ""
    capabilities: Any = None
    runtime_capabilities: dict = field(default_factory=dict)
    model_info: Any = None
    error_message: str = ""
    is_global: bool = False


def _clean(value) -> str:
    return str(value or "").strip()


def _provider_default(provider: str, config: Mapping) -> str:
    configured = match_configured_provider(
        provider, providers=config.get("providers"),
        custom_providers=config.get("custom_providers"),
    )
    if configured is not None and configured.model:
        return configured.model
    profile = get_provider_profile(provider)
    if profile is None:
        raise ValueError(f"Unknown provider '{provider}'.")
    preferred = static_provider_default_preference(profile.name)
    choices = static_provider_model_ids(profile.name)
    selection = select_default_model(
        profile.name, choices, preferred_model=preferred,
        allow_preferred_without_models=bool(preferred), purpose="provider_default",
    )
    if selection.selected is None:
        raise ValueError(f"No model detected for provider '{provider}'. Specify the model explicitly.")
    return selection.selected.ref.model


def resolve_tui_model_switch(
    *, raw_input: str, current_provider: str, current_model: str,
    current_base_url: str = "", current_api_key: Any = "",
    explicit_provider: str = "", is_global: bool = False,
    user_providers: dict | None = None, custom_providers: list | None = None,
    config: Mapping | None = None,
) -> TuiModelSwitchResult:
    """Interpret, acquire and validate a TUI switch without a CLI model coordinator.

    Existing Phase 6 acquisition is the only CLI runtime dependency; its returned
    model/route fields cannot override canonical selection or invocation semantics.
    """
    cfg = dict(config or {})
    if user_providers is not None:
        cfg["providers"] = user_providers
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers
    raw, current, explicit = _clean(raw_input), _clean(current_provider), _clean(explicit_provider)
    if not raw and explicit:
        raw = _provider_default(explicit, cfg)
    if not raw:
        raise ValueError("model value required")

    aliases = model_aliases_from_config(cfg)
    direct = aliases.get(raw.lower())
    known, named = provider_reference_context(cfg)
    matched = configured_model_matches(raw, cfg)
    fact_ids = tuple(dict.fromkeys((
        current, explicit, *(ref.provider for ref in matched),
        *(alias.ref.provider for alias in aliases.values()),
    )))
    selection = select_explicit_model(
        raw, current, explicit_provider=explicit,
        provider_facts=provider_facts(fact_ids, raw, cfg),
        direct_aliases=tuple(ExplicitAlias(alias.name, alias.ref) for alias in aliases.values()),
        configured_matches=matched, known_provider_ids=known,
        named_custom_provider_ids=named, detection=static_detection(raw, current),
        hold_current_provider=(normalize_provider(current) == "custom"
                               or current.startswith("custom:")),
    )
    if selection.selected is None:
        raise ValueError(f"Could not resolve model '{raw}'.")
    ref = selection.selected.ref
    provider = ref.provider
    alias_url = ""
    alias_key = ""
    alias_selected = bool(direct is not None and (
        selection.matched_alias or (explicit and ref.model == direct.ref.model)
    ))
    if alias_selected:
        alias_url = direct.base_url
        alias_key = alias_api_key(direct) if not explicit else ""
        if not explicit and alias_url:
            provider = "custom"

    configured = match_configured_provider(
        provider, providers=cfg.get("providers"), custom_providers=cfg.get("custom_providers"),
    )
    route_key = provider
    if configured is not None:
        route_key = configured.provider_key or configured.identity
        alias_url = alias_url or configured.base_url

    reuse_live = normalize_provider(provider) == normalize_provider(current) and not alias_url and not explicit
    if reuse_live:
        runtime = {"provider": provider, "base_url": current_base_url,
                   "api_key": current_api_key}
    else:
        if configured is None and get_provider_profile(route_key) is None:
            raise ValueError(f"Unknown provider '{route_key}'.")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        # Acquisition accepts the user's explicit spelling; canonical selection
        # independently determines the normalized model sent to the live agent.
        acquisition_model = (
            raw if explicit and direct is None and not selection.matched_alias else ref.model
        )
        runtime = resolve_runtime_provider(
            requested=route_key, target_model=acquisition_model,
            explicit_base_url=alias_url or None, explicit_api_key=alias_key or None,
        )
    effective_provider = (configured.identity if configured is not None else
                          route_key if alias_url and route_key.startswith("custom:") else
                          provider)
    base_url = _clean(runtime.get("base_url")) or alias_url or (
        current_base_url if reuse_live else ""
    )
    route = resolve_invocation_route(InvocationRequest(
        provider=effective_provider, requested_provider=route_key, model=ref.model,
        base_url=base_url, configured_api_mode=_clean(runtime.get("api_mode")),
        configured_provider=effective_provider,
        openai_runtime="codex_app_server" if runtime.get("runtime_kind") == "app_server" else "",
    ))
    result = TuiModelSwitchResult(
        success=True, new_model=route.model, target_provider=route.provider,
        api_key=runtime.get("api_key") or "", base_url=route.base_url,
        api_mode=route.api_mode, runtime_kind=route.runtime_kind,
        provider_changed=normalize_provider(route.provider) != normalize_provider(current),
        resolved_via_alias=selection.matched_alias, is_global=is_global,
    )
    return enrich_model_switch(result, cfg)
