"""TUI launch facts over canonical selection and provider routing.

This is TUI application precedence, not an additional provider/model authority.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from application_model_aliases import (
    alias_api_key, configured_provider_ids, model_aliases_from_config,
    provider_reference_context,
)
from application_model_facts import (
    configured_model_matches, provider_facts, static_detection,
)
from models import parse_configured_provider_ref
from models.aliases import MODEL_ALIASES
from models.catalog_static import (
    find_static_provider_model_id, static_provider_default_preference,
    static_provider_model_ids,
)
from models.selection import ExplicitAlias, explicit_provider_hint, select_explicit_model, select_default_model
from providers import (
    configured_custom_identity, get_provider_profile, is_routing_aggregator,
    list_providers, match_configured_provider, normalize_provider,
)
from providers.routing import InvocationRequest, resolve_invocation_route


@dataclass(frozen=True, slots=True)
class StartupModelRoute:
    model: str
    provider: str | None
    base_url: str = ""
    api_key: str = ""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def resolve_startup_seed(
    raw_model: str, current_provider: str, config: Mapping[str, Any], *,
    explicit_provider: str = "",
) -> StartupModelRoute:
    """Resolve a launch seed, retaining TUI precedence and explicit-route constraints."""
    raw, current, explicit = _clean(raw_model), _clean(current_provider), _clean(explicit_provider)
    if not raw:
        return StartupModelRoute(raw, explicit or None)

    aliases = model_aliases_from_config(config)
    direct = aliases.get(raw.lower())
    known, named = provider_reference_context(config)
    qualified = explicit_provider_hint(raw, known_provider_ids=known, named_custom_provider_ids=named)
    # An aggregator's own slash IDs take priority over a provider-looking prefix.
    if not direct and not explicit and is_routing_aggregator(current):
        if find_static_provider_model_id(current, raw):
            return StartupModelRoute(raw, None)

    configured = parse_configured_provider_ref(raw, configured_provider_ids(config))
    request_model, request_provider = raw, explicit
    if configured is not None and not direct and not explicit and not qualified:
        request_model = configured.model
        request_provider = raw.split("/", 1)[0]
    if not direct and not explicit and not qualified and not request_provider:
        profile = get_provider_profile(raw)
        if profile is not None and not is_routing_aggregator(raw) and raw not in {"custom", "auto"}:
            preferred = static_provider_default_preference(profile.name)
            selected = select_default_model(
                profile.name, static_provider_model_ids(profile.name),
                preferred_model=preferred, allow_preferred_without_models=bool(preferred),
                purpose="provider_default",
            )
            if selected.selected is not None:
                ref = selected.selected.ref
                route = resolve_invocation_route(InvocationRequest(provider=ref.provider, model=ref.model))
                return StartupModelRoute(route.model, route.provider)

    detection = static_detection(request_model, current)
    if current != "custom" and not current.startswith("custom:"):
        native = tuple(
            candidate for candidate in detection.static_candidates
            if not is_routing_aggregator(candidate.provider)
        )
        detection = replace(detection, static_candidates=native, allow_first_guess=True)
    else:
        detection = replace(detection, static_candidates=(), allow_first_guess=False)

    fact_ids = [current, request_provider, qualified, *(ref.provider for ref in detection.static_candidates)]
    if raw.lower() in MODEL_ALIASES:
        fact_ids.extend(profile.name for profile in list_providers())
    if direct is not None:
        fact_ids.append(direct.ref.provider)
    selection = select_explicit_model(
        request_model, current,
        explicit_provider=request_provider,
        direct_aliases=(ExplicitAlias(direct.name, direct.ref),) if direct else (),
        provider_facts=provider_facts(tuple(dict.fromkeys(fact_ids)), request_model, config),
        configured_matches=configured_model_matches(request_model, config),
        fallback_providers=tuple(profile.name for profile in list_providers()),
        known_provider_ids=known, named_custom_provider_ids=named,
        detection=detection, hold_current_provider=current == "custom" or current.startswith("custom:"),
        block_provider_fallback=bool(request_provider),
    )
    ref = selection.selected.ref if selection.selected else None
    if ref is None:
        return StartupModelRoute(raw, explicit or None)
    # Ordinary unqualified models already belong to the active configured runtime.
    if not direct and not request_provider and not qualified and ref.provider == normalize_provider(current):
        return StartupModelRoute(ref.model, None)

    alias_resolved = bool(direct and ref.model == direct.ref.model and
                          (selection.matched_alias or explicit))
    use_alias_key = bool(alias_resolved and not explicit and selection.matched_alias)
    alias_url = direct.base_url if alias_resolved else ""
    requested = request_provider or ("custom" if alias_url else ref.provider)
    route = resolve_invocation_route(InvocationRequest(
        provider=requested, model=ref.model, base_url=alias_url,
    ))
    return StartupModelRoute(
        route.model, requested, alias_url,
        alias_api_key(direct) if use_alias_key else "",
    )


def recover_custom_identity(config: Mapping[str, Any], *, base_url: str = "", model: str = "") -> str:
    """Recover named custom identity, including the owned local runtime endpoint."""
    configured_model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    # Older rows can lose both the endpoint and the named entry's identity.
    # The active default is a valid recovery hint only for that same model;
    # never heal an unrelated stale session onto the profile's default host.
    default_model = str(configured_model.get("default") or "").strip()
    config_provider = (
        str(configured_model.get("provider") or "").strip()
        if not model or (default_model and model == default_model)
        else ""
    )
    identity = configured_custom_identity(
        base_url=base_url, model=model, config_provider=config_provider,
        providers=config.get("providers") if isinstance(config.get("providers"), Mapping) else None,
        custom_providers=config.get("custom_providers") if isinstance(config.get("custom_providers"), list) else None,
    )
    if identity:
        return identity
    if base_url:
        try:
            from hermes_cli.local_runtime.endpoint import _state_endpoint
            from providers import normalize_route_base_url
            endpoint = _state_endpoint()
            if endpoint and normalize_route_base_url(endpoint["base_url"]) == normalize_route_base_url(base_url):
                return "llamacpp"
        except Exception:
            pass
    return ""


def is_routable_provider(provider: str, config: Mapping[str, Any]) -> bool:
    name = _clean(provider)
    if not name or name.lower() == "auto":
        return True
    if name.lower() == "custom":
        return False
    configured = match_configured_provider(
        name,
        providers=config.get("providers") if isinstance(config.get("providers"), Mapping) else None,
        custom_providers=config.get("custom_providers") if isinstance(config.get("custom_providers"), list) else None,
    )
    if name.lower().startswith("custom:"):
        # Registry fallback for any custom:<name> returns the generic custom
        # profile; that is NOT proof a deleted named endpoint is still routable.
        profile = get_provider_profile(name)
        return bool(configured or (profile and profile.name.lower() == name.lower()))
    return bool(get_provider_profile(name) or configured)
