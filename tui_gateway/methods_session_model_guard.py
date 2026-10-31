"""``session.create`` composer overrides and the model×provider coherence gate (#96817).

A composer, script or older client can pin a model the selected provider cannot serve
(``gpt-5.5`` on ``anthropic``); the session used to be minted fine and the FIRST turn died with
the provider's 404, leaving a dead chat. The gate is offline (curated catalogs only) and stays
permissive wherever Hermes cannot know better — see ``models_validate.static_model_provider_conflict``.
"""

from __future__ import annotations

import contextlib


def model_override_conflict(params: dict, build_scope) -> dict | None:
    """The conflict record for the create params' model override, or ``None`` when coherent /
    undecidable. Without an explicit ``provider`` the pair is judged against the provider the
    session would actually build with (profile config, then env) inside ``build_scope`` — the
    handler's ``_profile_build_scope(profile_home)`` context manager."""
    model = str(params.get("model") or "").strip()
    if not model:
        return None
    from hermes_cli.models_validate import static_model_provider_conflict
    from hermes_cli.runtime_provider import resolve_requested_provider

    provider = str(params.get("provider") or "").strip()
    if not provider:
        with build_scope:
            provider = resolve_requested_provider()
    return static_model_provider_conflict(model, provider)


def create_overrides(params: dict) -> tuple:
    """PER-SESSION (model, reasoning, service_tier) overrides from the composer — never a global config
    write. An explicit ``service_tier`` wins over legacy ``fast`` (an unknown word raises ``ValueError``);
    ``fast`` presence: omitted inherits, true pins priority, false pins normal (\"\")."""
    from utils import is_truthy_value

    model = str(params.get("model") or "").strip()
    model_override = {"model": model, "provider": str(params.get("provider") or "").strip() or None} if model else None
    if model_override and model_override["provider"]:
        _adopt_detected_pick(model_override["provider"], params)
    reasoning_override = None
    if effort := str(params.get("reasoning_effort") or "").strip():
        with contextlib.suppress(Exception):
            from hermes_constants import parse_reasoning_effort
            reasoning_override = parse_reasoning_effort(effort)
    service_tier_override = None
    if params.get("service_tier") is not None:
        from agent.fast_mode import parse_exact_service_tier
        service_tier_override = parse_exact_service_tier(params["service_tier"])
    elif "fast" in params:
        service_tier_override = "priority" if is_truthy_value(params.get("fast")) else ""
    return model_override, reasoning_override, service_tier_override


def _adopt_detected_pick(provider: str, params: dict) -> None:
    """A composer pick of a ``Detected:`` local server row (``local-lmstudio``) is pinned on the new
    chat without a ``/model`` switch, so nothing saved the server and the first turn died on an
    unknown provider. Save it into the OWNING profile's ``providers:`` before the session builds,
    the same entry ``switch_model`` writes; any other provider is left untouched."""
    from hermes_cli.models_local_detect import PROVIDER_KEY_PREFIX, adopt_detected_provider

    if not provider.lower().removeprefix("custom:").startswith(PROVIDER_KEY_PREFIX):
        return
    from hermes_cli.config import load_config

    from .server import _profile_build_scope, _profile_home

    with _profile_build_scope(_profile_home(str(params.get("profile") or "").strip() or None)):
        adopt_detected_provider(provider, load_config().get("providers"))
