"""``session.create`` model×provider coherence gate (#96817).

A composer, script or older client can pin a model the selected provider cannot serve
(``gpt-5.5`` on ``anthropic``); the session used to be minted fine and the FIRST turn died with
the provider's 404, leaving a dead chat. The gate is offline (curated catalogs only) and stays
permissive wherever Hermes cannot know better — see ``models_validate.static_model_provider_conflict``.
"""

from __future__ import annotations


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
