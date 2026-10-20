"""Pluggable-language JSON-RPC handlers: ``i18n.languages`` and ``i18n.catalog``.

Both are profile-scoped: the user overlay dir (``<home>/locales``) and ``display.language`` belong
to the profile the client is looking at. Plugin packs are process-wide (one PluginManager per home
registers them under the same registry), so a pack installed in profile A is visible to every
profile — a language switcher lists it everywhere, exactly like a bundled locale.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

_I18N_ERR = 5400


@method("i18n.languages")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from agent.i18n import language_options
    try:
        return _ok(rid, {"languages": language_options()})
    except Exception as e:
        return _err(rid, _I18N_ERR, str(e))


@method("i18n.catalog")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from agent.i18n import resolve_language_id, surface_catalog
    from agent.i18n_layers import CORE_SURFACE, SURFACES
    surface = str(params.get("surface") or CORE_SURFACE)
    if surface not in SURFACES:
        return _err(rid, 4002, f"surface must be one of {', '.join(SURFACES)}")
    lang = resolve_language_id(params.get("lang", "")) or "en"
    try:
        return _ok(rid, {"lang": lang, "surface": surface, "messages": surface_catalog(lang, surface)})
    except Exception as e:
        return _err(rid, _I18N_ERR, str(e))


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
