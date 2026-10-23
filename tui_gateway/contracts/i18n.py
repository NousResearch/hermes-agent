"""Pluggable languages (``methods_i18n.py``): the language list and the pack/overlay catalog layer a
renderer merges over its own bundled English. Both ride ``profile`` so a multiplexed backend answers
for the profile whose overlay dir and ``display.language`` the client is showing.
"""

from __future__ import annotations

from .base import Params, Result, WireEnum
from .common import ProfileParams
from .registry import method

# ── shapes ────────────────────────────────────────────────────────────────────────────────────


class LocaleSurface(WireEnum):
    core = "core"
    tui = "tui"
    desktop = "desktop"


class LanguageOption(Result):
    """``agent.i18n_languages.language_options`` row. ``source`` is ``bundled``, ``overlay`` or
    ``plugin:<name>`` — the highest layer that supplies the language."""

    id: str
    endonym: str
    rtl: bool
    source: str


class I18nLanguagesResult(Result):
    languages: list[LanguageOption]


class I18nCatalogParams(ProfileParams):
    lang: str
    surface: LocaleSurface = LocaleSurface.core


class I18nCatalogResult(Result):
    """``messages`` is ONLY the pack + user-overlay layer for that surface (flat dotted keys); the
    client merges it over its bundled ``en``/``<lang>``. ``lang`` is the canonical id the request
    resolved to (``pt-BR`` → ``pt-br``; an unknown id resolves to ``en`` with an empty layer)."""

    lang: str
    surface: LocaleSurface
    messages: dict[str, str]


# ── methods ───────────────────────────────────────────────────────────────────────────────────

method("i18n.languages", params=ProfileParams, result=I18nLanguagesResult,
       doc="Every language some layer supplies (bundled ∪ user overlay ∪ plugin packs), en first.")
method("i18n.catalog", params=I18nCatalogParams, result=I18nCatalogResult,
       doc="Pack + overlay messages for one language and surface; the renderer merges them over its bundled catalog.")
