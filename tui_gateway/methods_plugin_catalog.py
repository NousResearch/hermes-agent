"""``plugins.catalog.lookup``: one curated-catalog entry, resolved by the backend.

The Desktop's ``hermes://plugin/install?catalog=<name>`` deep link used to fetch the docs-site JSON
from the renderer. That URL answers with a cross-origin 301, so the browser's CORS check fails on the
redirect and every link reported the catalog as unreachable (hermes-agent 136483). The backend already
reads the live catalog (cache, failure window, in-tree fallback and kill list), and ``plugins.manage
install {catalog_name}`` resolves the pin from the same place, so the lookup lives here too.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("plugins.catalog.lookup")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from hermes_cli.plugin_catalog import load_catalog_live, match_removed, resolved_removed_entries
    from hermes_cli.plugins_cmd_catalog import looks_like_catalog_name

    name = str(params.get("name") or "").strip()
    if not looks_like_catalog_name(name):  # the same rule `hermes plugins install <name>` applies
        return _ok(rid, {"status": "invalid_name"})
    # The loaders never raise: a dead host falls back to the cache, then to the in-tree catalog.
    # Nothing readable at all is "unavailable", never a guess.
    entries = load_catalog_live()
    if not entries:
        return _ok(rid, {"status": "unavailable"})
    entry = next((e for e in entries if e.name == name), None)
    # A removed name or repo is never offered: install would refuse it anyway.
    removed = resolved_removed_entries() if entry else []
    if entry is None or match_removed(entry.name, removed) or match_removed(entry.repo, removed):
        return _ok(rid, {"status": "unknown"})
    return _ok(rid, {"status": "found", "entry": {"name": entry.name, "repo": entry.repo, "sha": entry.sha,
                                                  "subdir": entry.subdir, "tier": entry.tier}})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
