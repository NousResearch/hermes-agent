"""vault-retrieval — Hermes user plugin entrypoint.

This module is the directory-level ``__init__.py`` that
:func:`hermes_cli.plugins_loader._load_directory_module` imports to
discover and register the plugin. The previous candidate shipped the
package without the directory-level ``__init__.py``, so ``discover_plugins``
rejected it with ``No __init__.py in <path>`` and ``vault_context``
never reached the registry.

Hermes's plugin loader exposes the plugin module as
``hermes_plugins.<slug>` (e.g. ``hermes_plugins.vault_retrieval``) and
sets ``__package__`` to that name so ``from .plugin import ...``
resolves to ``<plugin_dir>/<plugin>.py``. We import the package's
internal ``vault_retrieval`` module by absolute path so the
implementation lives in one place and the loader-facing module is
trivial.
"""
from __future__ import annotations

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path
from typing import Any


_PLUGIN_DIR = _Path(__file__).resolve().parent
_PKG_DIR = _PLUGIN_DIR / "vault_retrieval"


def _load_pkg_module() -> Any:
    """Lazy-load ``vault_retrieval.plugin`` so a missing inner package
    surfaces a clear error at register time, not at import time.
    """
    if "vault_retrieval" in _sys.modules:
        return _sys.modules["vault_retrieval.plugin"]
    pkg_init = _PKG_DIR / "__init__.py"
    if not pkg_init.exists():
        raise ImportError(
            f"vault-retrieval inner package missing: {pkg_init}"
        )
    pkg_name = "vault_retrieval"
    pkg_spec = _importlib_util.spec_from_file_location(
        pkg_name, pkg_init, submodule_search_locations=[str(_PKG_DIR)],
    )
    if pkg_spec is None or pkg_spec.loader is None:
        raise ImportError(f"Cannot load {pkg_init}")
    pkg_mod = _importlib_util.module_from_spec(pkg_spec)
    # Set ``__package__`` AND register in ``sys.modules`` BEFORE
    # ``exec_module`` so ``from .plugin import register`` inside the
    # inner package resolves correctly via spec_from_file_location.
    pkg_mod.__package__ = pkg_name
    _sys.modules[pkg_name] = pkg_mod
    pkg_spec.loader.exec_module(pkg_mod)
    plugin_path = _PKG_DIR / "plugin.py"
    plugin_spec = _importlib_util.spec_from_file_location(
        f"{pkg_name}.plugin", plugin_path,
    )
    if plugin_spec is None or plugin_spec.loader is None:
        raise ImportError(f"Cannot load {plugin_path}")
    plugin_mod = _importlib_util.module_from_spec(plugin_spec)
    plugin_mod.__package__ = pkg_name
    _sys.modules[f"{pkg_name}.plugin"] = plugin_mod
    plugin_spec.loader.exec_module(plugin_mod)
    return plugin_mod


def register(ctx: Any) -> None:
    """Hermes plugin entrypoint — delegate to the inner package."""
    plugin_mod = _load_pkg_module()
    plugin_mod.register(ctx)


__all__ = ["register"]