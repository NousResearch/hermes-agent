"""Behavior contract: the tui_gateway onboarding methods module must import.

``methods_onboarding.py`` annotates ``_shared_nous_store`` with
``tuple[Path, ...]`` but (before this fix) never imported ``Path``.
Python evaluates annotations at ``def`` time, so the whole module failed
to import with ``NameError: name 'Path' is not defined`` --
unregistering every RPC method it binds. Consumers (cron session spawns
over the gateway WS) then died with
``InvalidStatus: server rejected WebSocket connection: HTTP 500``.
"""

import importlib


def test_methods_onboarding_module_imports():
    importlib.import_module("tui_gateway.methods_onboarding")
