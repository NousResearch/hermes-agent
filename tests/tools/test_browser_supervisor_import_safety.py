"""Import-safety regression tests for ``tools.browser_supervisor``.

Issue #31005: the Homebrew bottle ships a slim venv that omits
``websockets`` (a transitive dependency that hermes-agent does not declare
as its own runtime dep). Before the fix, the top-level
``import websockets`` in ``tools/browser_supervisor.py`` raised
``ModuleNotFoundError`` whenever tool discovery touched the module, and
``hermes`` startup logged
``Could not import tool module tools.browser_dialog_tool: No module named 'websockets'``
because ``browser_dialog_tool`` imports ``SUPERVISOR_REGISTRY`` from
``browser_supervisor`` at module level.

These tests prove:

1. ``tools.browser_supervisor`` continues to import cleanly even when
   ``websockets`` is unavailable.
2. The downstream ``tools.browser_dialog_tool`` also imports cleanly, so
   tool registration no longer fails at startup.
3. Attempting to actually start a supervisor without ``websockets`` raises
   a clear ImportError pointing to the install command, instead of a
   confusing ``NameError`` deep inside the background thread.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def _block_websockets_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``builtins.__import__`` to raise ``ImportError`` for ``websockets*``.

    Mirrors the canonical pattern used in
    ``tests/tools/test_memory_tool_import_fallback.py`` and
    ``tests/gateway/test_discord_imports.py``: rather than setting
    ``sys.modules[name] = None`` (which can leak through to attribute access),
    intercept the import call itself so any form of ``import websockets`` /
    ``from websockets... import X`` raises the same ``ImportError`` Python
    would raise if the package were genuinely missing.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "websockets" or name.startswith("websockets."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _reload_supervisor_module(
    monkeypatch: pytest.MonkeyPatch, available: bool
):
    """Reload ``tools.browser_supervisor`` with websockets present or absent."""
    if not available:
        for cached in ("websockets", "websockets.asyncio", "websockets.asyncio.client"):
            monkeypatch.delitem(sys.modules, cached, raising=False)
        _block_websockets_import(monkeypatch)
    monkeypatch.delitem(sys.modules, "tools.browser_supervisor", raising=False)
    monkeypatch.delitem(sys.modules, "tools.browser_dialog_tool", raising=False)
    return importlib.import_module("tools.browser_supervisor")


def test_browser_supervisor_imports_without_websockets(monkeypatch):
    module = _reload_supervisor_module(monkeypatch, available=False)
    assert module._WS_AVAILABLE is False
    assert module.SUPERVISOR_REGISTRY is not None


def test_browser_dialog_tool_imports_without_websockets(monkeypatch):
    _reload_supervisor_module(monkeypatch, available=False)
    dialog = importlib.import_module("tools.browser_dialog_tool")
    assert dialog.SUPERVISOR_REGISTRY is not None


def test_supervisor_start_raises_clear_importerror_without_websockets(monkeypatch):
    module = _reload_supervisor_module(monkeypatch, available=False)
    supervisor = module.CDPSupervisor(task_id="t", cdp_url="ws://example/devtools")
    with pytest.raises(ImportError, match="websockets"):
        supervisor.start()


def test_supervisor_importerror_message_includes_min_version_and_original(monkeypatch):
    module = _reload_supervisor_module(monkeypatch, available=False)
    supervisor = module.CDPSupervisor(task_id="t", cdp_url="ws://example/devtools")
    with pytest.raises(ImportError) as exc:
        supervisor.start()
    message = str(exc.value)
    assert ">=13" in message, "expected min-version hint in error message"
    assert "pip install" in message
    assert "original error" in message, "expected original ImportError detail"


def test_browser_supervisor_module_state_with_websockets_present():
    pytest.importorskip("websockets")
    sys.modules.pop("tools.browser_supervisor", None)
    module = importlib.import_module("tools.browser_supervisor")
    assert module._WS_AVAILABLE is True
    assert module.websockets is not None
