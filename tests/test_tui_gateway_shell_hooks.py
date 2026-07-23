"""Desktop surface shell-hook registration.

``hermes serve`` (the desktop backend) never passes through
``hermes_cli.main._prepare_agent_startup`` — ``"serve"`` is not in
``_AGENT_COMMANDS`` — so config-declared shell hooks (``hooks:`` in
config.yaml) were never registered in the process that actually runs the
agent turn.  The ``pre_llm_call`` dispatch itself is present and correct
(agent/turn_context.py) but iterated an empty hook table: every lifecycle
shell hook silently no-opped on the desktop surface while the identical
profile worked under ``hermes chat``.

These tests pin the fix: ``tui_gateway._make_agent`` — the shared
agent-build chokepoint for BOTH desktop sub-paths (serve's in-memory
gateway and the spawned ``tui_gateway.entry`` profile child) — must wire
shell hooks from the loaded config via the idempotent
``agent.shell_hooks.register_from_config``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod


@pytest.fixture()
def clean_hook_state():
    """Isolate the shell-hook idempotence set + plugin-manager hook table."""
    from agent import shell_hooks
    from hermes_cli.plugins import get_plugin_manager

    shell_hooks.reset_for_tests()
    manager = get_plugin_manager()
    saved = dict(manager._hooks)
    manager._hooks.clear()
    yield manager
    manager._hooks.clear()
    manager._hooks.update(saved)
    shell_hooks.reset_for_tests()


def _cfg():
    return {
        "hooks_auto_accept": True,
        "hooks": {
            "pre_llm_call": [
                {"command": "echo desktop-hook-test", "timeout": 5},
            ],
        },
    }


def test_ensure_shell_hooks_registers_pre_llm_call(server, clean_hook_state):
    """The desktop registration helper wires config hooks onto the manager."""
    manager = clean_hook_state
    server._ensure_shell_hooks(_cfg())
    assert manager._hooks.get("pre_llm_call"), (
        "desktop agent build must register config shell hooks on the plugin "
        "manager — pre_llm_call context injection is dead on this surface "
        "otherwise"
    )


def test_ensure_shell_hooks_swallows_failures(server, clean_hook_state):
    """A broken hooks block must never take down agent construction."""
    server._ensure_shell_hooks(None)          # not a dict
    server._ensure_shell_hooks({"hooks": 42})  # malformed block
    # reaching here without an exception is the assertion


def test_make_agent_wires_shell_hooks(server, clean_hook_state, monkeypatch):
    """_make_agent passes the loaded cfg through shell-hook registration."""
    calls = []
    monkeypatch.setattr(server, "_ensure_shell_hooks", lambda cfg: calls.append(cfg))
    monkeypatch.setattr(server, "_load_cfg", lambda: _cfg())
    monkeypatch.setattr(
        server, "_resolve_startup_runtime", lambda: ("test-model", None)
    )
    monkeypatch.setattr(
        server, "_resolve_runtime_with_fallback", lambda kw: {"provider": "openai"}
    )
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    with patch.dict(
        "sys.modules",
        {
            "run_agent": MagicMock(),
            "hermes_cli.mcp_startup": MagicMock(),
            "tui_gateway.entry": MagicMock(),
        },
    ):
        server._make_agent("sid-shell-hook-test", "key-shell-hook-test")
    assert calls, "_make_agent never invoked shell-hook registration"
    assert calls[0].get("hooks"), (
        "_make_agent must pass the loaded config (with its hooks block) to "
        "shell-hook registration"
    )
