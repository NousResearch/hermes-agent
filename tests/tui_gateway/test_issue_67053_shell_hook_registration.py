"""Regression test for issue #67053 — tui_gateway daemon must register
config-based shell hooks so they dispatch during an interactive
``hermes chat`` session.

Before the fix, ``register_from_config`` ran only in the transient
``hermes chat`` launcher process. The long-lived ``tui_gateway``
daemon built its agent via ``_make_agent`` without ever registering
hooks, and ``PluginManager._hooks`` (an in-memory per-process dict)
ended up with only the built-in plugin callbacks. Configured shell
hooks silently never fired during interactive sessions even though
``hermes hooks doctor`` / ``list`` / ``test`` reported healthy.

The fix introduces ``tui_gateway.server._register_shell_hooks_from_config``
and calls it from inside ``_make_agent``. The function is idempotent
(deduped on ``(event, matcher, command)``), so launcher-side
pre-registration is a safe no-op when both call paths run.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    """Reset shell-hook registration state and use a per-test HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")

    from agent import shell_hooks
    from hermes_cli import plugins

    shell_hooks.reset_for_tests()
    plugins._plugin_manager = plugins.PluginManager()
    yield
    shell_hooks.reset_for_tests()
    plugins._plugin_manager = plugins.PluginManager()


def _write_hook_script(tmp_path: Path, name: str = "hook.sh") -> Path:
    script = tmp_path / name
    script.write_text("#!/usr/bin/env bash\nprintf '{}\\n'\n")
    script.chmod(0o755)
    return script


def _write_config_yaml(home: Path, hook_command: str) -> Path:
    """Write a minimal config.yaml with one shell hook under HERMES_HOME."""
    config_path = home / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "hooks:\n"
        "  on_session_start:\n"
        f"    - command: {hook_command}\n"
    )
    return config_path


def test_register_shell_hooks_helper_exists_and_registers(tmp_path):
    """The fix added ``_register_shell_hooks_from_config`` to
    ``tui_gateway.server``. Calling it must wire up the configured
    shell hook on the in-process ``PluginManager`` — this is the exact
    code path ``_make_agent`` runs during agent build."""
    from hermes_cli import plugins

    script = _write_hook_script(tmp_path)
    _write_config_yaml(Path(os.environ["HERMES_HOME"]), str(script))

    server = importlib.import_module("tui_gateway.server")
    assert hasattr(server, "_register_shell_hooks_from_config"), (
        "fix for #67053 missing: tui_gateway.server must expose "
        "_register_shell_hooks_from_config"
    )

    server._register_shell_hooks_from_config()

    mgr = plugins.get_plugin_manager()
    on_session_start = mgr._hooks.get("on_session_start", [])
    assert any(
        "shell_hook" in repr(cb).lower() for cb in on_session_start
    ), (
        f"shell hook callback not registered on plugin manager after "
        f"calling _register_shell_hooks_from_config; got {on_session_start!r}"
    )


def test_register_shell_hooks_helper_is_idempotent(tmp_path):
    """Two consecutive calls must dedupe to a single registration. This
    is the property that makes it safe to deploy the fix alongside the
    existing CLI registration call site."""
    from hermes_cli import plugins

    script = _write_hook_script(tmp_path, "h.sh")
    _write_config_yaml(Path(os.environ["HERMES_HOME"]), str(script))

    server = importlib.import_module("tui_gateway.server")

    server._register_shell_hooks_from_config()
    server._register_shell_hooks_from_config()

    mgr = plugins.get_plugin_manager()
    on_session_start = mgr._hooks.get("on_session_start", [])
    shell_hook_cbs = [cb for cb in on_session_start if "shell_hook" in repr(cb).lower()]
    assert len(shell_hook_cbs) == 1, (
        f"idempotency broken: expected 1 shell hook callback, got "
        f"{len(shell_hook_cbs)} ({on_session_start!r})"
    )


def test_safe_mode_skips_registration(tmp_path, monkeypatch):
    """``HERMES_SAFE_MODE=1`` must short-circuit registration — a
    troubleshooting run should fire zero user-configured code (plugins,
    MCP, AND hooks). The helper must inherit this property from
    ``register_from_config``."""
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    from hermes_cli import plugins

    script = _write_hook_script(tmp_path, "safe.sh")
    _write_config_yaml(Path(os.environ["HERMES_HOME"]), str(script))

    server = importlib.import_module("tui_gateway.server")
    server._register_shell_hooks_from_config()

    mgr = plugins.get_plugin_manager()
    on_session_start = mgr._hooks.get("on_session_start", [])
    shell_hook_cbs = [cb for cb in on_session_start if "shell_hook" in repr(cb).lower()]
    assert shell_hook_cbs == [], (
        f"HERMES_SAFE_MODE=1 must skip shell-hook registration; got {shell_hook_cbs!r}"
    )