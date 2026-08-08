"""Standalone TUI shell-hook registration regression tests."""

from __future__ import annotations

import io
import json
import os
import shlex
import sys

import pytest
import yaml

from agent import shell_hooks
from hermes_cli import plugins
from hermes_cli.plugins import (
    get_pre_tool_call_block_message,
    get_pre_tool_call_directive,
)
from tui_gateway import entry


@pytest.fixture(autouse=True)
def _fresh_hook_registry():
    previous_manager = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()
    plugins._plugin_manager = previous_manager
    os.environ.pop("HERMES_PROFILE_SCOPED_UI", None)


def _configure_allowlisted_hook(
    tmp_path,
    monkeypatch,
    *,
    action: str = "block",
    message: str = "tui-hook-fired",
) -> None:
    home = tmp_path / "hermes-home"
    script = tmp_path / "hook.py"
    directive = json.dumps({"action": action, "message": message})
    script.write_text(
        f"import json, sys\njson.load(sys.stdin)\nprint({directive!r})\n",
        encoding="utf-8",
    )
    command = shlex.join([sys.executable, str(script)])
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "hooks": {
                "pre_tool_call": [
                    {"command": command, "matcher": "terminal"},
                ],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    shell_hooks._record_approval("pre_tool_call", command)


def _stub_tui_runtime(monkeypatch, events) -> None:
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: {})
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *_args: False)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(plugins, "discover_plugins", lambda: events.append("plugins"))
    monkeypatch.setattr(
        entry,
        "write_json",
        lambda message: events.append(message["params"]["type"]) or True,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.prewarm_picker_cache_async", lambda: None
    )


def test_tui_registers_allowlisted_hook_before_ready(tmp_path, monkeypatch):
    _configure_allowlisted_hook(tmp_path, monkeypatch)
    events = []
    _stub_tui_runtime(monkeypatch, events)

    original_register = shell_hooks.register_from_config

    def register(cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        return original_register(cfg, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", register)

    entry.main()

    assert events == ["plugins", ("hooks", False), "gateway.ready"]
    assert os.environ["HERMES_PROFILE_SCOPED_UI"] == "1"
    assert (
        get_pre_tool_call_block_message("terminal", {"command": "echo should-not-run"})
        == "tui-hook-fired"
    )


def test_tui_registration_failure_does_not_block_ready(monkeypatch):
    events = []
    _stub_tui_runtime(monkeypatch, events)

    def fail_registration(_cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        raise RuntimeError("test registration failure")

    monkeypatch.setattr(shell_hooks, "register_from_config", fail_registration)

    entry.main()

    assert events == ["plugins", ("hooks", False), "gateway.ready"]


def test_plugin_approval_precedes_shell_hook_block(tmp_path, monkeypatch):
    _configure_allowlisted_hook(
        tmp_path,
        monkeypatch,
        action="block",
        message="shell-blocked",
    )
    events = []
    _stub_tui_runtime(monkeypatch, events)

    def discover_plugins():
        events.append("plugins")
        plugins.get_plugin_manager()._hooks.setdefault("pre_tool_call", []).append(
            lambda **_kwargs: {
                "action": "approve",
                "message": "plugin-approval",
            }
        )

    monkeypatch.setattr(plugins, "discover_plugins", discover_plugins)
    original_register = shell_hooks.register_from_config

    def register(cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        return original_register(cfg, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", register)

    entry.main()

    assert events == ["plugins", ("hooks", False), "gateway.ready"]
    assert get_pre_tool_call_directive("terminal", {"command": "echo guarded"}) == (
        "approve",
        "plugin-approval",
    )
