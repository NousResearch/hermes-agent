"""Profile-scoped slash-worker shell-hook registration tests."""

from __future__ import annotations

import io
import os
import shlex
import sys

import pytest
import yaml

from agent import shell_hooks
from hermes_cli import plugins
from hermes_cli.plugins import get_pre_tool_call_block_message
from tui_gateway import slash_worker


@pytest.fixture(autouse=True)
def _fresh_hook_registry():
    previous_manager = plugins._plugin_manager
    previous_session_key = os.environ.get("HERMES_SESSION_KEY")
    previous_interactive = os.environ.get("HERMES_INTERACTIVE")
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()
    plugins._plugin_manager = previous_manager
    for key, previous in (
        ("HERMES_SESSION_KEY", previous_session_key),
        ("HERMES_INTERACTIVE", previous_interactive),
    ):
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _configure_allowlisted_block_hook(tmp_path, monkeypatch) -> None:
    home = tmp_path / "hermes-home"
    script = tmp_path / "block.py"
    script.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        'print(\'{"action": "block", "message": "slash-hook-fired"}\')\n',
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


def _stub_worker_runtime(monkeypatch, events) -> None:
    monkeypatch.setattr(
        slash_worker.sys,
        "argv",
        ["slash_worker", "--session-key", "agent:main:tui:dm:shell-hook-test"],
    )
    monkeypatch.setattr(slash_worker.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(
        slash_worker, "_start_parent_death_watchdog", lambda _ppid: None
    )
    monkeypatch.setattr(
        slash_worker,
        "_prepare_slash_worker_runtime",
        lambda: events.append("mcp"),
    )
    monkeypatch.setattr(plugins, "discover_plugins", lambda: events.append("plugins"))

    class FakeCLI:
        def __init__(self, **_kwargs):
            events.append("cli")

    monkeypatch.setattr(slash_worker, "HermesCLI", FakeCLI)


def test_slash_worker_registers_allowlisted_hook_before_cli(tmp_path, monkeypatch):
    _configure_allowlisted_block_hook(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_PROFILE_SCOPED_UI", "1")
    events = []
    _stub_worker_runtime(monkeypatch, events)
    original_register = shell_hooks.register_from_config

    def register(cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        return original_register(cfg, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", register)

    slash_worker.main()

    assert events == ["plugins", ("hooks", False), "mcp", "cli"]
    assert (
        get_pre_tool_call_block_message("terminal", {"command": "echo should-not-run"})
        == "slash-hook-fired"
    )


def test_slash_worker_registration_failure_does_not_block_cli(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE_SCOPED_UI", "1")
    events = []
    _stub_worker_runtime(monkeypatch, events)

    def fail_registration(_cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        raise RuntimeError("test registration failure")

    monkeypatch.setattr(shell_hooks, "register_from_config", fail_registration)

    slash_worker.main()

    assert events == ["plugins", ("hooks", False), "mcp", "cli"]


def test_shared_dashboard_slash_worker_does_not_register_profile_hooks(
    tmp_path, monkeypatch
):
    _configure_allowlisted_block_hook(tmp_path, monkeypatch)
    monkeypatch.delenv("HERMES_PROFILE_SCOPED_UI", raising=False)
    events = []
    _stub_worker_runtime(monkeypatch, events)
    original_register = shell_hooks.register_from_config

    def register(cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        return original_register(cfg, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", register)

    slash_worker.main()

    assert events == ["mcp", "cli"]
    assert (
        get_pre_tool_call_block_message("terminal", {"command": "echo allowed"}) is None
    )
