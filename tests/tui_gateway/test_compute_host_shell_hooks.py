"""Turn-isolated compute-host shell-hook registration tests."""

from __future__ import annotations

import io
import json
import shlex
import sys

import pytest
import yaml

from agent import shell_hooks
from hermes_cli import plugins
from hermes_cli.plugins import get_pre_tool_call_block_message
from tui_gateway import compute_host


@pytest.fixture(autouse=True)
def _fresh_hook_registry(monkeypatch):
    previous_manager = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "0")
    monkeypatch.setenv("HERMES_COMPUTE_HOST_HEARTBEAT_SECS", "0")
    yield
    shell_hooks.reset_for_tests()
    plugins._plugin_manager = previous_manager


def _configure_allowlisted_block_hook(tmp_path, monkeypatch) -> None:
    home = tmp_path / "hermes-home"
    script = tmp_path / "block.py"
    script.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        'print(\'{"action": "block", "message": "compute-hook-fired"}\')\n',
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


def _run_host(monkeypatch, events) -> list[dict]:
    monkeypatch.setattr(plugins, "discover_plugins", lambda: events.append("plugins"))
    original_emit = compute_host.ComputeHost.emit

    def emit(host, frame):
        if frame.get("type") == "hello":
            events.append("hello")
        return original_emit(host, frame)

    monkeypatch.setattr(compute_host.ComputeHost, "emit", emit)
    stdout = io.StringIO()
    compute_host.run_host(stdin=io.StringIO(""), stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_compute_host_registers_allowlisted_hook_before_hello(tmp_path, monkeypatch):
    _configure_allowlisted_block_hook(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_PROFILE_SCOPED_UI", "1")
    events = []
    original_register = shell_hooks.register_from_config

    def register(cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        return original_register(cfg, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", register)

    frames = _run_host(monkeypatch, events)

    assert events == ["plugins", ("hooks", False), "hello"]
    assert frames[0]["type"] == "hello"
    assert (
        get_pre_tool_call_block_message("terminal", {"command": "echo should-not-run"})
        == "compute-hook-fired"
    )


def test_compute_host_registration_failure_does_not_block_hello(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE_SCOPED_UI", "1")
    events = []

    def fail_registration(_cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        raise RuntimeError("test registration failure")

    monkeypatch.setattr(shell_hooks, "register_from_config", fail_registration)

    frames = _run_host(monkeypatch, events)

    assert events == ["plugins", ("hooks", False), "hello"]
    assert frames[0]["type"] == "hello"


def test_shared_dashboard_compute_host_does_not_register_profile_hooks(
    tmp_path, monkeypatch
):
    _configure_allowlisted_block_hook(tmp_path, monkeypatch)
    monkeypatch.delenv("HERMES_PROFILE_SCOPED_UI", raising=False)
    events = []
    original_register = shell_hooks.register_from_config

    def register(cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        return original_register(cfg, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", register)

    frames = _run_host(monkeypatch, events)

    assert events == ["hello"]
    assert frames[0]["type"] == "hello"
    assert (
        get_pre_tool_call_block_message("terminal", {"command": "echo allowed"}) is None
    )
