"""Contract tests for the direct POSIX stdio MCP child watchdog."""

import asyncio
import os
import sys

import pytest

from tools import mcp_stdio_watchdog, mcp_tool


def test_is_orphaned_is_false_while_direct_parent_is_unchanged():
    original_ppid = 1234

    assert mcp_stdio_watchdog._is_orphaned(
        original_ppid,
        getppid=lambda: original_ppid,
    ) is False


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
def test_wrap_command_uses_stable_parent_pid_and_preserves_command_tail():
    parent_pid = os.getpid()
    command = "/opt/hermes/bin/mcp-server"
    command_args = ["--label", "value with spaces", "--", "literal-tail"]

    wrapped_command, wrapped_args = mcp_tool._wrap_command_with_watchdog(
        command,
        command_args,
    )

    assert wrapped_command == sys.executable
    assert wrapped_args == [
        os.path.join(os.path.dirname(mcp_tool.__file__), "mcp_stdio_watchdog.py"),
        "--ppid",
        str(parent_pid),
        "--",
        command,
        *command_args,
    ]
    assert "--create-time" not in wrapped_args


class _StdioReached(Exception):
    """Raised from the patched transport once the spawn argv is observable."""


def _spawn_params_for(monkeypatch, config):
    """Drive ``_run_stdio`` up to the transport and return its spawn params.

    Stops at ``stdio_client`` so nothing is actually executed -- the argv it
    would have spawned is the whole assertion surface here.
    """
    captured = {}

    def fake_stdio_client(server_params, **_kwargs):
        captured["params"] = server_params
        raise _StdioReached

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp_tool, "_resolve_stdio_command", lambda cmd, env: (cmd, env))
    monkeypatch.setattr(
        "tools.osv_check.check_package_for_malware", lambda *_a, **_kw: None
    )

    task = mcp_tool.MCPServerTask("probe")
    with pytest.raises(_StdioReached):
        asyncio.run(task._run_stdio(config))

    return captured["params"]


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
def test_stdio_spawn_is_watchdog_wrapped_by_default(monkeypatch):
    params = _spawn_params_for(
        monkeypatch,
        {"command": "/opt/hermes/bin/mcp-server", "args": ["--flag"]},
    )

    assert params.command == sys.executable
    assert params.args[0].endswith("mcp_stdio_watchdog.py")
    assert params.args[-2:] == ["/opt/hermes/bin/mcp-server", "--flag"]


def test_watchdog_false_spawns_the_real_command_directly(monkeypatch):
    """``watchdog: false`` opts a server out of the supervisor wrapper.

    Some binaries break under the supervisor's new-session process group (Go
    binaries that manage their own guard subprocesses), so the escape hatch
    trades orphan-reaping for a working connection.
    """
    params = _spawn_params_for(
        monkeypatch,
        {
            "command": "/opt/hermes/bin/mcp-server",
            "args": ["--flag"],
            "watchdog": False,
        },
    )

    assert params.command == "/opt/hermes/bin/mcp-server"
    assert params.args == ["--flag"]

