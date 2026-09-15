"""Regression coverage for retiring MCP runtimes removed from fresh config."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest


@contextmanager
def _profile_home(path: Path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(path))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _run_reconnect_after_config_change(tmp_path: Path, fresh_config: str) -> tuple[int, bool]:
    from tools import mcp_tool

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mcp_servers:\n  owned-server:\n    command: test\n",
        encoding="utf-8",
    )
    server = mcp_tool.MCPServerTask("owned-server")
    run_count = 0

    async def run_stdio(_config):
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            config_path.write_text(fresh_config, encoding="utf-8")
            return "reconnect"
        server._shutdown_event.set()
        return "reconnect"

    async def exercise():
        with patch.object(server, "_run_stdio", side_effect=run_stdio):
            await server.run({"command": "test"})

    with _profile_home(tmp_path):
        mcp_tool._servers[server.name] = server
        mcp_tool._server_scope_keys[server.name] = None
        try:
            asyncio.run(exercise())
            retained = mcp_tool._servers.get(server.name) is server
        finally:
            mcp_tool._servers.pop(server.name, None)
            mcp_tool._server_scope_keys.pop(server.name, None)
    return run_count, retained


def test_removed_server_retires_before_keepalive_reconnect(tmp_path):
    run_count, retained = _run_reconnect_after_config_change(tmp_path, "{}\n")

    assert run_count == 1
    assert retained is False


def test_retained_enabled_server_reconnects(tmp_path):
    run_count, retained = _run_reconnect_after_config_change(
        tmp_path,
        "mcp_servers:\n  owned-server:\n    command: test\n    enabled: true\n",
    )

    assert run_count == 2
    assert retained is True


def test_disabled_server_retires_before_reconnect(tmp_path):
    run_count, retained = _run_reconnect_after_config_change(
        tmp_path,
        "mcp_servers:\n  owned-server:\n    command: test\n    enabled: false\n",
    )

    assert run_count == 1
    assert retained is False


@pytest.mark.parametrize(
    "fresh_config",
    [
        "mcp_servers: [\n",
        "mcp_servers:\n  owned-server: []\n",
    ],
    ids=["unreadable-yaml", "ambiguous-membership"],
)
def test_uncertain_fresh_config_fails_open_to_reconnect(tmp_path, fresh_config):
    run_count, retained = _run_reconnect_after_config_change(tmp_path, fresh_config)

    assert run_count == 2
    assert retained is True
