"""MCP startup failures must be diagnosable from the log (#114746).

The discovery summary reports an aggregate ``N failed`` count. That count is derived from
cached state, not from the pass's own connect attempts: a server skipped because its retry
cooldown is still active is counted as failed without anything naming it, and even a fresh
failure is only identifiable by cross-reading the per-server connect warnings. These tests
pin one line per failed server carrying its name and the recorded reason.
"""

import logging
from unittest.mock import patch

import pytest

import tools.mcp_tool as mcp_mod
from tools import mcp_tool_config as _mcp_config
from tools import mcp_tool_discovery as _mcp_discovery
from tools import mcp_tool_loop as _mcp_loop
from tools.mcp_tool_scope import _server_key


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    """Snapshot and restore the module-level MCP state around each test."""
    snapshot = (
        dict(mcp_mod._servers),
        set(mcp_mod._server_connecting),
        dict(mcp_mod._server_connect_errors),
        dict(mcp_mod._server_connect_retry_after),
        dict(mcp_mod._server_connect_failures),
    )
    mcp_mod._servers.clear()
    mcp_mod._server_connecting.clear()
    mcp_mod._server_connect_errors.clear()
    mcp_mod._server_connect_retry_after.clear()
    mcp_mod._server_connect_failures.clear()
    try:
        yield
    finally:
        (servers, connecting, errors, retry_after, failures) = snapshot
        mcp_mod._servers.clear(); mcp_mod._servers.update(servers)
        mcp_mod._server_connecting.clear(); mcp_mod._server_connecting.update(connecting)
        mcp_mod._server_connect_errors.clear(); mcp_mod._server_connect_errors.update(errors)
        mcp_mod._server_connect_retry_after.clear(); mcp_mod._server_connect_retry_after.update(retry_after)
        mcp_mod._server_connect_failures.clear(); mcp_mod._server_connect_failures.update(failures)


def _connect_with_failures(failures: dict):
    """A ``_connect_server`` stand-in: names in *failures* raise, everything else connects."""
    async def _connect(name, config):
        exc = failures.get(name)
        if exc is not None:
            raise exc
        server = mcp_mod.MCPServerTask(name)
        server._registered_tool_names = []
        server._tools = []
        return server
    return _connect


def _discovery_pass(caplog, servers: dict, failures: dict, tools: tuple = ()) -> list:
    """Run one ``discover_mcp_tools()`` pass; return that pass's log messages (oldest first)."""
    caplog.clear()
    with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
            patch.object(_mcp_config, "_load_mcp_config", return_value=dict(servers)), \
            patch.object(_mcp_config, "_filter_suspicious_mcp_servers", side_effect=lambda x: x), \
            patch.object(_mcp_discovery, "_connect_server", side_effect=_connect_with_failures(failures)), \
            patch.object(_mcp_discovery._registration, "_register_server_tools", return_value=list(tools)), \
            patch.object(_mcp_loop, "_try_acquire_mcp_discovery_lock", return_value=mcp_mod._LOCK_UNAVAILABLE), \
            caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        _mcp_discovery.discover_mcp_tools()
    return [record.getMessage() for record in caplog.records]


def test_failed_server_is_logged_with_name_and_reason(caplog):
    """A server that fails to connect is named, with the underlying reason, next to the count."""
    messages = _discovery_pass(caplog, {"good": {"command": "good"}, "bad": {"command": "bad"}},
                               {"bad": ConnectionError("HTTP 401 unauthorized")})

    assert any("1 failed" in m for m in messages), messages
    assert any("'bad'" in m and "HTTP 401 unauthorized" in m for m in messages), messages


def test_server_still_in_retry_cooldown_is_named_when_counted_failed(caplog):
    """A server whose failure happened in an EARLIER pass (cooldown active, so this pass never
    attempts it) is still counted ``failed``; that pass must name it and give the reason, or the
    only remaining evidence is a summary count."""
    servers = {"good": {"command": "good"}, "bad": {"command": "bad"}}
    _discovery_pass(caplog, servers, {"bad": ConnectionError("exec: bad: not found")})
    assert _mcp_discovery._connect_cooldown_active("bad") is True  # precondition: next pass skips it

    messages = _discovery_pass(caplog, servers, {})

    assert any("1 failed" in m for m in messages), messages
    assert any("'bad'" in m and "exec: bad: not found" in m for m in messages), messages


def test_no_failure_lines_when_every_server_connects(caplog):
    """All-success start: the summary carries no failure count and nothing is logged per server."""
    messages = _discovery_pass(caplog, {"one": {"command": "one"}, "two": {"command": "two"}}, {},
                               tools=("mcp__one__t",))

    assert any("2 server(s)" in m for m in messages), messages
    assert not [m for m in messages if "failed" in m], messages
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], \
        [r.getMessage() for r in caplog.records]


def test_recorded_reason_is_redacted_before_logging(caplog):
    """A reason recorded by any path is credential-scrubbed on its way to the log."""
    mcp_mod._server_connect_errors[_server_key("bad")] = "auth failed: Bearer sk-supersecret123"
    _mcp_discovery._record_connect_failure("bad")  # arms the cooldown: this pass skips the attempt

    messages = _discovery_pass(caplog, {"good": {"command": "good"}, "bad": {"command": "bad"}}, {})

    assert any("'bad'" in m for m in messages), messages
    assert not [m for m in messages if "sk-supersecret123" in m], messages
    assert any("[REDACTED]" in m for m in messages), messages
