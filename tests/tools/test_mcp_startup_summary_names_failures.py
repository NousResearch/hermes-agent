"""The MCP startup summary names every failing server with its reason (#114746).

``MCP: registered N tool(s) from M server(s) (2 failed)`` left the failing identity diagnosable
only by elimination from the per-server ``registered`` lines, and a candidate skipped for its
retry cooldown never got a per-server WARNING at all.
"""

import logging
from types import SimpleNamespace

import pytest

from tools.mcp_tool_scope import _server_key


@pytest.fixture
def _clean_registry():
    from tools import mcp_tool

    yield
    with mcp_tool._lock:
        for name in ("ghost", "cooled", "ok"):
            mcp_tool._servers.pop(_server_key(name), None)
            mcp_tool._server_connect_errors.pop(_server_key(name), None)
            mcp_tool._server_connect_retry_after.pop(_server_key(name), None)
            mcp_tool._server_connecting.discard(_server_key(name))


def _summaries(caplog):
    return [r.getMessage() for r in caplog.records if "tool(s) from" in r.getMessage()]


@pytest.mark.no_isolate
def test_register_summary_names_failed_server_with_reason(monkeypatch, tmp_path, caplog, _clean_registry):
    """Production path: ``register_mcp_servers`` with a stdio command that cannot start. The
    summary line itself carries the server name and the recorded connect error."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_discovery import register_mcp_servers

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 1)
    monkeypatch.setattr(mcp_tool, "_MAX_BACKOFF_SECONDS", 0.1)
    missing = str(tmp_path / "no-such-mcp-binary")

    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        register_mcp_servers({"ghost": {"command": missing, "connect_timeout": 15}})

    summaries = _summaries(caplog)
    assert len(summaries) == 1, summaries
    assert summaries[0].startswith("MCP: registered 0 tool(s) from 0 server(s) (1 failed: ghost (")
    assert "no-such-mcp-binary" in summaries[0]


def test_summary_marks_candidate_skipped_for_cooldown(caplog, _clean_registry):
    """A candidate this pass never attempted (still inside its retry cooldown) has no fresh
    connect error: it is named with an explicit "not attempted" reason instead of vanishing
    into the count; a healthy server keeps the summary unchanged."""
    from tools import mcp_tool
    from tools.mcp_tool_discovery import _log_summary

    with mcp_tool._lock:
        mcp_tool._servers[_server_key("ok")] = SimpleNamespace(_registered_tool_names=["t1", "t2"])
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        _log_summary("  MCP:", ["ok", "cooled"])

    assert _summaries(caplog) == [
        "  MCP: 2 tool(s) from 1 server(s) (1 failed: cooled (not attempted (in retry cooldown)))"
    ]
