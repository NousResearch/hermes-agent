"""The discovery summary must name failing MCP servers with their reasons (#114746): an
aggregate ``(2 failed)`` count left the failing identity diagnosable only by elimination
from the per-server ``registered N tool(s)`` lines."""

import logging
from types import SimpleNamespace

import pytest

from tools.mcp_tool_discovery import _log_summary
from tools.mcp_tool_scope import _server_key


@pytest.fixture(autouse=True)
def _clean_server_state():
    from tools import mcp_tool

    yield
    with mcp_tool._lock:
        for name in ("ok", "bad", "ghost", "stale"):
            mcp_tool._servers.pop(_server_key(name), None)
            mcp_tool._server_connect_errors.pop(_server_key(name), None)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_failed_servers_are_named_with_recorded_reasons(caplog):
    from tools import mcp_tool

    with mcp_tool._lock:
        mcp_tool._servers[_server_key("ok")] = SimpleNamespace(
            _registered_tool_names=["t1"]
        )
        mcp_tool._server_connect_errors[_server_key("bad")] = (
            "connect timeout after 10s"
        )
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        _log_summary("MCP: registered", ["ok", "bad"])

    summaries = [
        r.getMessage() for r in caplog.records if "tool(s) from" in r.getMessage()
    ]
    assert summaries == ["MCP: registered 1 tool(s) from 1 server(s) (1 failed)"]
    assert _warnings(caplog) == [
        "MCP server 'bad' failed to register: connect timeout after 10s"
    ]


def test_failed_server_without_recorded_error_gets_a_diagnosable_fallback(caplog):
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        _log_summary("  MCP:", ["ghost"])

    assert _warnings(caplog) == [
        "MCP server 'ghost' failed to register: no connection error recorded"
    ]


def test_connected_server_with_stale_error_is_reported_not_silent(caplog):
    """A server that adopted a connection while an error from its previous attempt is still
    recorded counts as failed (``_connected_summary`` requires both states clean); its WARNING
    must carry the recorded reason rather than dropping the identity."""
    from tools import mcp_tool

    with mcp_tool._lock:
        mcp_tool._servers[_server_key("stale")] = SimpleNamespace(
            _registered_tool_names=["t1"]
        )
        mcp_tool._server_connect_errors[_server_key("stale")] = "401 Unauthorized"
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        _log_summary("MCP: registered", ["stale"])

    assert _warnings(caplog) == [
        "MCP server 'stale' failed to register: 401 Unauthorized"
    ]


def test_all_connected_summary_stays_warning_free(caplog):
    from tools import mcp_tool

    with mcp_tool._lock:
        mcp_tool._servers[_server_key("ok")] = SimpleNamespace(
            _registered_tool_names=["t1", "t2"]
        )
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        _log_summary("MCP: registered", ["ok"])

    assert [r.getMessage() for r in caplog.records] == [
        "MCP: registered 2 tool(s) from 1 server(s)"
    ]
    assert _warnings(caplog) == []


def test_recorded_reason_is_credential_scrubbed_before_logging(caplog):
    """Defence in depth for any future writer that records a raw exception string: whatever
    sits in the connect-error map is scrubbed on its way to the log."""
    from tools import mcp_tool

    with mcp_tool._lock:
        mcp_tool._server_connect_errors[_server_key("bad")] = (
            "auth failed: Bearer sk-supersecret123"
        )
    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        _log_summary("MCP: registered", ["bad"])

    assert _warnings(caplog) == [
        "MCP server 'bad' failed to register: auth failed: [REDACTED]"
    ]
