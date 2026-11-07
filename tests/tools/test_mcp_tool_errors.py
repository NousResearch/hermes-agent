"""Invariants for ``tools/mcp_tool_errors._format_connect_error`` on malformed exception chains.

``__cause__``/``__context__`` can form a cycle (the same OAuth error re-raised on the SSE fallback,
a raised-and-caught pair) and stdio failures can nest deeper than the recursion limit; either used
to turn ``hermes mcp test`` into a RecursionError that hid the real connect error (#111952, #111997).
"""
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.mcp_tool_errors import _format_connect_error


def test_format_connect_error_reports_real_messages_on_cyclic_chain():
    """A two-node ``__cause__``/``__context__`` cycle renders every distinct message, once, in chain order."""
    first = RuntimeError("first failure")
    second = RuntimeError("second failure")
    first.__cause__ = second
    second.__context__ = first

    assert _format_connect_error(first) == "first failure; second failure"


def test_format_connect_error_finds_missing_executable_through_deep_cyclic_chain():
    """A missing stdio binary wrapped deeper than the recursion limit, with the chain looping back to the top,
    is still reported as the missing executable rather than as a RecursionError."""
    missing = FileNotFoundError(2, "No such file or directory", "/opt/homebrew/bin/removed-mcp-server")
    current = missing
    for _ in range(sys.getrecursionlimit() + 10):
        wrapper = RuntimeError("stdio startup failed")
        wrapper.__cause__ = current
        current = wrapper
    missing.__context__ = current

    assert _format_connect_error(current) == "missing executable '/opt/homebrew/bin/removed-mcp-server'"


# ---------------------------------------------------------------------------
# A 401 on tools/call that the SDK folded into a generic error (#121285)
# ---------------------------------------------------------------------------

def _folded_sdk_error():
    """The error mcp 2.0 raises for a non-404 >= 400 it cannot parse as JSON-RPC."""
    mcp_shared = pytest.importorskip("mcp.shared.exceptions", reason="mcp SDK not installed")
    mcp_types = pytest.importorskip("mcp.types", reason="mcp SDK not installed")
    return mcp_shared.MCPError(mcp_types.INTERNAL_ERROR, "Server returned an error response")

def test_folded_error_alone_is_not_an_auth_error():
    """The premise: the SDK discards the status, so the type check cannot see a 401."""
    from tools.mcp_tool_errors import _is_auth_error

    assert _is_auth_error(_folded_sdk_error()) is False

def test_a_recorded_401_is_recognised_as_an_auth_error():
    """The recorded status is the only surviving evidence of the 401."""
    from tools.mcp_tool_errors import _is_recorded_auth_error

    server = SimpleNamespace(_http_rejection={"status": 401, "method": "POST"})
    assert _is_recorded_auth_error(server) is True

def test_only_401_counts_as_a_recorded_auth_error():
    """Every other rejection must keep its own classification."""
    from tools.mcp_tool_errors import _is_recorded_auth_error

    for status in (400, 403, 404, 405, 429, 500, 502, 503):
        server = SimpleNamespace(_http_rejection={"status": status})
        assert _is_recorded_auth_error(server) is False, f"status {status} was read as an auth error"

def test_no_recording_is_not_an_auth_error():
    """A server that never recorded a rejection must not be treated as auth-failing."""
    from tools.mcp_tool_errors import _is_recorded_auth_error

    assert _is_recorded_auth_error(SimpleNamespace(_http_rejection={})) is False
    assert _is_recorded_auth_error(SimpleNamespace()) is False
    assert _is_recorded_auth_error(object()) is False

def test_a_malformed_recorded_status_is_not_an_auth_error():
    """A junk sink must not raise, and must not classify."""
    from tools.mcp_tool_errors import _recorded_status_is_401

    assert _recorded_status_is_401({"status": "not-a-number"}) is False
    assert _recorded_status_is_401({"status": None}) is False
    assert _recorded_status_is_401(None) is False


# ---------------------------------------------------------------------------
# The auth-recovery ladder takes the recorded-401 path (#121285)
# ---------------------------------------------------------------------------

def _install_server_with_rejection(status):
    """Register a fake live server whose response hook recorded *status*."""
    from tools import mcp_tool as _core
    from tools import mcp_tool_handlers as _handlers
    from tools.mcp_tool_scope import _resolve_server_key

    server = SimpleNamespace(_http_rejection={"status": status})
    key = _resolve_server_key("web")
    with _core._lock:
        previous = _core._servers.get(key)
        _core._servers[key] = server
    return server, key, previous, _handlers

def test_a_recorded_401_reaches_the_auth_recovery_ladder():
    """The user-visible outcome: a 401 on tools/call must be handled as an auth failure.

    Without the recorded status the handler returns None and the model sees an opaque
    ``MCPError`` instead of the ``needs_reauth`` shape that stops it refreshing.
    """
    from tools import mcp_tool_handlers as _handlers

    server, key, previous, _ = _install_server_with_rejection(401)
    try:
        with patch.object(_handlers, "_is_auth_error", return_value=False), \
             patch.object(_handlers, "_strike", return_value="NEEDS_REAUTH") as strike:
            result = _handlers._handle_auth_error_and_retry(
                "web", _folded_sdk_error(), lambda: None, "call")
        assert result == "NEEDS_REAUTH", (
            "a recorded 401 did not reach the auth ladder; the model would see a transport error")
        assert strike.called
        # Consumed, so a later unrelated failure is not reclassified as an auth error.
        assert server._http_rejection == {}
    finally:
        from tools import mcp_tool as _core
        with _core._lock:
            if previous is None:
                _core._servers.pop(key, None)
            else:
                _core._servers[key] = previous

def test_a_recorded_403_does_not_reach_the_auth_ladder():
    """Only a 401 is an auth failure; other rejections keep their own handling."""
    from tools import mcp_tool_handlers as _handlers

    _server, key, previous, _ = _install_server_with_rejection(403)
    try:
        with patch.object(_handlers, "_is_auth_error", return_value=False):
            result = _handlers._handle_auth_error_and_retry(
                "web", _folded_sdk_error(), lambda: None, "call")
        assert result is None, "a 403 was routed through OAuth recovery"
    finally:
        from tools import mcp_tool as _core
        with _core._lock:
            if previous is None:
                _core._servers.pop(key, None)
            else:
                _core._servers[key] = previous
