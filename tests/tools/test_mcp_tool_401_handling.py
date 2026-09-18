"""Tests for MCP tool-handler auth-failure detection.

When a tool call raises UnauthorizedError / OAuthNonInteractiveError /
httpx.HTTPStatusError(401), the handler should:
  1. Ask MCPOAuthManager.handle_401 if recovery is viable.
  2. If yes, trigger MCPServerTask._reconnect_event and retry once.
  3. If no, return a structured needs_reauth error so the model stops
     hallucinating manual refresh attempts.
"""
import asyncio
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")
from tools import mcp_tool_loop as _mcp_loop  # noqa: E402


def test_is_auth_error_detects_oauth_flow_error():
    from tools.mcp_tool_errors import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    assert _is_auth_error(OAuthFlowError("expired")) is True


def test_call_tool_handler_returns_needs_reauth_on_unrecoverable_401(monkeypatch, tmp_path):
    """When session.call_tool raises 401 and handle_401 returns False,
    handler returns a structured needs_reauth error (not a generic failure)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from mcp.client.auth import OAuthFlowError

    reset_manager_for_tests()

    # Stub server
    server = MagicMock()
    server.name = "srv"
    session = MagicMock()

    async def _call_tool_raises(*a, **kw):
        raise OAuthFlowError("token expired")

    session.call_tool = _call_tool_raises
    server.session = session
    server._reconnect_event = MagicMock()
    server._ready = MagicMock()
    server._ready.is_set.return_value = True

    from tools import mcp_tool
    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)

    # Ensure the MCP loop exists (run_on_mcp_loop needs it)
    _mcp_loop._ensure_mcp_loop()

    # Force handle_401 to return False (no recovery available)
    mgr = get_manager()

    async def _h401(name, token=None):
        return False

    monkeypatch.setattr(mgr, "handle_401", _h401)

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({"arg": "v"})
        parsed = json.loads(result)
        assert parsed.get("needs_reauth") is True, f"expected needs_reauth, got: {parsed}"
        assert parsed.get("server") == "srv"
        assert "re-auth" in parsed.get("error", "").lower() or "reauth" in parsed.get("error", "").lower()
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)


def test_call_tool_handler_non_auth_error_still_generic(monkeypatch, tmp_path):
    """Non-auth exceptions still surface via the generic error path, not needs_reauth."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool_handlers import _make_tool_handler

    server = MagicMock()
    server.name = "srv"
    session = MagicMock()

    async def _raises(*a, **kw):
        raise RuntimeError("unrelated")

    session.call_tool = _raises
    server.session = session

    from tools import mcp_tool
    from tools import mcp_tool_loop as _mcp_loop
    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    _mcp_loop._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({"arg": "v"})
        parsed = json.loads(result)
        assert "needs_reauth" not in parsed
        assert "MCP call failed" in parsed.get("error", "")
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)


def test_anonymous_server_401_asks_for_sign_in_not_reauth(monkeypatch, tmp_path):
    """A server configured with only ``url`` (no ``auth: oauth``, no headers) that 401s on tools/call
    is the MCP runtime-auth shape: the model must be told to set up sign-in, not to "re-authenticate"
    credentials that never existed. A header-authenticated server is told its static credential was refused."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool_handlers import _handle_auth_error_and_retry
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from mcp.client.auth import OAuthFlowError
    from tools import mcp_tool

    reset_manager_for_tests()

    async def _h401(name, token=None):
        return False
    monkeypatch.setattr(get_manager(), "handle_401", _h401)
    _mcp_loop._ensure_mcp_loop()

    def _server(config):
        server = MagicMock()
        server.name = "anon"
        server._config = config
        server._auth_type = (config.get("auth") or "")
        return server

    try:
        mcp_tool._servers["anon"] = _server({"url": "https://hyper3d.example/mcp"})
        parsed = json.loads(_handle_auth_error_and_retry("anon", OAuthFlowError("401"), lambda: None, "call"))
        assert parsed["needs_reauth"] is True
        assert "anonymously" in parsed["error"] and "hermes mcp login anon" in parsed["error"]

        mcp_tool._servers["anon"] = _server({"url": "https://api.example/mcp", "headers": {"Authorization": "x"}})
        parsed = json.loads(_handle_auth_error_and_retry("anon", OAuthFlowError("401"), lambda: None, "call"))
        assert "anonymously" not in parsed["error"] and "static headers" in parsed["error"]
    finally:
        mcp_tool._servers.pop("anon", None)
        mcp_tool._server_error_counts.pop("anon", None)


def test_swallowed_401_from_transport_hook_is_treated_as_auth_error(monkeypatch, tmp_path):
    """mcp >= 2.0 Streamable HTTP folds a 401 on tools/call into a generic INTERNAL_ERROR MCPError. The owned
    httpx client's hook records the 401 on the server task; the auth recoverer consumes it exactly once."""
    import time
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool_handlers import _handle_auth_error_and_retry
    from tools.mcp_tool_errors import _make_unauthorized_recorder
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from mcp.shared.exceptions import MCPError
    from mcp.types import INTERNAL_ERROR
    from tools import mcp_tool

    reset_manager_for_tests()

    async def _h401(name, token=None):
        return False
    monkeypatch.setattr(get_manager(), "handle_401", _h401)
    _mcp_loop._ensure_mcp_loop()

    server = MagicMock()
    server.name = "anon"
    server._config = {"url": "https://hyper3d.example/mcp"}
    server._auth_type = ""
    server._unauthorized_at = 0.0
    generic = MCPError(INTERNAL_ERROR, "Server returned an error response")
    try:
        mcp_tool._servers["anon"] = server
        # No 401 observed on the wire: the generic error is NOT an auth error.
        assert _handle_auth_error_and_retry("anon", generic, lambda: None, "call") is None
        # The transport hook saw a 401 -> the same generic error becomes the sign-in error, once.
        asyncio.run(_make_unauthorized_recorder(server)(MagicMock(status_code=401)))
        assert server._unauthorized_at > 0 and time.monotonic() - server._unauthorized_at < 5
        parsed = json.loads(_handle_auth_error_and_retry("anon", generic, lambda: None, "call"))
        assert parsed["needs_reauth"] is True and "anonymously" in parsed["error"]
        assert server._unauthorized_at == 0.0
        assert _handle_auth_error_and_retry("anon", generic, lambda: None, "call") is None
    finally:
        mcp_tool._servers.pop("anon", None)
        mcp_tool._server_error_counts.pop("anon", None)
