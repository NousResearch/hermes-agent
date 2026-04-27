"""Tests for MCP tool-handler auth-failure detection.

When a tool call raises UnauthorizedError / OAuthNonInteractiveError /
httpx.HTTPStatusError(401), the handler should:
  1. Ask MCPOAuthManager.handle_401 if recovery is viable.
  2. If yes, trigger MCPServerTask._reconnect_event and retry once.
  3. If no, return a structured needs_reauth error so the model stops
     hallucinating manual refresh attempts.
"""
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


def test_is_auth_error_detects_oauth_flow_error():
    from tools.mcp_tool import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    assert _is_auth_error(OAuthFlowError("expired")) is True


def test_is_auth_error_detects_oauth_non_interactive():
    from tools.mcp_tool import _is_auth_error
    from tools.mcp_oauth import OAuthNonInteractiveError

    assert _is_auth_error(OAuthNonInteractiveError("no browser")) is True


def test_is_auth_error_detects_httpx_401():
    from tools.mcp_tool import _is_auth_error
    import httpx

    response = MagicMock()
    response.status_code = 401
    exc = httpx.HTTPStatusError("unauth", request=MagicMock(), response=response)
    assert _is_auth_error(exc) is True


def test_is_auth_error_rejects_httpx_500():
    from tools.mcp_tool import _is_auth_error
    import httpx

    response = MagicMock()
    response.status_code = 500
    exc = httpx.HTTPStatusError("oops", request=MagicMock(), response=response)
    assert _is_auth_error(exc) is False


def test_is_auth_error_rejects_generic_exception():
    from tools.mcp_tool import _is_auth_error
    assert _is_auth_error(ValueError("not auth")) is False
    assert _is_auth_error(RuntimeError("not auth")) is False


def test_call_tool_handler_returns_needs_reauth_on_unrecoverable_401(monkeypatch, tmp_path):
    """When session.call_tool raises 401 and handle_401 returns False,
    handler returns a structured needs_reauth error (not a generic failure)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_tool import _make_tool_handler
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
    mcp_tool._ensure_mcp_loop()

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
    from tools.mcp_tool import _make_tool_handler

    server = MagicMock()
    server.name = "srv"
    session = MagicMock()

    async def _raises(*a, **kw):
        raise RuntimeError("unrelated")

    session.call_tool = _raises
    server.session = session

    from tools import mcp_tool
    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({"arg": "v"})
        parsed = json.loads(result)
        assert "needs_reauth" not in parsed
        assert "MCP call failed" in parsed.get("error", "")
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)
