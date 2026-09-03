"""Tests for MCP tool-handler auth-failure detection.

When a tool call raises UnauthorizedError / OAuthNonInteractiveError /
httpx.HTTPStatusError(401), the handler should:
  1. Ask MCPOAuthManager.handle_401 if recovery is viable.
  2. If yes, trigger MCPServerTask._reconnect_event and retry once.
  3. If no, return a structured needs_reauth error so the model stops
     hallucinating manual refresh attempts.
"""
import json
import threading
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


def test_is_auth_error_detects_oauth_flow_error():
    from tools.mcp_tool import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    assert _is_auth_error(OAuthFlowError("expired")) is True


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


def test_call_tool_handler_returns_app_error_after_auth_recovery(
    monkeypatch, tmp_path
):
    """A completed post-auth retry returns its application error as-is.

    The response proves the recovered transport is reachable, so it must not
    be replaced with a misleading ``needs_reauth`` error or leave the circuit
    breaker partially tripped.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from mcp.client.auth import OAuthFlowError
    from tools import mcp_tool
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests

    reset_manager_for_tests()
    mcp_tool._ensure_mcp_loop()

    server = MagicMock()
    server.name = "srv-auth-app-error"
    ready_flag = threading.Event()
    ready_flag.set()

    class _ReadyAdapter:
        def is_set(self):
            return ready_flag.is_set()

        def clear(self):
            ready_flag.clear()

        def set(self):
            ready_flag.set()

    call_count = {"n": 0}

    async def _call_sequence(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OAuthFlowError("token expired")
        result = MagicMock()
        result.isError = True
        result.content = [MagicMock(type="text", text="Prompt not found")]
        result.structuredContent = None
        return result

    old_session = MagicMock()
    old_session.call_tool = _call_sequence
    server.session = old_session
    server._ready = _ReadyAdapter()

    class _ReconnectAdapter:
        def set(self):
            new_session = MagicMock()
            new_session.call_tool = _call_sequence
            server.session = new_session
            ready_flag.set()

    server._reconnect_event = _ReconnectAdapter()
    mcp_tool._servers[server.name] = server
    mcp_tool._server_error_counts[server.name] = (
        mcp_tool._CIRCUIT_BREAKER_THRESHOLD - 1
    )

    manager = get_manager()

    async def _h401(name, token=None):
        return True

    monkeypatch.setattr(manager, "handle_401", _h401)

    try:
        handler = mcp_tool._make_tool_handler(server.name, "tool1", 10.0)
        parsed = json.loads(handler({"prompt_id": "stale"}))

        assert parsed == {"error": "Prompt not found"}
        assert "needs_reauth" not in parsed
        assert call_count["n"] == 2
        assert mcp_tool._server_error_counts.get(server.name, 0) == 0
    finally:
        mcp_tool._servers.pop(server.name, None)
        mcp_tool._server_error_counts.pop(server.name, None)
        mcp_tool._server_breaker_opened_at.pop(server.name, None)
        reset_manager_for_tests()


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
