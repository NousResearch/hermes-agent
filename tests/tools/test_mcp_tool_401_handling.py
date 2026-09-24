"""Tests for MCP tool-handler auth-failure detection.

When a tool call raises UnauthorizedError / OAuthNonInteractiveError /
httpx.HTTPStatusError(401), the handler should:
  1. Ask MCPOAuthManager.handle_401 if recovery is viable.
  2. If yes, trigger MCPServerTask._reconnect_event and retry once.
  3. If no, return a structured needs_reauth error so the model stops
     hallucinating manual refresh attempts.
"""
import json
import asyncio
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
        assert parsed.get("error")
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)


class _OpaqueMcpError(Exception):
    """The mcp 2.x shape after it folds a non-2xx tools/call response."""

    def __init__(self):
        super().__init__("Server returned an error response")
        self.error = type("Error", (), {"code": -32603})()


@pytest.mark.parametrize(("auth_type", "credential_kind"), [
    ("", "anonymous"), ("oauth", "OAuth"), ("", "static credentials"),
])
def test_call_tool_handler_uses_recent_tools_call_401_evidence_once(monkeypatch, tmp_path, auth_type, credential_kind):
    """An opaque mcp 2.x error inherits only its own just-recorded tools/call 401."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from tools import mcp_tool

    reset_manager_for_tests()
    server = MagicMock()
    server.name = "srv"
    server._auth_type = auth_type
    server._http_rejection = {}
    session = MagicMock()

    calls = 0

    async def _raises(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            server._http_rejection.update(status=401, rpc_method="tools/call",
                                          recorded_at=__import__("time").monotonic())
        raise _OpaqueMcpError()

    session.call_tool = _raises
    server.session = session
    server._reconnect_event = MagicMock()
    server._ready = MagicMock()
    server._ready.is_set.return_value = True
    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)
    _mcp_loop._ensure_mcp_loop()
    mgr = get_manager()

    async def _h401(name, token=None):
        return False

    monkeypatch.setattr(mgr, "handle_401", _h401)
    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        first = json.loads(handler({"arg": "v"}))
        assert first.get("needs_reauth") is True
        assert server._http_rejection == {}

        # The evidence is consumed: a later identical opaque error is not an auth error.
        second = json.loads(handler({"arg": "v"}))
        assert "needs_reauth" not in second
        assert second.get("error")
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)


def test_http_rejection_recorder_marks_only_tools_call_401():
    """The HTTP hook records the JSON-RPC method so other 401s cannot trigger call recovery."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_errors import _make_http_rejection_recorder

    httpx = sdk_httpx()

    async def _roundtrip(rpc_method):
        sink = {}
        transport = httpx.MockTransport(lambda request: httpx.Response(401, request=request))
        async with httpx.AsyncClient(transport=transport, event_hooks={
                "response": [_make_http_rejection_recorder(sink)]}) as client:
            await client.post("http://127.0.0.1:1/mcp", json={"method": rpc_method})
        return sink

    tools_call = asyncio.run(_roundtrip("tools/call"))
    assert tools_call["status"] == 401
    assert tools_call["rpc_method"] == "tools/call"
    assert isinstance(tools_call["recorded_at"], float)
    assert asyncio.run(_roundtrip("tools/list"))["rpc_method"] == "tools/list"
