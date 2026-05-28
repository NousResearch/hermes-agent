import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.client.auth.oauth2")


def _install_stub_server(mcp_tool_module, name: str, call_tool_impl):
    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl
    server.session = session
    server._reconnect_event = MagicMock()
    server._ready = MagicMock()
    server._ready.is_set.return_value = True
    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup(mcp_tool_module, name: str) -> None:
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)


def test_mcp_tool_handler_passes_current_hermes_session_meta(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_API_BASE_URL", "http://gateway:8642")
    monkeypatch.setenv("HERMES_GATEWAY_API_KEY", "secret")

    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    captured = {}

    async def _call_tool_success(*a, **kw):
        captured.update(kw)
        result = MagicMock()
        result.isError = False
        block = MagicMock()
        block.text = "ok"
        result.content = [block]
        result.structuredContent = None
        return result

    tokens = set_session_vars(
        platform="matrix",
        chat_id="!room:lumeny.io",
        thread_id="$thread",
        user_id="@user:lumeny.io",
        user_name="lumenyang",
        session_key="agent:main:matrix:group:!room:lumeny.io:$thread",
        message_id="$msg",
    )
    _install_stub_server(mcp_tool, "session-meta-srv", _call_tool_success)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("session-meta-srv", "tool1", 10.0)
        parsed = json.loads(handler({"x": 1}))
        assert parsed.get("result") == "ok", parsed
        assert captured["arguments"] == {"x": 1}
        meta = captured["meta"]["hermes_session"]
        assert meta["platform"] == "matrix"
        assert meta["chat_id"] == "!room:lumeny.io"
        assert meta["thread_id"] == "$thread"
        assert meta["session_key"] == "agent:main:matrix:group:!room:lumeny.io:$thread"
        assert meta["message_id"] == "$msg"
        assert meta["api_base_url"] == "http://gateway:8642"
        assert meta["api_key"] == "secret"
    finally:
        _cleanup(mcp_tool, "session-meta-srv")
        clear_session_vars(tokens)
