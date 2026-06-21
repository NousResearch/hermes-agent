import json
from types import SimpleNamespace


def test_tool_handler_reconnects_stale_wrapper_before_returning_not_connected(monkeypatch):
    from tools import mcp_tool

    calls = []
    fake_server = SimpleNamespace(session=object())

    with mcp_tool._lock:
        mcp_tool._servers.pop("control-tower", None)
        mcp_tool._server_error_counts.pop("control-tower", None)

    def fake_reconnect(server_name):
        calls.append(server_name)
        return fake_server

    monkeypatch.setattr(mcp_tool, "_try_reconnect_disconnected_server", fake_reconnect)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", lambda _call, timeout: json.dumps({"result": "ok"}))

    handler = mcp_tool._make_tool_handler("control-tower", "get_capabilities", 1)
    result = json.loads(handler({}))

    assert calls == ["control-tower"]
    assert result == {"result": "ok"}


def test_tool_handler_reports_not_connected_when_reconnect_fails(monkeypatch):
    from tools import mcp_tool

    with mcp_tool._lock:
        mcp_tool._servers.pop("control-tower", None)
        mcp_tool._server_error_counts.pop("control-tower", None)

    monkeypatch.setattr(mcp_tool, "_try_reconnect_disconnected_server", lambda _name: None)

    handler = mcp_tool._make_tool_handler("control-tower", "get_capabilities", 1)
    result = json.loads(handler({}))

    assert result == {"error": "MCP server 'control-tower' is not connected"}
    assert mcp_tool._server_error_counts.get("control-tower") == 1
