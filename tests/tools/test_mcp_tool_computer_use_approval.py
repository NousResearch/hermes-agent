import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tools import mcp_tool


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_computer_use_handler_requests_app_access_approval_and_retries(monkeypatch):
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=[
        SimpleNamespace(isError=False, content=[], structuredContent={
            "approval_required": True,
            "approved": False,
            "app_name": "TextEdit",
            "app_session_id": "app-1",
        }),
        SimpleNamespace(isError=False, content=[], structuredContent={"success": True}),
        SimpleNamespace(isError=False, content=[], structuredContent={
            "success": True,
            "approved": True,
            "app_name": "TextEdit",
            "app_session_id": "app-1",
        }),
    ])
    server = SimpleNamespace(session=session)
    monkeypatch.setattr(mcp_tool, "_servers", {"hermes-computer-use": server})
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", lambda coro, timeout=30: _run(coro))
    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": "sess-1")
    monkeypatch.setattr("tools.approval.request_gateway_choice", lambda *args, **kwargs: {
        "resolved": True,
        "choice": "session",
        "request_id": "gar_test",
    })
    mcp_tool._COMPUTER_USE_SESSION_APPROVALS.clear()

    handler = mcp_tool._make_tool_handler("hermes-computer-use", "get_app_state", 30)
    result = json.loads(handler({"app_name": "TextEdit"}))

    assert result["result"]["success"] is True
    assert result["result"]["approved"] is True
    assert mcp_tool._COMPUTER_USE_SESSION_APPROVALS["sess-1"] == {"TextEdit"}
    calls = [call.args[0] for call in session.call_tool.await_args_list]
    assert calls == ["get_app_state", "grant_temporary_app_approval", "get_app_state"]
