"""Tests for send_message action='react'/'unreact' dispatch.

Kept separate from ``test_send_message_tool.py`` because that module skips
wholesale when optional Telegram dependencies are not installed.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import AsyncMock, MagicMock

import tools.send_message_tool as smt


class _FakePhotonAdapter:
    """Adapter exposing add_reaction/remove_reaction coroutines."""

    def __init__(self):
        self.calls = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("add", chat_id, emoji, message_id))
        return {"success": True, "emoji": emoji}

    async def remove_reaction(self, chat_id, message_id=None):
        self.calls.append(("remove", chat_id, message_id))
        return {"success": True}


class _NoReactionAdapter:
    """Adapter with no reaction support at all."""


def _runner_with(adapter):
    from gateway.config import Platform

    return SimpleNamespace(adapters={Platform("photon"): adapter})


def _call(args):
    return json.loads(smt.send_message_tool(args))


def test_react_dispatches_to_add_reaction():
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "❤️"}
        )
    assert result["success"] is True
    assert adapter.calls == [("add", "+15551234567", "❤️", None)]


def test_react_without_live_gateway():
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "👍"}
        )
    assert result.get("success") is not True
    assert "live" in json.dumps(result)


def test_whatsapp_reaction_reaches_bridge_with_native_message_id():
    from gateway.config import Platform
    from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter

    adapter = _make_adapter()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    runner = SimpleNamespace(adapters={Platform.WHATSAPP: adapter})
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        added = _call({"action": "react", "target": "whatsapp:+15551234567", "message_id": "msg-in", "emoji": "👍"})
        removed = _call({"action": "unreact", "target": "whatsapp:+15551234567", "message_id": "msg-in"})
    assert added["success"] is True
    assert removed["success"] is True
    calls = adapter._http_session.post.call_args_list
    assert [call.args[0] for call in calls] == ["http://127.0.0.1:3000/reaction"] * 2
    assert [call.kwargs["json"] for call in calls] == [
        {"chatId": "15551234567@s.whatsapp.net", "messageId": "msg-in", "emoji": "👍"},
        {"chatId": "15551234567@s.whatsapp.net", "messageId": "msg-in", "emoji": ""},
    ]


def test_whatsapp_reaction_reports_bridge_target_rejection():
    from gateway.config import Platform
    from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter

    adapter = _make_adapter()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    resp = MagicMock(status=404)
    resp.text = AsyncMock(return_value='{"error":"Message not found in this chat"}')
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    runner = SimpleNamespace(adapters={Platform.WHATSAPP: adapter})
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        result = _call({"action": "react", "target": "whatsapp:15551234567", "message_id": "wrong-chat", "emoji": "👍"})
    assert result["success"] is False
    assert "Message not found in this chat" in result["error"]


def test_whatsapp_reaction_requires_message_id_before_bridge_send():
    from gateway.config import Platform
    from tests.gateway.test_whatsapp_formatting import _make_adapter

    adapter = _make_adapter()
    runner = SimpleNamespace(adapters={Platform.WHATSAPP: adapter})
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        result = _call({"action": "react", "target": "whatsapp:+15551234567", "emoji": "👍"})
    assert result["success"] is False
    assert "message_id" in result["error"]
    adapter._http_session.post.assert_not_called()
