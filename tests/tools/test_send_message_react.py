"""Tests for send_message action='react'/'unreact' dispatch.

Kept separate from ``test_send_message_tool.py`` because that module skips
wholesale when optional Telegram dependencies are not installed.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

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


def test_unreact_dispatches_to_remove_reaction():
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {
                "action": "unreact",
                "target": "photon:+15551234567",
                "message_id": "msg-9",
            }
        )
    assert result["success"] is True
    assert adapter.calls == [("remove", "+15551234567", "msg-9")]


def test_react_requires_emoji():
    result = _call({"action": "react", "target": "photon:+15551234567"})
    assert result.get("success") is not True
    assert "emoji" in json.dumps(result)


def test_unreact_does_not_require_emoji():
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call({"action": "unreact", "target": "photon:+15551234567"})
    assert result["success"] is True
    assert adapter.calls == [("remove", "+15551234567", None)]


def test_react_unsupported_platform_adapter():
    adapter = _NoReactionAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "👍"}
        )
    assert result.get("success") is not True
    assert "does not support" in json.dumps(result)


def test_react_without_live_gateway():
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "👍"}
        )
    assert result.get("success") is not True
    assert "live" in json.dumps(result)


# ---------------------------------------------------------------------------
# WhatsApp
#
# Exercised through the real adapter (not a fake) so the tool's keyword-arg
# call convention stays pinned to the adapter's actual signature.
# ---------------------------------------------------------------------------

def _whatsapp_runner():
    from gateway.config import Platform
    from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter
    from unittest.mock import AsyncMock, MagicMock

    adapter = _make_adapter()
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True})
    resp.text = AsyncMock(return_value="")
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    return SimpleNamespace(adapters={Platform.WHATSAPP: adapter}), adapter


def test_react_dispatches_to_whatsapp_adapter():
    runner, adapter = _whatsapp_runner()
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        result = _call(
            {
                "action": "react",
                "target": "whatsapp:15551234567",
                "emoji": "👍",
                "message_id": "msg-1",
            }
        )
    assert result["success"] is True
    call = adapter._http_session.post.call_args
    assert call.args[0].endswith("/react")
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "messageId": "msg-1",
        "emoji": "👍",
        "fromMe": False,
    }


def test_unreact_dispatches_to_whatsapp_adapter_with_empty_emoji():
    runner, adapter = _whatsapp_runner()
    adapter._record_last_inbound("15551234567@s.whatsapp.net", "inbound-9")
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        result = _call({"action": "unreact", "target": "whatsapp:15551234567"})
    assert result["success"] is True
    assert adapter._http_session.post.call_args.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "messageId": "inbound-9",
        "emoji": "",
        "fromMe": False,
    }
