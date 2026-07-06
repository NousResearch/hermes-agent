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


# --------------------------------------------------------------------------
# WhatsApp adapter add_reaction/remove_reaction exercised end-to-end through
# the generic send_message(action='react') path, with the bridge HTTP call
# mocked (no live network, no live gateway process).
# --------------------------------------------------------------------------


class _FakeBridgeResponse:
    """Async-context-manager mimicking aiohttp's POST response."""

    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload if payload is not None else {"success": True}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)


class _FakeBridgeSession:
    """Records POSTs and returns a canned bridge response."""

    def __init__(self, status=200):
        self._status = status
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        return _FakeBridgeResponse(status=self._status)


def _make_whatsapp_adapter(session):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = True
    adapter._http_session = session
    adapter._bridge_port = 3000
    adapter._bridge_process = None  # no managed child → exit-check is a no-op
    return adapter


def _whatsapp_runner_with(adapter):
    from gateway.config import Platform

    return SimpleNamespace(adapters={Platform("whatsapp"): adapter})


def test_whatsapp_react_posts_to_bridge_react_endpoint():
    session = _FakeBridgeSession()
    adapter = _make_whatsapp_adapter(session)
    with patch("gateway.run._gateway_runner_ref", lambda: _whatsapp_runner_with(adapter)):
        result = _call(
            {
                "action": "react",
                "target": "whatsapp:1234567890@s.whatsapp.net",
                "emoji": "❤️",
                "message_id": "wamid-1",
            }
        )
    assert result["success"] is True
    assert len(session.posts) == 1
    post = session.posts[0]
    assert post["url"].endswith("/react")
    assert post["json"] == {
        "chatId": "1234567890@s.whatsapp.net",
        "emoji": "❤️",
        "messageId": "wamid-1",
    }


def test_whatsapp_react_without_message_id_omits_it():
    session = _FakeBridgeSession()
    adapter = _make_whatsapp_adapter(session)
    with patch("gateway.run._gateway_runner_ref", lambda: _whatsapp_runner_with(adapter)):
        result = _call(
            {
                "action": "react",
                "target": "whatsapp:1234567890@s.whatsapp.net",
                "emoji": "👍",
            }
        )
    assert result["success"] is True
    # message_id omitted so the bridge resolves the most recent message itself.
    assert session.posts[0]["json"] == {
        "chatId": "1234567890@s.whatsapp.net",
        "emoji": "👍",
    }


def test_whatsapp_unreact_sends_empty_emoji():
    session = _FakeBridgeSession()
    adapter = _make_whatsapp_adapter(session)
    with patch("gateway.run._gateway_runner_ref", lambda: _whatsapp_runner_with(adapter)):
        result = _call(
            {
                "action": "unreact",
                "target": "whatsapp:1234567890@s.whatsapp.net",
                "message_id": "wamid-9",
            }
        )
    assert result["success"] is True
    assert session.posts[0]["url"].endswith("/react")
    assert session.posts[0]["json"] == {
        "chatId": "1234567890@s.whatsapp.net",
        "emoji": "",
        "messageId": "wamid-9",
    }


def test_whatsapp_react_bridge_rejection_reports_failure():
    session = _FakeBridgeSession(status=500)
    adapter = _make_whatsapp_adapter(session)
    with patch("gateway.run._gateway_runner_ref", lambda: _whatsapp_runner_with(adapter)):
        result = _call(
            {
                "action": "react",
                "target": "whatsapp:1234567890@s.whatsapp.net",
                "emoji": "👍",
            }
        )
    # add_reaction returns False on a non-200 bridge response.
    assert result["success"] is False


def test_whatsapp_react_not_connected_reports_failure():
    session = _FakeBridgeSession()
    adapter = _make_whatsapp_adapter(session)
    adapter._running = False  # bridge not connected
    with patch("gateway.run._gateway_runner_ref", lambda: _whatsapp_runner_with(adapter)):
        result = _call(
            {
                "action": "react",
                "target": "whatsapp:1234567890@s.whatsapp.net",
                "emoji": "👍",
            }
        )
    assert result["success"] is False
    assert session.posts == []
