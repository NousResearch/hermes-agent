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


def test_react_without_a_chat_uses_the_current_turn(monkeypatch):
    """A bare platform target means "here", not the home channel."""
    adapter = _FakePhotonAdapter()
    monkeypatch.setattr("gateway.session_context.get_session_env", lambda k, d="": {
        "HERMES_SESSION_PLATFORM": "photon",
        "HERMES_SESSION_CHAT_ID": "+15559998888",
    }.get(k, d))
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call({"action": "react", "target": "photon", "emoji": "🟢"})
    assert result["success"] is True
    assert adapter.calls == [("add", "+15559998888", "🟢", None)]


def test_react_without_a_chat_ignores_a_different_platforms_turn(monkeypatch):
    """The current chat only applies to the platform the turn is running on."""
    adapter = _FakePhotonAdapter()
    monkeypatch.setattr("gateway.session_context.get_session_env", lambda k, d="": {
        "HERMES_SESSION_PLATFORM": "discord",
        "HERMES_SESSION_CHAT_ID": "999",
    }.get(k, d))
    home = SimpleNamespace(chat_id="+15551110000")
    config = SimpleNamespace(get_home_channel=lambda _p: home)
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)), \
            patch("gateway.config.load_gateway_config", lambda: config):
        result = _call({"action": "react", "target": "photon", "emoji": "🟢"})
    assert result["success"] is True
    assert adapter.calls == [("add", "+15551110000", "🟢", None)]


def test_react_without_live_gateway():
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "👍"}
        )
    assert result.get("success") is not True
    assert "live" in json.dumps(result)
