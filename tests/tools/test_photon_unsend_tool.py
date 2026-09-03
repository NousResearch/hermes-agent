"""Tests for the Photon unsend tool (``photon_unsend``).

The tool must only be *offered* when a live Photon adapter exists in-process
(the check_fn); handler routes to ``adapter.unsend``, defaults the chat to
the Photon home channel when none given and the message to the most recent
message the bot sent in that chat, and returns a JSON string (the registry's
normal result shape).
"""
from __future__ import annotations

import asyncio
import json

import pytest

import tools.photon_unsend_tool as put
from gateway.platforms.base import SendResult


class _FakeAdapter:
    def __init__(self) -> None:
        self.unsent: list[tuple] = []
        self._last_sent_by_chat = {}

    async def unsend(self, chat_id, message_id):
        self.unsent.append((chat_id, message_id))
        return SendResult(success=True, message_id=message_id)

    def _normalize_chat_key(self, chat_id):
        return chat_id


@pytest.fixture(autouse=True)
def _home_channel(monkeypatch):
    """Photon home channel resolves without touching real config files."""

    class _Home:
        chat_id = "any;-;+162****1234"

    class _Config:
        @staticmethod
        def get_home_channel(_platform):
            return _Home()

    import gateway.config as gw_config

    monkeypatch.setattr(gw_config, "load_gateway_config", lambda: _Config)


def test_check_fn_requires_live_adapter(monkeypatch) -> None:
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: None)
    assert put._photon_unsend_check() is False


def test_check_fn_passes_with_adapter(monkeypatch) -> None:
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: _FakeAdapter())
    assert put._photon_unsend_check() is True


def test_handler_without_live_adapter_errors(monkeypatch) -> None:
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: None)
    result = json.loads(
        asyncio.run(put._photon_unsend({"message_id": "m1"}))
    )
    assert "live Photon adapter" in result["error"]


def test_requires_message_id_or_recent(monkeypatch) -> None:
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: _FakeAdapter())
    # No message_id and no recent sent message -> error
    result = json.loads(
        asyncio.run(put._photon_unsend({}))
    )
    assert "No message_id given and no recently-sent message" in result["error"]


def test_unsend_with_explicit_message_id(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(put._photon_unsend({"message_id": "target-msg-1"}))
    )
    assert result["success"] is True
    assert fake.unsent == [("any;-;+162****1234", "target-msg-1")]


def test_unsend_defaults_to_recent_sent(monkeypatch) -> None:
    fake = _FakeAdapter()
    fake._last_sent_by_chat = {"any;-;+162****1234": "recent-bot-msg"}
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(put._photon_unsend({}))
    )
    assert result["success"] is True
    assert fake.unsent == [("any;-;+162****1234", "recent-bot-msg")]


def test_unsend_prefers_current_chat(monkeypatch) -> None:
    fake = _FakeAdapter()
    fake._last_sent_by_chat = {"chat-current": "recent-in-current"}
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: fake)
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-current")
    asyncio.run(put._photon_unsend({}))
    assert fake.unsent[0][0] == "chat-current"


def test_adapter_failure_propagates_error(monkeypatch) -> None:
    class _BoomAdapter(_FakeAdapter):
        async def unsend(self, chat_id, message_id):
            return SendResult(success=False, error="not unsent: not our message")

    fake = _BoomAdapter()
    monkeypatch.setattr(put, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(put._photon_unsend({"message_id": "m1"}))
    )
    assert result["error"] == "not unsent: not our message"