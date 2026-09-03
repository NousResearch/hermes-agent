"""Tests for the Photon native-message tools (``photon_poll`` / ``photon_effect``).

Both tools must only be *offered* when a live Photon adapter exists in-process
(the check_fn); handlers must route to ``adapter.send_poll`` /
``adapter.send_effect``, default the chat to the Photon home channel when none
is given, and return JSON strings (the registry's normal result shape).
"""
from __future__ import annotations

import asyncio
import json

import pytest

import tools.photon_poll_tool as ppt
import tools.photon_effect_tool as pet
from gateway.platforms.base import SendResult


class _FakeAdapter:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    async def send_poll(self, chat_id, title, options, metadata=None):
        self.sent.append(("poll", chat_id, title, options))
        return SendResult(success=True, message_id="m1")

    async def send_effect(self, chat_id, text, effect, metadata=None):
        self.sent.append(("effect", chat_id, text, effect))
        return SendResult(success=True, message_id="m1")


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
    monkeypatch.setattr(ppt, "_live_photon_adapter", lambda: None)
    assert ppt._photon_poll_check() is False
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: None)
    assert pet._photon_effect_check() is False


def test_check_fn_passes_with_adapter(monkeypatch) -> None:
    monkeypatch.setattr(ppt, "_live_photon_adapter", lambda: _FakeAdapter())
    assert ppt._photon_poll_check() is True
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: _FakeAdapter())
    assert pet._photon_effect_check() is True


def test_handler_without_live_adapter_errors(monkeypatch) -> None:
    monkeypatch.setattr(ppt, "_live_photon_adapter", lambda: None)
    result = json.loads(
        asyncio.run(ppt._photon_poll({"title": "t", "options": ["a", "b"]}))
    )
    assert "live Photon adapter" in result["error"]


def test_poll_requires_title_and_options(monkeypatch) -> None:
    monkeypatch.setattr(ppt, "_live_photon_adapter", lambda: _FakeAdapter())
    assert json.loads(asyncio.run(ppt._photon_poll({"options": ["a", "b"]})))[
        "error"
    ]
    assert json.loads(
        asyncio.run(ppt._photon_poll({"title": "t", "options": ["a"]}))
    )["error"]


def test_poll_sends_with_default_chat(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(ppt, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(ppt._photon_poll({"title": "pick", "options": ["a", "b"]}))
    )
    assert result["success"] is True
    assert fake.sent == [("poll", "any;-;+162****1234", "pick", ["a", "b"])]


def test_poll_default_prefers_current_chat(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(ppt, "_live_photon_adapter", lambda: fake)
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-of-current-msg")
    asyncio.run(ppt._photon_poll({"title": "t", "options": ["a", "b"]}))
    assert fake.sent[0][1] == "chat-of-current-msg"


def test_effect_requires_text_and_effect(monkeypatch) -> None:
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: _FakeAdapter())
    assert (
        json.loads(asyncio.run(pet._photon_effect({"text": "hi"})))["error"]
        or json.loads(asyncio.run(pet._photon_effect({"effect": "slam"})))["error"]
    )


def test_effect_rejects_unknown_effect(monkeypatch) -> None:
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: _FakeAdapter())
    result = json.loads(
        asyncio.run(pet._photon_effect({"text": "hi", "effect": "nope"}))
    )
    assert "Unknown effect" in result["error"]


def test_effect_sends_with_default_chat(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(pet._photon_effect({"text": "boom", "effect": "slam"}))
    )
    assert result["success"] is True
    assert result["effect"] == "slam"
    assert fake.sent[0] == ("effect", "any;-;+162****1234", "boom", "slam")


def test_adapter_failure_propagates_error(monkeypatch) -> None:
    class _BoomAdapter(_FakeAdapter):
        async def send_effect(self, chat_id, text, effect, metadata=None):
            return SendResult(success=False, error="sidecar down")

    fake = _BoomAdapter()
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(pet._photon_effect({"text": "boom", "effect": "slam"}))
    )
    assert result["error"] == "sidecar down"