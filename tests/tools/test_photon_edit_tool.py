"""Tests for the Photon edit tool (``photon_edit``).

The tool must only be *offered* when a live Photon adapter exists in-process
(the check_fn); handler routes to ``adapter.edit_message``, defaults the chat
to the Photon home channel when none given, and returns a JSON string (the
registry's normal result shape).
"""
from __future__ import annotations

import asyncio
import json

import pytest

import tools.photon_edit_tool as pet
from gateway.platforms.base import SendResult


class _FakeAdapter:
    def __init__(self) -> None:
        self.edited: list[tuple] = []

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edited.append((chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)


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
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: None)
    assert pet._photon_edit_check() is False


def test_check_fn_passes_with_adapter(monkeypatch) -> None:
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: _FakeAdapter())
    assert pet._photon_edit_check() is True


def test_handler_without_live_adapter_errors(monkeypatch) -> None:
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: None)
    result = json.loads(
        asyncio.run(pet._photon_edit({"message_id": "m1", "text": "hi"}))
    )
    assert "live Photon adapter" in result["error"]


def test_requires_message_id_and_text(monkeypatch) -> None:
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: _FakeAdapter())
    assert json.loads(
        asyncio.run(pet._photon_edit({"text": "hi"}))
    )["error"] == "Both 'message_id' and 'text' are required."
    assert json.loads(
        asyncio.run(pet._photon_edit({"message_id": "m1"}))
    )["error"] == "Both 'message_id' and 'text' are required."


def test_edit_with_explicit_chat(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(
            pet._photon_edit(
                {"message_id": "m1", "text": "corrected", "chat_id": "my-chat"}
            )
        )
    )
    assert result["success"] is True
    assert fake.edited == [("my-chat", "m1", "corrected")]


def test_edit_defaults_to_home_channel(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(pet._photon_edit({"message_id": "m1", "text": "hi"}))
    )
    assert result["success"] is True
    assert fake.edited[0][0] == "any;-;+162****1234"


def test_edit_prefers_current_chat(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: fake)
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-current")
    asyncio.run(pet._photon_edit({"message_id": "m1", "text": "hi"}))
    assert fake.edited[0][0] == "chat-current"


def test_adapter_failure_propagates_error(monkeypatch) -> None:
    class _BoomAdapter(_FakeAdapter):
        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            return SendResult(success=False, error="edit failed: window closed")

    fake = _BoomAdapter()
    monkeypatch.setattr(pet, "_live_photon_adapter", lambda: fake)
    result = json.loads(
        asyncio.run(pet._photon_edit({"message_id": "m1", "text": "hi"}))
    )
    assert result["error"] == "edit failed: window closed"