"""Tests for the opt-in Photon reaction tool (``photon_react``).

The tool must only be *offered* when the operator enabled reactions
(``PHOTON_REACTIONS=true``) and a live Photon adapter exists in-process;
its handler must route to ``adapter.add_reaction`` / ``remove_reaction``
and default the chat to the Photon home channel when none is given.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import tools.photon_react_tool as prt


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("add", chat_id, emoji, message_id))
        return {"success": True, "message_id": message_id or "m1"}

    async def remove_reaction(self, chat_id, message_id=None):
        self.calls.append(("remove", chat_id, message_id))
        return {"success": True, "message_id": message_id or "m1"}


@pytest.fixture(autouse=True)
def _home_channel(monkeypatch):
    """Photon home channel resolves without touching real config files."""
    class _Home:
        chat_id = "any;-;+16265551234"

    class _Config:
        @staticmethod
        def get_home_channel(_platform):
            return _Home()

    # The handler imports load_gateway_config lazily; patch that path.
    import gateway.config as gw_config

    monkeypatch.setattr(gw_config, "load_gateway_config", lambda: _Config)


def test_check_fn_requires_env_flag(monkeypatch) -> None:
    monkeypatch.delenv("PHOTON_REACTIONS", raising=False)
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: _FakeAdapter())
    assert prt._photon_react_check() is False


def test_check_fn_requires_live_adapter(monkeypatch) -> None:
    monkeypatch.setenv("PHOTON_REACTIONS", "true")
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: None)
    assert prt._photon_react_check() is False


def test_check_fn_passes_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("PHOTON_REACTIONS", "true")
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: _FakeAdapter())
    assert prt._photon_react_check() is True


def test_handler_without_live_adapter_errors(monkeypatch) -> None:
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: None)
    result = json.loads(asyncio.run(prt._photon_react({"emoji": "🔥"})))
    assert "live Photon adapter" in result["error"]


def test_handler_adds_reaction_with_default_chat(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: fake)
    result = asyncio.run(prt._photon_react({"emoji": "🔥"}))
    assert result["success"] is True
    assert fake.calls == [("add", "any;-;+16265551234", "🔥", None)]


def test_handler_default_prefers_current_chat_over_home_channel(
    monkeypatch,
) -> None:
    """Omitting chat_id reacts in the conversation being answered, not the
    home DM — the contract the schema promises (review fix)."""
    fake = _FakeAdapter()
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: fake)
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "any;-;+15559999999")
    result = asyncio.run(prt._photon_react({"emoji": "👍"}))
    assert result["success"] is True
    assert fake.calls == [("add", "any;-;+15559999999", "👍", None)]


def test_handler_targets_specific_message(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: fake)
    result = asyncio.run(
        prt._photon_react(
            {"emoji": "😂", "chat_id": "any;-;+15550001111", "message_id": "abc"}
        )
    )
    assert result["success"] is True
    assert fake.calls == [("add", "any;-;+15550001111", "😂", "abc")]


def test_handler_remove_path(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: fake)
    result = asyncio.run(prt._photon_react({"emoji": "🔥", "remove": True}))
    assert result["success"] is True
    assert fake.calls == [("remove", "any;-;+16265551234", None)]


def test_handler_requires_emoji(monkeypatch) -> None:
    monkeypatch.setattr(prt, "_live_photon_adapter", lambda: _FakeAdapter())
    result = json.loads(asyncio.run(prt._photon_react({})))
    assert "error" in result
