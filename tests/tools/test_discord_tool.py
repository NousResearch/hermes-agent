from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import tools.discord_tool as discord_tool


class FakeResponse:
    def __init__(self, status: int, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload)


class FakeClientSession:
    def __init__(self, responses, recorder, headers=None):
        self._responses = list(responses)
        self._recorder = recorder
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, json=None):
        self._recorder.append({"method": method, "url": url, "json": json, "headers": dict(self.headers)})
        if not self._responses:
            raise AssertionError(f"No fake response left for {method} {url}")
        status, payload = self._responses.pop(0)
        return FakeResponse(status, payload)


class FakeAioHttpModule:
    def __init__(self, responses, recorder):
        self._responses = responses
        self._recorder = recorder

    def ClientSession(self, headers=None):
        return FakeClientSession(self._responses, self._recorder, headers=headers)


@pytest.fixture
def fake_runner(monkeypatch):
    monkeypatch.setattr(discord_tool, "_load_discord_token", lambda: "token-123")

    def _run(coro):
        import asyncio
        return asyncio.run(coro)

    monkeypatch.setitem(sys.modules, "model_tools", SimpleNamespace(_run_async=_run))


def test_resolve_target_uses_current_discord_chat(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123456")

    target_id, note = discord_tool._resolve_target_chat_id(None)

    assert target_id == "123456"
    assert "current Discord chat" in note


@pytest.mark.asyncio
async def test_create_thread_uses_parent_channel_when_origin_is_thread(monkeypatch):
    calls = []
    fake_aiohttp = FakeAioHttpModule(
        responses=[
            (200, {"id": "777", "type": 11, "parent_id": "555"}),
            (201, {"id": "999", "name": "Spec review"}),
        ],
        recorder=calls,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "777")
    monkeypatch.setattr(discord_tool, "_load_discord_token", lambda: "token-123")

    result = await discord_tool._create_thread({"name": "Spec review"})

    assert result["success"] is True
    assert result["thread_id"] == "999"
    assert result["parent_channel_id"] == "555"
    assert result["creation_mode"] == "without_message"
    assert calls[1]["url"].endswith("/channels/555/threads")


@pytest.mark.asyncio
async def test_create_thread_falls_back_to_message_seed_when_direct_create_fails(monkeypatch):
    calls = []
    fake_aiohttp = FakeAioHttpModule(
        responses=[
            (200, {"id": "123", "type": 0}),
            (400, {"message": "Cannot create thread that way"}),
            (200, {"id": "seed-1"}),
            (201, {"id": "thread-2", "name": "Implementation"}),
        ],
        recorder=calls,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setattr(discord_tool, "_load_discord_token", lambda: "token-123")
    monkeypatch.setattr(discord_tool, "_resolve_target_chat_id", lambda target: ("123", "Used Discord target 123."))

    result = await discord_tool._create_thread({"name": "Implementation"})

    assert result["success"] is True
    assert result["creation_mode"] == "from_message"
    assert result["starter_message_id"] == "seed-1"
    assert calls[2]["url"].endswith("/channels/123/messages")
    assert calls[3]["url"].endswith("/channels/123/messages/seed-1/threads")


@pytest.mark.asyncio
async def test_create_thread_posts_opening_message_when_direct_create_succeeds(monkeypatch):
    calls = []
    fake_aiohttp = FakeAioHttpModule(
        responses=[
            (200, {"id": "123", "type": 0}),
            (201, {"id": "thread-9", "name": "Bugs"}),
            (200, {"id": "msg-9"}),
        ],
        recorder=calls,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setattr(discord_tool, "_load_discord_token", lambda: "token-123")
    monkeypatch.setattr(discord_tool, "_resolve_target_chat_id", lambda target: ("123", "Used Discord target 123."))

    result = await discord_tool._create_thread({"name": "Bugs", "message": "Track rough edges here."})

    assert result["success"] is True
    assert result["starter_message_id"] == "msg-9"
    assert calls[2]["url"].endswith("/channels/thread-9/messages")
    assert calls[2]["json"] == {"content": "Track rough edges here."}


@pytest.mark.asyncio
async def test_create_channel_uses_same_category_as_current_channel(monkeypatch):
    calls = []
    fake_aiohttp = FakeAioHttpModule(
        responses=[
            (200, {"id": "123", "type": 0, "guild_id": "guild-1", "parent_id": "cat-9"}),
            (201, {"id": "chan-2", "name": "planning-room", "topic": "Roadmap", "nsfw": False}),
        ],
        recorder=calls,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setattr(discord_tool, "_load_discord_token", lambda: "token-123")
    monkeypatch.setattr(discord_tool, "_resolve_target_chat_id", lambda target: ("123", "Used Discord target 123."))

    result = await discord_tool._create_channel({"name": "planning-room", "topic": "Roadmap"})

    assert result["success"] is True
    assert result["channel_id"] == "chan-2"
    assert result["guild_id"] == "guild-1"
    assert result["parent_category_id"] == "cat-9"
    assert calls[1]["url"].endswith("/guilds/guild-1/channels")
    assert calls[1]["json"] == {"name": "planning-room", "type": 0, "nsfw": False, "parent_id": "cat-9", "topic": "Roadmap"}


@pytest.mark.asyncio
async def test_create_channel_from_thread_uses_parent_channel_category(monkeypatch):
    calls = []
    fake_aiohttp = FakeAioHttpModule(
        responses=[
            (200, {"id": "thread-7", "type": 11, "parent_id": "123", "guild_id": "guild-1"}),
            (200, {"id": "123", "type": 0, "guild_id": "guild-1", "parent_id": "cat-9"}),
            (201, {"id": "chan-8", "name": "bugs"}),
        ],
        recorder=calls,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setattr(discord_tool, "_load_discord_token", lambda: "token-123")
    monkeypatch.setattr(discord_tool, "_resolve_target_chat_id", lambda target: ("thread-7", "Used Discord target thread-7."))

    result = await discord_tool._create_channel({"name": "bugs"})

    assert result["success"] is True
    assert result["channel_id"] == "chan-8"
    assert result["parent_category_id"] == "cat-9"
    assert calls[1]["url"].endswith("/channels/123")
    assert calls[2]["url"].endswith("/guilds/guild-1/channels")


@pytest.mark.asyncio
async def test_create_channel_rejects_dm_targets(monkeypatch):
    calls = []
    fake_aiohttp = FakeAioHttpModule(
        responses=[
            (200, {"id": "dm-1", "type": 1}),
        ],
        recorder=calls,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setattr(discord_tool, "_load_discord_token", lambda: "token-123")
    monkeypatch.setattr(discord_tool, "_resolve_target_chat_id", lambda target: ("dm-1", "Used Discord target dm-1."))

    result = await discord_tool._create_channel({"name": "private-lair"})

    assert result["error"] == "Discord channels can only be created inside servers, not DMs."


def test_discord_manage_tool_serializes_create_thread(fake_runner, monkeypatch):
    monkeypatch.setattr(
        discord_tool,
        "_create_thread",
        lambda args: _immediate_result({"success": True, "thread_id": "42", "thread_name": args["name"]}),
    )

    result = json.loads(discord_tool.discord_manage_tool({"action": "create_thread", "name": "Planning"}))

    assert result["success"] is True
    assert result["thread_id"] == "42"
    assert result["thread_name"] == "Planning"


def test_discord_manage_tool_serializes_create_channel(fake_runner, monkeypatch):
    monkeypatch.setattr(
        discord_tool,
        "_create_channel",
        lambda args: _immediate_result({"success": True, "channel_id": "84", "channel_name": args["name"]}),
    )

    result = json.loads(discord_tool.discord_manage_tool({"action": "create_channel", "name": "planning-room"}))

    assert result["success"] is True
    assert result["channel_id"] == "84"
    assert result["channel_name"] == "planning-room"


async def _immediate_result(value):
    return value
