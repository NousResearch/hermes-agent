"""Tests for the Discord read-only tools."""

import json
import sys
from types import ModuleType

import pytest

from model_tools import get_tool_definitions
import tools.discord_read_tool  # noqa: F401 - ensure tool registration
from tools.registry import registry


class _FakeHTTPError(Exception):
    pass


class _FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)

    def json(self):
        return self._data


class _FakeAsyncClient:
    def __init__(self, responder, **_kwargs):
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, params=None):
        status_code, payload = self._responder(method, url, params or {})
        return _FakeResponse(status_code, payload)


def _install_fake_httpx(monkeypatch, responder):
    httpx_mod = ModuleType("httpx")
    httpx_mod.AsyncClient = lambda **kwargs: _FakeAsyncClient(responder, **kwargs)
    httpx_mod.Timeout = lambda *args, **kwargs: None
    httpx_mod.HTTPError = _FakeHTTPError
    monkeypatch.setitem(sys.modules, "httpx", httpx_mod)


@pytest.fixture(autouse=True)
def _clear_discord_env(monkeypatch):
    for key in (
        "DISCORD_BOT_TOKEN",
        "DISCORD_READ_ALLOWED_GUILDS",
        "DISCORD_READ_ALLOWED_CHANNELS",
        "DISCORD_READ_INCLUDE_DMS",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_CHAT_NAME",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_HOME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_history_current_guild_session_is_auto_allowed(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "200")

    def responder(method, url, params):
        if url.endswith("/channels/200"):
            return 200, {"id": "200", "type": 0, "name": "general", "guild_id": "100"}
        if url.endswith("/guilds/100"):
            return 200, {"id": "100", "name": "Ops"}
        if url.endswith("/channels/200/messages"):
            assert params["limit"] == 2
            return 200, [
                {
                    "id": "m2",
                    "timestamp": "2026-04-02T10:00:00+00:00",
                    "content": "second",
                    "author": {"id": "u2", "username": "sam"},
                    "attachments": [],
                },
                {
                    "id": "m1",
                    "timestamp": "2026-04-02T09:00:00+00:00",
                    "content": "first",
                    "author": {"id": "u1", "username": "pat"},
                    "attachments": [],
                },
            ]
        raise AssertionError(f"Unexpected request: {method} {url} {params}")

    _install_fake_httpx(monkeypatch, responder)

    result = json.loads(registry.dispatch("discord_read_history", {"limit": 2}))

    assert result["channel"]["id"] == "200"
    assert result["channel"]["qualified_name"] == "Ops / #general"
    assert result["messages"][0]["permalink"] == "https://discord.com/channels/100/200/m2"
    assert len(result["messages"]) == 2


def test_history_current_dm_session_is_auto_allowed(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "900")

    def responder(method, url, params):
        if url.endswith("/channels/900"):
            return 200, {
                "id": "900",
                "type": 1,
                "recipients": [{"id": "42", "username": "avery"}],
            }
        if url.endswith("/channels/900/messages"):
            return 200, [
                {
                    "id": "dm1",
                    "timestamp": "2026-04-02T11:00:00+00:00",
                    "content": "hello from DM",
                    "author": {"id": "42", "username": "avery"},
                    "attachments": [],
                }
            ]
        raise AssertionError(f"Unexpected request: {method} {url} {params}")

    _install_fake_httpx(monkeypatch, responder)

    result = json.loads(registry.dispatch("discord_read_history", {}))

    assert result["channel"]["id"] == "900"
    assert result["channel"]["qualified_name"] == "avery"
    assert result["messages"][0]["permalink"] == "https://discord.com/channels/@me/900/dm1"


def test_history_denies_channel_outside_scope(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("DISCORD_READ_ALLOWED_CHANNELS", "200")

    def responder(method, url, params):
        if url.endswith("/channels/200"):
            return 200, {"id": "200", "type": 0, "name": "general", "guild_id": "100"}
        if url.endswith("/guilds/100"):
            return 200, {"id": "100", "name": "Ops"}
        if url.endswith("/guilds/100/threads/active"):
            return 200, {"threads": []}
        raise AssertionError(f"Unexpected request: {method} {url} {params}")

    _install_fake_httpx(monkeypatch, responder)

    result = json.loads(registry.dispatch("discord_read_history", {"channel": "400"}))

    assert "not accessible" in result["error"]


def test_list_channels_includes_active_threads_for_allowed_parent(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("DISCORD_READ_ALLOWED_CHANNELS", "201")

    def responder(method, url, params):
        if url.endswith("/channels/201"):
            return 200, {"id": "201", "type": 0, "name": "deploys", "guild_id": "100"}
        if url.endswith("/guilds/100"):
            return 200, {"id": "100", "name": "Ops"}
        if url.endswith("/guilds/100/threads/active"):
            return 200, {
                "threads": [
                    {
                        "id": "210",
                        "type": 11,
                        "name": "incident-7",
                        "guild_id": "100",
                        "parent_id": "201",
                    }
                ]
            }
        raise AssertionError(f"Unexpected request: {method} {url} {params}")

    _install_fake_httpx(monkeypatch, responder)

    result = json.loads(registry.dispatch("discord_list_channels", {}))

    by_id = {entry["id"]: entry for entry in result["channels"]}
    assert by_id["201"]["qualified_name"] == "Ops / #deploys"
    assert by_id["210"]["qualified_name"] == "Ops / #deploys / incident-7"
    assert by_id["210"]["allow_reason"] == "allowed_parent_channel"


def test_search_respects_result_and_scan_limits(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "200")

    def responder(method, url, params):
        if url.endswith("/channels/200"):
            return 200, {"id": "200", "type": 0, "name": "general", "guild_id": "100"}
        if url.endswith("/guilds/100"):
            return 200, {"id": "100", "name": "Ops"}
        if url.endswith("/channels/200/messages"):
            assert params["limit"] == 3
            return 200, [
                {
                    "id": "m3",
                    "timestamp": "2026-04-02T12:00:00+00:00",
                    "content": "deploy is green",
                    "author": {"id": "u1", "username": "sam"},
                    "attachments": [],
                },
                {
                    "id": "m2",
                    "timestamp": "2026-04-02T11:00:00+00:00",
                    "content": "deploy failed once",
                    "author": {"id": "u2", "username": "pat"},
                    "attachments": [],
                },
                {
                    "id": "m1",
                    "timestamp": "2026-04-02T10:00:00+00:00",
                    "content": "unrelated chatter",
                    "author": {"id": "u3", "username": "lee"},
                    "attachments": [],
                },
            ]
        raise AssertionError(f"Unexpected request: {method} {url} {params}")

    _install_fake_httpx(monkeypatch, responder)

    result = json.loads(
        registry.dispatch(
            "discord_search_messages",
            {"query": "deploy", "limit": 2, "scan_limit": 3},
        )
    )

    assert result["returned"] == 2
    assert result["scanned_messages"] == 3
    assert [match["id"] for match in result["matches"]] == ["m3", "m2"]


def test_search_requires_qualified_name_when_raw_name_is_ambiguous(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
    monkeypatch.setenv("DISCORD_READ_ALLOWED_GUILDS", "100,300")

    def responder(method, url, params):
        if url.endswith("/guilds/100"):
            return 200, {"id": "100", "name": "Ops"}
        if url.endswith("/guilds/100/channels"):
            return 200, [{"id": "200", "type": 0, "name": "general", "guild_id": "100"}]
        if url.endswith("/guilds/100/threads/active"):
            return 200, {"threads": []}
        if url.endswith("/guilds/300"):
            return 200, {"id": "300", "name": "Eng"}
        if url.endswith("/guilds/300/channels"):
            return 200, [{"id": "400", "type": 0, "name": "general", "guild_id": "300"}]
        if url.endswith("/guilds/300/threads/active"):
            return 200, {"threads": []}
        raise AssertionError(f"Unexpected request: {method} {url} {params}")

    _install_fake_httpx(monkeypatch, responder)

    result = json.loads(
        registry.dispatch("discord_search_messages", {"channel": "general", "query": "foo"})
    )

    assert "ambiguous" in result["error"]


def test_tool_definitions_expose_discord_read_toolset(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")

    tool_defs = get_tool_definitions(enabled_toolsets=["discord_read"], quiet_mode=True)
    tool_names = {tool["function"]["name"] for tool in tool_defs}

    assert tool_names == {
        "discord_list_channels",
        "discord_read_history",
        "discord_search_messages",
    }
