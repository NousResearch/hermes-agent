"""Behavior tests for tools/telegram_tool.py (tg_send_sticker / tg_manage_stickers).

Covers check_fn gating (send: active Telegram session only; manage: session or
bot token), the session-scoped delivery contract (no model-facing chat_id),
the send happy path against a mocked ``telegram.Bot`` (normalized chat_id from
session env, file_id, forum-thread mapping incl. General-topic "1" → None),
resolve-failure wording, the TELEGRAM_PROXY branch, and all four
tg_manage_stickers actions against the real collection store in the isolated
temp HERMES_HOME.

Mocking conventions mirror tests/tools/test_send_message_tool.py and
test_send_message_telegram_proxy.py: a stub ``telegram`` package is installed
into ``sys.modules`` whose ``Bot`` is a factory mock.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import tools.telegram_tool as telegram_tool
from plugins.platforms.telegram import sticker_collection
from tools.registry import registry


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Deterministic environment: no ambient proxy/session/token state.

    Proxy vars are wiped so the no-proxy assertions can't flip with the host
    machine's proxy settings, and sys.platform is pinned to linux so macOS
    system-proxy auto-detection (scutil) can't kick in.
    """
    for var in (
        "TELEGRAM_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_THREAD_ID",
        "TELEGRAM_BOT_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")


def _make_bot(message_id: int = 42) -> MagicMock:
    bot = MagicMock()
    bot.send_sticker = AsyncMock(return_value=SimpleNamespace(message_id=message_id))
    return bot


def _install_telegram_mock(
    monkeypatch: pytest.MonkeyPatch,
    bot_factory: MagicMock,
    httpx_request_factory: Any = None,
) -> None:
    """Install a stub ``telegram`` package whose ``Bot`` is the supplied factory."""
    request_mod = SimpleNamespace(HTTPXRequest=httpx_request_factory or MagicMock())
    telegram_mod = SimpleNamespace(Bot=bot_factory, request=request_mod)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.request", request_mod)


def _sticker_obj(uid: str, emoji: str, set_name: str = "") -> Any:
    """Duck-typed telegram.Sticker for get_sticker_set responses."""
    return SimpleNamespace(
        file_id=f"fid-{uid}",
        file_unique_id=uid,
        emoji=emoji,
        is_animated=False,
        is_video=False,
        set_name=set_name,
    )


def _record(uid: str = "uid1", emoji: str = "😀", set_name: str = "MyPack", **kw) -> None:
    sticker_collection.record_sticker(
        uid, f"fid-{uid}", emoji=emoji, set_name=set_name, kind="static", **kw
    )


# ---------------------------------------------------------------------------
# Registration + check_fn gating
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_both_tools_registered_under_telegram_toolset(self) -> None:
        for name in ("tg_send_sticker", "tg_manage_stickers"):
            entry = registry._tools.get(name)
            assert entry is not None, f"{name} not registered"
            assert entry.toolset == "telegram"
            assert entry.is_async is True

    def test_send_sticker_gated_on_active_session_manage_on_token_or_session(self) -> None:
        assert registry._tools["tg_send_sticker"].check_fn is telegram_tool._check_telegram_session
        assert registry._tools["tg_manage_stickers"].check_fn is telegram_tool._check_telegram

    def test_send_sticker_schema_has_no_arbitrary_target(self) -> None:
        """Review contract: delivery is scoped to the active session, so the
        model-facing schema must not offer a chat_id-style outbound target."""
        props = registry._tools["tg_send_sticker"].schema["parameters"]["properties"]
        assert "chat_id" not in props


class TestCheckFnGating:
    def test_visible_in_telegram_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        assert telegram_tool._check_telegram() is True
        assert telegram_tool._check_telegram_session() is True

    def test_send_hidden_with_bot_token_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bot token without an active Telegram session is enough to MANAGE
        the collection (add_set works from the CLI) but not to SEND — there is
        no session chat to deliver to."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        assert telegram_tool._check_telegram() is True
        assert telegram_tool._check_telegram_session() is False

    def test_hidden_in_other_platform_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
        assert telegram_tool._check_telegram() is False
        assert telegram_tool._check_telegram_session() is False

    def test_hidden_with_neither(self) -> None:
        assert telegram_tool._check_telegram() is False
        assert telegram_tool._check_telegram_session() is False


# ---------------------------------------------------------------------------
# tg_send_sticker
# ---------------------------------------------------------------------------


class TestSendSticker:
    @pytest.mark.asyncio
    async def test_happy_path_uses_session_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record(description="a cat waving")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "-1001234567890")
        monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "17585")
        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock(monkeypatch, bot_factory)

        result = await telegram_tool.tg_send_sticker("😀")

        assert result["success"] is True
        assert result["message_id"] == 42
        assert result["chat_id"] == "-1001234567890"
        assert result["sticker"] == {
            "file_unique_id": "uid1",
            "emoji": "😀",
            "set_name": "MyPack",
        }
        assert "end your turn" in result["note"]
        # No proxy configured → plain one-shot Bot construction.
        bot_factory.assert_called_once_with(token="tok123")
        bot.send_sticker.assert_awaited_once()
        kwargs = bot.send_sticker.await_args.kwargs
        assert kwargs["chat_id"] == -1001234567890  # normalized to int
        assert kwargs["sticker"] == "fid-uid1"
        assert kwargs["message_thread_id"] == 17585  # forum thread id → int

    @pytest.mark.asyncio
    async def test_general_topic_thread_id_maps_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forum General topic arrives as thread id "1" but the Bot API rejects
        message_thread_id=1 — the kwarg must be omitted entirely."""
        _record()
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "-1001234567890")
        monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "1")
        bot = _make_bot()
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

        result = await telegram_tool.tg_send_sticker("😀")

        assert result["success"] is True
        kwargs = bot.send_sticker.await_args.kwargs
        assert "message_thread_id" not in kwargs

    @pytest.mark.asyncio
    async def test_no_thread_id_no_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _record()
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123")
        bot = _make_bot()
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

        result = await telegram_tool.tg_send_sticker("😀")

        assert result["success"] is True
        kwargs = bot.send_sticker.await_args.kwargs
        assert kwargs["chat_id"] == 123
        assert "message_thread_id" not in kwargs

    @pytest.mark.asyncio
    async def test_set_name_disambiguates_shared_emoji(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record("uid-a", emoji="😀", set_name="PackOne")
        _record("uid-b", emoji="😀", set_name="PackTwo")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123")
        bot = _make_bot()
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

        result = await telegram_tool.tg_send_sticker("😀", set_name="PackOne")

        assert result["success"] is True
        assert result["sticker"]["file_unique_id"] == "uid-a"
        assert bot.send_sticker.await_args.kwargs["sticker"] == "fid-uid-a"

    @pytest.mark.asyncio
    async def test_resolve_failure_message_points_to_add_set(self) -> None:
        result = await telegram_tool.tg_send_sticker("🦄")

        assert result["success"] is False
        assert "not in your collection" in result["error"]
        assert "add_set" in result["error"]

    @pytest.mark.asyncio
    async def test_no_active_session_chat_errors(self) -> None:
        _record()
        result = await telegram_tool.tg_send_sticker("😀")

        assert result["success"] is False
        assert "No active Telegram chat" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_bot_token_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _record()
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123")
        result = await telegram_tool.tg_send_sticker("😀")

        assert result["success"] is False
        assert "TELEGRAM_BOT_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_api_error_is_returned_with_token_redacted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record()
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123")
        bot = _make_bot()
        bot.send_sticker = AsyncMock(
            side_effect=Exception("Bad Request: wrong file_id for bot tok123")
        )
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

        result = await telegram_tool.tg_send_sticker("😀")

        assert result["success"] is False
        assert "wrong file_id" in result["error"]
        assert "tok123" not in result["error"]
        assert "***" in result["error"]

    @pytest.mark.asyncio
    async def test_proxy_routes_bot_through_httpx_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record()
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123")
        monkeypatch.setenv("TELEGRAM_PROXY", "socks5://127.0.0.1:1080")
        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock(monkeypatch, bot_factory, httpx_request_factory)

        result = await telegram_tool.tg_send_sticker("😀")

        assert result["success"] is True
        bot_factory.assert_called_once()
        call_kwargs = bot_factory.call_args.kwargs
        assert call_kwargs["token"] == "tok123"
        assert "request" in call_kwargs, "request= kwarg missing — proxy not wired"
        assert "get_updates_request" in call_kwargs
        assert httpx_request_factory.call_count == 2
        for call in httpx_request_factory.call_args_list:
            assert call.kwargs.get("proxy") == "socks5://127.0.0.1:1080"
        bot.send_sticker.assert_awaited_once()


# ---------------------------------------------------------------------------
# tg_manage_stickers — list / update / remove (local store, no bot needed)
# ---------------------------------------------------------------------------


class TestManageList:
    @pytest.mark.asyncio
    async def test_list_returns_summaries_without_file_id(self) -> None:
        _record("uid1", emoji="😀", set_name="A", description="cat")
        _record("uid2", emoji="😂", set_name="B")
        _record("uid3", emoji="🚀", set_name="B")

        result = await telegram_tool.tg_manage_stickers("list")

        assert result["success"] is True
        assert result["total"] == 3
        assert result["returned"] == 3
        by_uid = {e["file_unique_id"]: e for e in result["stickers"]}
        assert set(by_uid) == {"uid1", "uid2", "uid3"}
        for entry in result["stickers"]:
            assert set(entry) == {"file_unique_id", "emoji", "set_name", "kind", "description"}
        assert by_uid["uid1"]["description"] == "cat"

    @pytest.mark.asyncio
    async def test_list_set_name_filter_and_limit(self) -> None:
        _record("uid1", emoji="😀", set_name="A")
        _record("uid2", emoji="😂", set_name="B")
        _record("uid3", emoji="🚀", set_name="B")

        filtered = await telegram_tool.tg_manage_stickers("list", set_name="B")
        assert filtered["total"] == 2
        assert filtered["set_name"] == "B"
        assert {e["file_unique_id"] for e in filtered["stickers"]} == {"uid2", "uid3"}

        capped = await telegram_tool.tg_manage_stickers("list", limit=2)
        assert capped["returned"] == 2
        assert capped["total"] == 3
        assert "note" in capped  # truncation hint

    @pytest.mark.asyncio
    async def test_handler_wrapper_returns_json_string(self) -> None:
        raw = await telegram_tool._handle_tg_manage_stickers({"action": "list"})
        parsed = json.loads(raw)
        assert parsed["success"] is True
        assert parsed["total"] == 0


class TestManageUpdate:
    @pytest.mark.asyncio
    async def test_update_by_file_unique_id(self) -> None:
        _record("uid1")

        result = await telegram_tool.tg_manage_stickers(
            "update", file_unique_id="uid1", description="use when the user is sarcastic"
        )

        assert result["success"] is True
        assert result["updated"] == 1
        assert result["entry"]["description"] == "use when the user is sarcastic"
        assert sticker_collection.resolve("uid1")["description"] == "use when the user is sarcastic"

    @pytest.mark.asyncio
    async def test_update_by_sticker_query_and_clear(self) -> None:
        _record("uid1", emoji="😀", description="first")

        result = await telegram_tool.tg_manage_stickers(
            "update", sticker="😀", description="second"
        )
        assert result["success"] is True
        assert sticker_collection.resolve("uid1")["description"] == "second"

        cleared = await telegram_tool.tg_manage_stickers(
            "update", sticker="😀", description=""
        )
        assert cleared["success"] is True
        assert cleared["entry"]["description"] == ""
        assert sticker_collection.resolve("uid1")["description"] == ""

    @pytest.mark.asyncio
    async def test_update_requires_a_selector(self) -> None:
        result = await telegram_tool.tg_manage_stickers("update", description="x")

        assert result["success"] is False
        assert "file_unique_id" in result["error"]

    @pytest.mark.asyncio
    async def test_update_unknown_selector_errors(self) -> None:
        result = await telegram_tool.tg_manage_stickers(
            "update", file_unique_id="nope", description="x"
        )

        assert result["success"] is False
        assert "nope" in result["error"]
        assert "list" in result["error"]


class TestManageRemove:
    @pytest.mark.asyncio
    async def test_remove_by_file_unique_id(self) -> None:
        _record("uid1", emoji="😀")

        result = await telegram_tool.tg_manage_stickers("remove", file_unique_id="uid1")

        assert result["success"] is True
        assert result["removed"] == 1
        assert result["entry"]["file_unique_id"] == "uid1"
        assert sticker_collection.list_stickers() == []

    @pytest.mark.asyncio
    async def test_remove_by_sticker_query(self) -> None:
        _record("uid1", emoji="😀")

        result = await telegram_tool.tg_manage_stickers("remove", sticker="😀")

        assert result["success"] is True
        assert sticker_collection.resolve("😀") is None

    @pytest.mark.asyncio
    async def test_remove_unknown_selector_errors(self) -> None:
        result = await telegram_tool.tg_manage_stickers("remove", file_unique_id="nope")

        assert result["success"] is False
        assert "nope" in result["error"]


# ---------------------------------------------------------------------------
# tg_manage_stickers — add_set (one-shot bot + refresh_from_sets)
# ---------------------------------------------------------------------------


class TestManageAddSet:
    def _bot_with_set(self, monkeypatch: pytest.MonkeyPatch, stickers: list) -> MagicMock:
        bot = MagicMock()
        bot.get_sticker_set = AsyncMock(
            return_value=SimpleNamespace(name="HotCherry", stickers=stickers)
        )
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
        return bot

    @pytest.mark.asyncio
    async def test_add_set_imports_pack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        bot = self._bot_with_set(
            monkeypatch, [_sticker_obj("uid1", "😀"), _sticker_obj("uid2", "😂")]
        )

        result = await telegram_tool.tg_manage_stickers("add_set", set_name="HotCherry")

        assert result["success"] is True
        assert result["set_name"] == "HotCherry"
        assert result["sets"] == 1
        assert result["stickers"] == 2
        assert result["new"] == 2
        bot.get_sticker_set.assert_awaited_once_with("HotCherry")
        entries = {e["file_unique_id"] for e in sticker_collection.list_stickers()}
        assert entries == {"uid1", "uid2"}

    @pytest.mark.asyncio
    async def test_add_set_parses_tme_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        bot = self._bot_with_set(monkeypatch, [_sticker_obj("uid1", "😀")])

        result = await telegram_tool.tg_manage_stickers(
            "add_set", set_name="https://t.me/addstickers/HotCherry"
        )

        assert result["success"] is True
        assert result["set_name"] == "HotCherry"
        bot.get_sticker_set.assert_awaited_once_with("HotCherry")

    @pytest.mark.asyncio
    async def test_add_set_fetch_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        bot = MagicMock()
        bot.get_sticker_set = AsyncMock(side_effect=Exception("Bad Request: STICKERSET_INVALID"))
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

        result = await telegram_tool.tg_manage_stickers("add_set", set_name="NoSuchPack")

        assert result["success"] is False
        assert "NoSuchPack" in result["error"]
        assert sticker_collection.list_stickers() == []

    @pytest.mark.asyncio
    async def test_add_set_requires_name(self) -> None:
        result = await telegram_tool.tg_manage_stickers("add_set")

        assert result["success"] is False
        assert "set_name" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_action_errors(self) -> None:
        result = await telegram_tool.tg_manage_stickers("search")

        assert result["success"] is False
        assert "list" in result["error"] and "add_set" in result["error"]


class TestSetNameParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("HotCherry", "HotCherry"),
            ("https://t.me/addstickers/HotCherry", "HotCherry"),
            ("http://t.me/addstickers/HotCherry", "HotCherry"),
            ("t.me/addstickers/HotCherry", "HotCherry"),
            ("  HotCherry  ", "HotCherry"),
            ("", ""),
        ],
    )
    def test_parse_set_short_name(self, raw: str, expected: str) -> None:
        assert telegram_tool._parse_set_short_name(raw) == expected
