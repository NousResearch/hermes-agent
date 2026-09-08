"""Standalone Telegram outbound policy and shared PEER_FLOOD circuit tests."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import outbound_circuit
from plugins.platforms.telegram.adapter import TelegramAdapter, _standalone_send
from plugins.platforms.telegram.outbound_policy import deactivate_coupang_urls
from tools.send_message_tool import _send_telegram


class PeerFloodError(Exception):
    def __init__(self, retry_after=None):
        super().__init__("Telegram server error: PEER_FLOOD")
        self.retry_after = retry_after


@pytest.fixture(autouse=True)
def _isolated_circuit(tmp_path, monkeypatch):
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")


@pytest.fixture
def telegram_bot(monkeypatch):
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=2))
    bot.send_video = AsyncMock(return_value=SimpleNamespace(message_id=3))
    bot.send_voice = AsyncMock(return_value=SimpleNamespace(message_id=4))
    bot.send_audio = AsyncMock(return_value=SimpleNamespace(message_id=5))
    bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=6))
    factory = MagicMock(return_value=bot)
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants = SimpleNamespace(ParseMode=parse_mode)
    telegram = SimpleNamespace(
        Bot=factory,
        MessageEntity=lambda **kwargs: SimpleNamespace(**kwargs),
        constants=constants,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants)
    monkeypatch.setattr(
        "gateway.platforms.base.resolve_proxy_url", lambda *_args, **_kwargs: None
    )
    return bot, factory


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    return adapter


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "상품 https://www.coupang.com/vp/products/1?itemId=2#review",
            "상품 https://www.coupang[.]com/vp/products/1?itemId=2#review",
        ),
        ("//link.coupang.com/a/ABC", "//link.coupang[.]com/a/ABC"),
        ("coupang.com/item", "coupang[.]com/item"),
        (
            "https://example.com/?next=//coupang.com/item#coupang.com",
            "https://example.com/?next=//coupang.com/item#coupang.com",
        ),
    ],
)
def test_standalone_defangs_only_coupang_origins(
    source, expected, telegram_bot, monkeypatch
):
    bot, _factory = telegram_bot
    monkeypatch.setattr(TelegramAdapter, "format_message", lambda _self, value: value)

    result = asyncio.run(_send_telegram("token", "123", source))

    assert deactivate_coupang_urls(source) == expected
    assert result["success"] is True
    assert bot.send_message.await_args.kwargs["text"] == expected


def test_adapter_opened_circuit_blocks_standalone_with_same_normalized_chat_key(
    telegram_bot,
):
    bot, _factory = telegram_bot
    pconfig = SimpleNamespace(token="token", extra={})

    async def run():
        adapter = _adapter()
        adapter._open_peer_flood_circuit("00123", PeerFloodError(60.0))
        return await _standalone_send(pconfig, " 123 ", "blocked")

    result = asyncio.run(run())

    assert result["error_kind"] == "peer_flood"
    assert result["retryable"] is False
    bot.send_message.assert_not_awaited()


def test_standalone_peer_flood_opens_circuit_seen_by_adapter_without_retry(
    telegram_bot, monkeypatch
):
    bot, _factory = telegram_bot
    bot.send_message.side_effect = PeerFloodError(42.0)
    sleep = AsyncMock()
    monkeypatch.setattr("tools.send_message_tool.asyncio.sleep", sleep)

    result = asyncio.run(_send_telegram("token", "00123", "hello"))
    blocked = asyncio.run(_adapter().send("123", "blocked", metadata={"notify": True}))

    assert result["error_kind"] == "peer_flood"
    assert result["retryable"] is False
    assert 0 < result["retry_after"] <= 42.0
    assert blocked.error_kind == "peer_flood"
    assert bot.send_message.await_count == 1
    sleep.assert_not_awaited()


def test_standalone_peer_flood_response_opens_circuit(telegram_bot):
    bot, _factory = telegram_bot
    bot.send_message.return_value = {
        "ok": False,
        "description": "Bad Request: PEER_FLOOD",
        "parameters": {"retry_after": 30},
    }

    result = asyncio.run(_send_telegram("token", "123", "hello"))

    assert result["error_kind"] == "peer_flood"
    assert 0 < result["retry_after"] <= 30.0
    assert outbound_circuit.remaining("123") is not None


def test_media_peer_flood_stops_remaining_media_and_fallback(
    telegram_bot, tmp_path
):
    bot, _factory = telegram_bot
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"png")
    second.write_bytes(b"png")
    bot.send_photo.side_effect = [
        PeerFloodError(60.0),
        SimpleNamespace(message_id=2),
    ]

    result = asyncio.run(
        _send_telegram(
            "token",
            "123",
            "",
            media_files=[(str(first), False), (str(second), False)],
        )
    )

    assert result["error_kind"] == "peer_flood"
    assert bot.send_photo.await_count == 1
    bot.send_document.assert_not_awaited()


def test_standalone_async_path_uses_async_circuit_wrappers(
    telegram_bot, monkeypatch
):
    bot, _factory = telegram_bot
    bot.send_message.side_effect = PeerFloodError(60.0)
    monkeypatch.setattr(
        outbound_circuit,
        "remaining",
        lambda *_args: (_ for _ in ()).throw(AssertionError("sync DB read")),
    )
    monkeypatch.setattr(
        outbound_circuit,
        "open_circuit",
        lambda *_args: (_ for _ in ()).throw(AssertionError("sync DB write")),
    )
    remaining_async = AsyncMock(return_value=None)
    open_async = AsyncMock(return_value=60.0)
    monkeypatch.setattr(outbound_circuit, "remaining_async", remaining_async)
    monkeypatch.setattr(outbound_circuit, "open_circuit_async", open_async)

    result = asyncio.run(_send_telegram("token", "123", "hello"))

    assert result["error_kind"] == "peer_flood"
    remaining_async.assert_awaited_once_with("123")
    open_async.assert_awaited_once_with("123", 60.0)


def test_standalone_write_failure_blocks_next_call(
    telegram_bot, monkeypatch
):
    bot, _factory = telegram_bot
    bot.send_message.side_effect = PeerFloodError(60.0)
    real_connection = outbound_circuit._connection

    def broken_connection():
        raise OSError("disk unavailable")

    real_open_async = outbound_circuit.open_circuit_async

    async def fail_persist(chat_id, delay):
        outbound_circuit._connection = broken_connection
        try:
            return await real_open_async(chat_id, delay)
        finally:
            outbound_circuit._connection = real_connection

    monkeypatch.setattr(outbound_circuit, "open_circuit_async", fail_persist)
    failed = asyncio.run(_send_telegram("token", "123", "first"))
    bot.send_message.side_effect = None

    blocked = asyncio.run(_send_telegram("token", "123", "second"))

    assert failed["error_kind"] == "peer_flood"
    assert blocked["error_kind"] == "peer_flood"
    assert bot.send_message.await_count == 1
