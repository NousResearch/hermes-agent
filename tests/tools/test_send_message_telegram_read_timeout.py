"""Regression tests for the standalone Telegram send read timeouts (#133164).

The ``send_message`` tool, when invoked outside the gateway (agent / TUI /
cron), runs ``_send_telegram`` directly. That standalone path used to build
``telegram.Bot(token=...)`` with python-telegram-bot's default 5s read
timeout: after a media upload finishes, Telegram often takes longer than 5s
to answer sendPhoto/sendVideo, so the send reported ``Timed out`` although
the message was delivered - and callers that retried sent duplicates. The
gateway adapter already used a 20s request read timeout and a 60s per-media
budget; these tests pin the standalone path to the same parity.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.tools.test_send_message_telegram_proxy import (
    _install_telegram_mock_with_request,
    _make_bot,
)


class TestStandaloneTelegramReadTimeouts:
    """The standalone send path must not run on PTB's 5s defaults."""

    def test_bot_request_carries_read_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a proxy, Bot() is still constructed with an HTTPXRequest
        carrying the adapter-parity 20s read timeout - not PTB's 5s default
        (#133164)."""
        from tools.send_message_tool import _send_telegram

        for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
                    "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
        monkeypatch.setattr(
            "gateway.platforms.base._detect_macos_system_proxy", lambda: None
        )

        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock_with_request(monkeypatch, bot_factory, httpx_request_factory)

        result: dict[str, Any] = asyncio.run(
            _send_telegram("tok", "123", "hello world")
        )

        assert result["success"] is True
        httpx_request_factory.assert_called_once()
        read_timeout = httpx_request_factory.call_args.kwargs.get("read_timeout")
        assert isinstance(read_timeout, (int, float)) and read_timeout >= 20.0, (
            f"standalone Bot request read_timeout is {read_timeout!r}; "
            "PTB's 5s default reports delivered media sends as Timed out (#133164)"
        )

    def test_media_send_overrides_read_timeout_per_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A media send passes the 60s media read timeout per call, matching
        the gateway adapter's _MEDIA_SEND_READ_TIMEOUT: sendVideo transcodes
        before answering and outlasts the 20s text budget (#133164)."""
        from tools.send_message_senders import _telegram_send_media

        bot = MagicMock()
        bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=7))

        async def run():
            class _File:
                def tell(self):
                    return 0

                def seek(self, *a):
                    return None

            return await _telegram_send_media(
                bot, "123", _File(), ".png", is_voice=False, force_document=False
            )

        asyncio.run(run())

        bot.send_photo.assert_awaited_once()
        call_kwargs = bot.send_photo.call_args.kwargs
        assert call_kwargs.get("read_timeout") == 60.0, (
            f"media send read_timeout is {call_kwargs.get('read_timeout')!r}; "
            "sendPhoto/sendVideo need the 60s media budget (#133164)"
        )
