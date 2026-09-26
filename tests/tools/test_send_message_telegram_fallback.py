"""Regression tests for the standalone Telegram send path's fallback-IP transport.

The gateway adapter routes Telegram requests through ``TelegramFallbackTransport``, so a
failed connect on the primary api.telegram.org path is retried against a known Telegram IP.
The standalone ``_send_telegram`` path (``hermes send``, cron delivery without a live
adapter) built a plain ``Bot`` instead, so on networks where the primary path intermittently
fails, standalone sends failed where the gateway kept working (#20915). Separately, a connect
failure was never retried, although it provably happens before any request byte is sent.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tests.tools.test_send_message_telegram_proxy import (
    _install_telegram_mock_with_request,
    _make_bot,
)


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
                "TELEGRAM_FALLBACK_IPS", "HERMES_TELEGRAM_DISABLE_FALLBACK_IPS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)


def _mocks(monkeypatch: pytest.MonkeyPatch, bot=None):
    bot = bot or _make_bot()
    bot_factory = MagicMock(return_value=bot)
    httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
    _install_telegram_mock_with_request(monkeypatch, bot_factory, httpx_request_factory)
    return bot, bot_factory, httpx_request_factory


def _transport(httpx_request_factory: MagicMock):
    return httpx_request_factory.call_args.kwargs["httpx_kwargs"]["transport"]


def _chained(outer: Exception, cause: Exception) -> Exception:
    """``outer`` raised ``from cause``, as PTB's HTTPXRequest wraps httpx errors."""
    try:
        try:
            raise cause
        except Exception as c:
            raise outer from c
    except Exception as e:
        return e


class TestSendTelegramStandaloneFallbackTransport:

    def test_no_proxy_uses_fallback_transport_with_seed_ips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.platforms.telegram.telegram_network import SEED_FALLBACK_IPS, TelegramFallbackTransport
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        bot, bot_factory, httpx_request_factory = _mocks(monkeypatch)

        result: dict[str, Any] = asyncio.run(_send_telegram("tok", "123", "hello world"))

        assert result["success"] is True
        assert "request" in bot_factory.call_args.kwargs
        httpx_request_factory.assert_called_once()
        transport = _transport(httpx_request_factory)
        assert isinstance(transport, TelegramFallbackTransport)
        assert transport._fallback_ips == list(SEED_FALLBACK_IPS)
        bot.send_message.assert_awaited_once()

    def test_configured_fallback_ips_override_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        monkeypatch.setenv("TELEGRAM_FALLBACK_IPS", "149.154.167.221")
        _, _, httpx_request_factory = _mocks(monkeypatch)

        asyncio.run(_send_telegram("tok", "123", "hello world"))

        assert _transport(httpx_request_factory)._fallback_ips == ["149.154.167.221"]

    def test_proxy_takes_precedence_over_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        monkeypatch.setenv("TELEGRAM_PROXY", "socks5://127.0.0.1:1080")
        _, bot_factory, httpx_request_factory = _mocks(monkeypatch)

        asyncio.run(_send_telegram("tok", "123", "hello world"))

        assert "get_updates_request" in bot_factory.call_args.kwargs
        for call in httpx_request_factory.call_args_list:
            assert "transport" not in (call.kwargs.get("httpx_kwargs") or {})

    def test_fallback_setup_failure_degrades_to_plain_bot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        bot, bot_factory, _ = _mocks(monkeypatch)
        monkeypatch.setattr("plugins.platforms.telegram.telegram_network.TelegramFallbackTransport",
                            MagicMock(side_effect=RuntimeError("boom")))

        result: dict[str, Any] = asyncio.run(_send_telegram("tok", "123", "hello world"))

        assert result["success"] is True
        assert "request" not in bot_factory.call_args.kwargs
        bot.send_message.assert_awaited_once()

    def test_connect_failure_is_retried_then_delivered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=[
            _chained(RuntimeError("httpx.ConnectError: "), httpx.ConnectError("")),
            SimpleNamespace(message_id=42),
        ])
        _mocks(monkeypatch, bot)

        result: dict[str, Any] = asyncio.run(_send_telegram("tok", "123", "hello world"))

        assert result["success"] is True
        assert bot.send_message.await_count == 2

    def test_read_timeout_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=_chained(RuntimeError("Timed out"), httpx.ReadTimeout("")))
        _mocks(monkeypatch, bot)

        result: dict[str, Any] = asyncio.run(_send_telegram("tok", "123", "hello world"))

        assert "error" in result
        assert bot.send_message.await_count == 1


class TestTelegramRetryDelayConnectFailures:

    @pytest.mark.parametrize("cause", [httpx.ConnectError(""), httpx.ConnectTimeout("")])
    def test_connect_phase_failures_back_off(self, cause: Exception) -> None:
        from tools.send_message_senders import _telegram_retry_delay

        # ConnectTimeout reaches us as a bare "Timed out"; only the chained cause tells it apart.
        exc = _chained(RuntimeError("Timed out" if isinstance(cause, httpx.ConnectTimeout)
                                    else "httpx.ConnectError: "), cause)
        assert _telegram_retry_delay(exc, 0) == 1.0
        assert _telegram_retry_delay(exc, 1) == 2.0

    @pytest.mark.parametrize("cause", [httpx.ReadTimeout(""), httpx.WriteTimeout(""), httpx.WriteError("")])
    def test_failures_after_sending_began_are_not_retried(self, cause: Exception) -> None:
        from tools.send_message_senders import _telegram_retry_delay

        assert _telegram_retry_delay(_chained(RuntimeError("Timed out"), cause), 0) is None

    def test_connect_error_text_without_a_cause_is_not_retried(self) -> None:
        from tools.send_message_senders import _telegram_retry_delay

        assert _telegram_retry_delay(RuntimeError("httpx.ConnectError: "), 0) is None
