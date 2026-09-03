"""Regression tests for the standalone Telegram send path's Bot API base_url support.

The ``send_message`` tool, when invoked from a process *other than* the
gateway (agent / TUI / cron), runs ``_send_telegram`` directly instead of
delegating to the in-process gateway adapter.  Before the fix that accompanies
these tests, that standalone path constructed ``telegram.Bot(token=...)``
without honouring the custom Bot API ``base_url`` / ``base_file_url`` that the
gateway adapter already reads from ``platforms.telegram.extra``.

As a result proactive / cron-triggered sends bypassed a configured self-hosted
Bot API server and went to ``api.telegram.org`` directly, timing out in regions
where that host is blocked — even though interactive gateway replies worked
fine (issue #51223).

These tests verify that the standalone path now mirrors the gateway adapter:
``base_url`` and ``base_file_url`` are passed through to ``Bot()``.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock_with_request(
    monkeypatch: pytest.MonkeyPatch,
    bot_factory: MagicMock,
    httpx_request_factory: MagicMock | None = None,
) -> None:
    """Install a stub ``telegram`` package with the supplied ``Bot`` mock."""
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    request_mod = SimpleNamespace(HTTPXRequest=httpx_request_factory or MagicMock())
    _MessageEntity = lambda **_kw: SimpleNamespace(**_kw)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=_MessageEntity,
        constants=constants_mod,
        request=request_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)
    monkeypatch.setitem(sys.modules, "telegram.request", request_mod)


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    return bot


def _wipe_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every env var ``resolve_proxy_url`` inspects so the host's
    ambient proxy settings cannot flip a test green-or-red."""
    for var in (
        "TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy",
        "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
        "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


class TestSendTelegramBaseUrl:
    """The standalone ``_send_telegram`` path must honour a custom Bot API
    ``base_url`` / ``base_file_url``, mirroring the gateway adapter."""

    def test_base_url_passed_to_bot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``base_url`` set, ``Bot()`` receives ``base_url=`` and
        ``base_file_url=`` kwargs."""
        from tools.send_message_tool import _send_telegram

        _wipe_proxy_env(monkeypatch)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
        monkeypatch.setattr(sys, "platform", "linux")

        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock_with_request(monkeypatch, bot_factory)

        result: dict[str, Any] = asyncio.run(
            _send_telegram(
                "tok", "123", "hello world",
                base_url="https://botapi.example.com:8443/bot",
            )
        )

        assert result["success"] is True
        bot_factory.assert_called_once()
        call_kwargs = bot_factory.call_args.kwargs
        assert call_kwargs.get("token") == "tok"
        assert call_kwargs.get("base_url") == "https://botapi.example.com:8443/bot"
        # base_file_url must default to base_url when not separately configured.
        assert call_kwargs.get("base_file_url") == "https://botapi.example.com:8443/bot"
        bot.send_message.assert_awaited_once()

    def test_explicit_base_file_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``base_file_url`` is provided separately, it is passed as-is
        rather than defaulting to ``base_url``."""
        from tools.send_message_tool import _send_telegram

        _wipe_proxy_env(monkeypatch)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
        monkeypatch.setattr(sys, "platform", "linux")

        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock_with_request(monkeypatch, bot_factory)

        result: dict[str, Any] = asyncio.run(
            _send_telegram(
                "tok", "123", "hello world",
                base_url="https://botapi.example.com:8443/bot",
                base_file_url="https://botapi.example.com:8443/file/bot",
            )
        )

        assert result["success"] is True
        call_kwargs = bot_factory.call_args.kwargs
        assert call_kwargs.get("base_url") == "https://botapi.example.com:8443/bot"
        assert call_kwargs.get("base_file_url") == "https://botapi.example.com:8443/file/bot"

    def test_base_url_with_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both ``base_url`` and a proxy can be active simultaneously — the
        Bot gets HTTPXRequest instances *and* the custom base URLs."""
        from tools.send_message_tool import _send_telegram

        proxy_url = "socks5://127.0.0.1:1080"
        monkeypatch.setenv("TELEGRAM_PROXY", proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock_with_request(monkeypatch, bot_factory, httpx_request_factory)

        result: dict[str, Any] = asyncio.run(
            _send_telegram(
                "tok", "123", "hello world",
                base_url="https://botapi.example.com:8443/bot",
            )
        )

        assert result["success"] is True
        bot_factory.assert_called_once()
        call_kwargs = bot_factory.call_args.kwargs
        assert call_kwargs.get("base_url") == "https://botapi.example.com:8443/bot"
        assert call_kwargs.get("base_file_url") == "https://botapi.example.com:8443/bot"
        # Proxy must still be wired.
        assert "request" in call_kwargs
        assert "get_updates_request" in call_kwargs
        assert httpx_request_factory.call_count == 2
        bot.send_message.assert_awaited_once()

    def test_no_base_url_uses_plain_bot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``base_url`` (the default), ``Bot()`` has no
        ``base_url``/``base_file_url`` kwargs — backward compatible."""
        from tools.send_message_tool import _send_telegram

        _wipe_proxy_env(monkeypatch)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
        monkeypatch.setattr(sys, "platform", "linux")

        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock_with_request(monkeypatch, bot_factory)

        result: dict[str, Any] = asyncio.run(
            _send_telegram("tok", "123", "hello world")
        )

        assert result["success"] is True
        bot_factory.assert_called_once()
        call_kwargs = bot_factory.call_args.kwargs
        assert "base_url" not in call_kwargs
        assert "base_file_url" not in call_kwargs
        bot.send_message.assert_awaited_once()

    def test_send_to_platform_dispatches_base_url_to_telegram(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``_send_to_platform`` dispatch boundary forwards
        ``pconfig.extra`` base_url / base_file_url through to the mocked
        Telegram ``Bot``, so a caller with a configured
        ``PlatformConfig(extra={...})`` gets custom API endpoints
        without ever touching ``_send_telegram`` directly."""
        from gateway.config import Platform, PlatformConfig
        from tools.send_message_tool import _send_to_platform

        _wipe_proxy_env(monkeypatch)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
        monkeypatch.setattr(sys, "platform", "linux")

        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock_with_request(monkeypatch, bot_factory)

        pconfig = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "base_url": "https://mybot.example.com:8443/bot",
                "base_file_url": "https://mybot.example.com:8443/file/bot",
            },
        )

        result: dict[str, Any] = asyncio.run(
            _send_to_platform(
                Platform.TELEGRAM,
                pconfig,
                "123",
                "hello from dispatch",
            )
        )

        assert result["success"] is True
        bot_factory.assert_called_once()
        call_kwargs = bot_factory.call_args.kwargs
        assert call_kwargs.get("token") == "test-token"
        # Both extra values must propagate through the dispatch boundary.
        assert call_kwargs.get("base_url") == "https://mybot.example.com:8443/bot"
        assert call_kwargs.get("base_file_url") == "https://mybot.example.com:8443/file/bot"
        bot.send_message.assert_awaited_once()
