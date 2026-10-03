"""Regression tests for standalone Telegram sends against a custom (self-hosted) Bot API server.

The ``send_message`` tool, when invoked from a process *other than* the gateway
(agent / TUI / cron), runs ``_send_telegram`` directly instead of delegating to the
in-process gateway adapter. Two gaps made that standalone path unusable for
deployments that run their own Bot API server (issue #51223):

1. It constructed ``telegram.Bot(token=...)`` without honouring the custom Bot API
   ``base_url`` / ``base_file_url`` that the gateway adapter already reads from
   ``platforms.telegram.extra`` — so proactive / cron sends bypassed the
   self-hosted server and went to ``api.telegram.org`` directly.
2. Even with ``base_url`` wired, PTB's 5s default read timeout fails any sizeable
   media upload: the server (local or proxied) only answers once the whole upload
   has been forwarded to Telegram — measured ~15s for an 8.9 MB file before
   ``Timed out``.

These tests verify the standalone path now mirrors the gateway adapter: ``base_url``
/ ``base_file_url`` reach ``Bot()``, and every request is built with the same
``HERMES_TELEGRAM_HTTP_READ_TIMEOUT`` knob the adapter honours (standalone default
60s — long enough to ride out a media upload's forward).
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(
    monkeypatch: pytest.MonkeyPatch,
    bot_factory: MagicMock,
    httpx_request_factory: MagicMock,
) -> None:
    """Stub the ``telegram`` package (Bot + request.HTTPXRequest).

    Mirrors ``_install_telegram_mock_with_request`` in
    ``test_send_message_telegram_proxy.py``.
    """
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    request_mod = SimpleNamespace(HTTPXRequest=httpx_request_factory)
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


def _wipe_connection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every env var the connection setup inspects, so ambient proxy
    settings (or a host read-timeout override) cannot flip a test green-or-red.
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
        "HERMES_TELEGRAM_HTTP_READ_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    # Keep macOS system-proxy auto-detection out of the picture on every runner.
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    # Ensure the test does not depend on the in-process gateway runner.
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)


def _request_kwargs(factory: MagicMock) -> list[dict[str, Any]]:
    return [call.kwargs for call in factory.call_args_list]


class TestSendTelegramSelfHostedBaseUrl:
    """``base_url`` / ``base_file_url`` from ``platforms.telegram.extra`` must reach
    the standalone ``Bot()``, mirroring the gateway adapter."""

    def test_base_url_and_file_url_reach_bot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _wipe_connection_env(monkeypatch)
        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock(monkeypatch, bot_factory, MagicMock())

        result: dict[str, Any] = asyncio.run(
            _send_telegram(
                "tok",
                "123",
                "hello world",
                extra={
                    "base_url": "http://127.0.0.1:8081/bot",
                    "base_file_url": "http://127.0.0.1:8081/file",
                },
            )
        )

        assert result["success"] is True
        kwargs = bot_factory.call_args.kwargs
        assert kwargs.get("base_url") == "http://127.0.0.1:8081/bot"
        assert kwargs.get("base_file_url") == "http://127.0.0.1:8081/file"
        bot.send_message.assert_awaited_once()

    def test_base_file_url_defaults_to_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _wipe_connection_env(monkeypatch)
        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock(monkeypatch, bot_factory, MagicMock())

        result: dict[str, Any] = asyncio.run(
            _send_telegram("tok", "123", "hi", extra={"base_url": "http://127.0.0.1:8081/bot"})
        )

        assert result["success"] is True
        assert bot_factory.call_args.kwargs.get("base_file_url") == "http://127.0.0.1:8081/bot"

    def test_explicit_base_url_wins_over_proxy_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``base_url`` targets the configured server directly; the env
        proxy (which exists for reaching api.telegram.org) is not attached."""
        from tools.send_message_tool import _send_telegram

        _wipe_connection_env(monkeypatch)
        monkeypatch.setenv("TELEGRAM_PROXY", "socks5://127.0.0.1:1080")
        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock(monkeypatch, bot_factory, request_factory)

        result: dict[str, Any] = asyncio.run(
            _send_telegram("tok", "123", "hi", extra={"base_url": "http://127.0.0.1:8081/bot"})
        )

        assert result["success"] is True
        assert bot_factory.call_args.kwargs.get("base_url") == "http://127.0.0.1:8081/bot"
        for kwargs in _request_kwargs(request_factory):
            assert "proxy" not in kwargs, (
                f"base_url sends must not route through the env proxy: {kwargs!r}"
            )


class TestSendTelegramReadTimeout:
    """Every standalone request carries ``HERMES_TELEGRAM_HTTP_READ_TIMEOUT``
    (adapter parity). PTB's 5s default fails any sizeable media upload — the server
    answers only after the upload has been forwarded to Telegram."""

    def test_default_read_timeout_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_senders import _telegram_bot

        _wipe_connection_env(monkeypatch)
        request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock(monkeypatch, MagicMock(return_value=MagicMock()), request_factory)

        _telegram_bot("tok")

        (kwargs,) = _request_kwargs(request_factory)
        assert kwargs["read_timeout"] == 60.0

    def test_env_override_read_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_senders import _telegram_bot

        _wipe_connection_env(monkeypatch)
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "123.5")
        request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock(monkeypatch, MagicMock(return_value=MagicMock()), request_factory)

        _telegram_bot("tok")

        (kwargs,) = _request_kwargs(request_factory)
        assert kwargs["read_timeout"] == 123.5

    def test_invalid_env_value_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_senders import _telegram_bot

        _wipe_connection_env(monkeypatch)
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "garbage")
        request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock(monkeypatch, MagicMock(return_value=MagicMock()), request_factory)

        _telegram_bot("tok")

        (kwargs,) = _request_kwargs(request_factory)
        assert kwargs["read_timeout"] == 60.0

    def test_proxy_path_carries_timeout_and_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_senders import _telegram_bot

        _wipe_connection_env(monkeypatch)
        proxy_url = "socks5://127.0.0.1:1080"
        monkeypatch.setenv("TELEGRAM_PROXY", proxy_url)
        request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock(monkeypatch, MagicMock(return_value=MagicMock()), request_factory)

        _telegram_bot("tok")

        kwargs_list = _request_kwargs(request_factory)
        assert len(kwargs_list) == 2, "request + get_updates_request both ride the proxy"
        for kwargs in kwargs_list:
            assert kwargs["proxy"] == proxy_url
            assert kwargs["read_timeout"] == 60.0


class TestStandaloneSenderForwardsExtra:
    """``_standalone_send`` — the out-of-process ``deliver=telegram`` cron path — must forward
    the platform's ``extra`` like the ``send_message`` tool does; otherwise that path keeps
    talking to ``api.telegram.org`` on a self-hosted deployment even though the tool path works."""

    def test_standalone_sender_honours_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.platforms.telegram.adapter import _standalone_send

        _wipe_connection_env(monkeypatch)
        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        _install_telegram_mock(monkeypatch, bot_factory, MagicMock())

        pconfig = SimpleNamespace(token="tok", extra={"base_url": "http://127.0.0.1:8081/bot"})
        result: dict[str, Any] = asyncio.run(
            _standalone_send(pconfig, "123", "hello world")
        )

        assert result["success"] is True
        assert bot_factory.call_args.kwargs.get("base_url") == "http://127.0.0.1:8081/bot"
        bot.send_message.assert_awaited_once()
