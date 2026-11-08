"""Regression tests for the standalone Telegram send path's proxy support.

The ``send_message`` tool, when invoked from a process *other than* the
gateway (agent / TUI / cron), runs ``_send_telegram`` directly instead of
delegating to the in-process gateway adapter. Before the fix that
accompanies these tests, that standalone path constructed
``telegram.Bot(token=...)`` with no proxy, so in regions where
api.telegram.org is blocked (e.g. RU) the send would just time out with
``Telegram send failed: Timed out`` and never show up in ``gateway.log``.

These tests verify that the standalone path now honours ``TELEGRAM_PROXY``
the same way the gateway adapter (and the Discord standalone path) do.
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
    httpx_request_factory: MagicMock,
) -> None:
    """Install a stub ``telegram`` package whose ``Bot`` and
    ``telegram.request.HTTPXRequest`` are the supplied mocks.

    Mirrors ``_install_telegram_mock`` in test_send_message_tool.py but also
    provides the ``telegram.request`` submodule that the proxy branch needs.
    """
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    request_mod = SimpleNamespace(HTTPXRequest=httpx_request_factory)
    # MessageEntity needed by #27865 mention-detection path.
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


class TestSendTelegramStandaloneProxy:
    """The standalone ``_send_telegram`` path must route through
    ``TELEGRAM_PROXY`` when one is configured, even when no in-process
    gateway runner is available.
    """

    def test_proxy_env_passed_to_httpx_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With TELEGRAM_PROXY set, Bot() is constructed with HTTPXRequest
        instances whose ``proxy=`` kwarg is the configured URL — applied to
        both ``request`` and ``get_updates_request``.
        """
        from tools.send_message_tool import _send_telegram

        proxy_url = "socks5://127.0.0.1:1080"
        monkeypatch.setenv("TELEGRAM_PROXY", proxy_url)
        # Clear NO_PROXY so resolve_proxy_url() doesn't short-circuit on
        # leftover env from the host running the tests.
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        # Ensure the test does not depend on the in-process gateway runner.
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock_with_request(monkeypatch, bot_factory, httpx_request_factory)

        result: dict[str, Any] = asyncio.run(
            _send_telegram("tok", "123", "hello world")
        )

        assert result["success"] is True
        bot_factory.assert_called_once()
        call_kwargs = bot_factory.call_args.kwargs
        assert call_kwargs.get("token") == "tok"
        assert "request" in call_kwargs, "request= kwarg missing — proxy not wired"
        assert "get_updates_request" in call_kwargs, (
            "get_updates_request= kwarg missing — proxy not wired"
        )

        # HTTPXRequest must have been invoked twice, both times with the
        # resolved proxy URL.
        assert httpx_request_factory.call_count == 2
        for call in httpx_request_factory.call_args_list:
            assert call.kwargs.get("proxy") == proxy_url, (
                f"HTTPXRequest called without proxy={proxy_url!r}: {call.kwargs!r}"
            )

        # And the bot was actually used to send.
        bot.send_message.assert_awaited_once()

    def test_no_proxy_env_uses_adapter_timeouts_and_fallback_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a proxy the standalone Bot gets the gateway adapter's request shape: its
        HERMES_TELEGRAM_HTTP_* timeouts (not PTB's 5s/1s/20s defaults) and the fallback-IP
        transport, so a blocked api.telegram.org DNS path fails over like the gateway does.
        """
        from plugins.platforms.telegram.telegram_network import SEED_FALLBACK_IPS, TelegramFallbackTransport
        from tools.send_message_tool import _send_telegram

        for var in (
            "TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
            "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
            "TELEGRAM_FALLBACK_IPS", "HERMES_TELEGRAM_DISABLE_FALLBACK_IPS",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "33")
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
        assert "base_url" not in bot_factory.call_args.kwargs
        req = httpx_request_factory.call_args.kwargs
        assert (req["connect_timeout"], req["read_timeout"], req["write_timeout"], req["pool_timeout"],
                req["media_write_timeout"]) == (10.0, 33.0, 20.0, 8.0, 60.0)
        transport = req["httpx_kwargs"]["transport"]
        assert isinstance(transport, TelegramFallbackTransport)
        assert transport._fallback_ips == list(SEED_FALLBACK_IPS)
        bot.send_message.assert_awaited_once()


def test_send_to_platform_honours_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """``platforms.telegram.extra.base_url`` reaches the standalone Bot through the production
    dispatch (``_send_to_platform``), so cron / ``hermes send`` / send_message use the same
    self-hosted or compatible Bot API server as the gateway, never api.telegram.org."""
    from gateway.config import Platform, PlatformConfig
    from tools.send_message_tool import _send_to_platform

    monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    bot = _make_bot()
    bot_factory = MagicMock(return_value=bot)
    httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
    _install_telegram_mock_with_request(monkeypatch, bot_factory, httpx_request_factory)
    pconfig = PlatformConfig(enabled=True, token="tok", extra={"base_url": "https://botapi.example/bot"})

    result = asyncio.run(_send_to_platform(Platform.TELEGRAM, pconfig, "123", "hello"))

    assert result["success"] is True
    kwargs = bot_factory.call_args.kwargs
    assert (kwargs["base_url"], kwargs["base_file_url"]) == ("https://botapi.example/bot",) * 2
    # A custom server is reached directly: the api.telegram.org fallback-IP transport stays off.
    assert "transport" not in httpx_request_factory.call_args.kwargs["httpx_kwargs"]
    bot.send_message.assert_awaited_once()


def test_local_bot_api_server_honours_no_proxy_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited HTTPS_PROXY with ``NO_PROXY=127.0.0.1`` reaches a local ``telegram-bot-api``
    (``base_url: http://127.0.0.1:8081/bot``) directly: a remote proxy cannot reach it."""
    from gateway.config import Platform, PlatformConfig
    from tools.send_message_tool import _send_to_platform

    monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr("gateway.platforms.base.gateway_trust_env", lambda: True)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    bot = _make_bot()
    httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
    _install_telegram_mock_with_request(monkeypatch, MagicMock(return_value=bot), httpx_request_factory)
    pconfig = PlatformConfig(enabled=True, token="tok", extra={"base_url": "http://127.0.0.1:8081/bot"})

    assert asyncio.run(_send_to_platform(Platform.TELEGRAM, pconfig, "123", "hello"))["success"] is True
    assert all(call.kwargs.get("proxy") is None for call in httpx_request_factory.call_args_list)

    monkeypatch.setenv("NO_PROXY", "localhost")  # the bypass no longer matches: the proxy is used
    asyncio.run(_send_to_platform(Platform.TELEGRAM, pconfig, "123", "hello"))
    assert httpx_request_factory.call_args.kwargs.get("proxy") == "http://proxy.example:3128"


def test_standalone_requests_match_the_gateway_adapter_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone Bot gets the gateway's pool size, keepalive limits and TLS context, not a
    hand-copied subset (HERMES_TELEGRAM_HTTP_POOL_SIZE, platform keepalive, platform_ssl_context)."""
    import ssl
    from tools.send_message_tool import _send_telegram

    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_POOL_SIZE", "64")
    shared = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr("agent.ssl_verify.platform_ssl_context", lambda: shared)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    bot = _make_bot()
    bot_factory = MagicMock(return_value=bot)
    httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
    _install_telegram_mock_with_request(monkeypatch, bot_factory, httpx_request_factory)

    extra = {"base_url": "https://botapi.example/bot"}
    assert asyncio.run(_send_telegram("tok", "123", "hello", extra=extra))["success"] is True

    general, updates = (call.kwargs for call in httpx_request_factory.call_args_list)
    assert general["connection_pool_size"] == 64
    assert general["httpx_kwargs"]["verify"] is shared
    assert general["httpx_kwargs"]["limits"].max_connections == 64
    assert updates["httpx_kwargs"]["limits"].max_keepalive_connections == 0
