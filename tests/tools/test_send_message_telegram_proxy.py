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

    def test_no_proxy_env_uses_plain_bot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without TELEGRAM_PROXY (and no inherited HTTPS_PROXY/etc), Bot()
        is constructed plainly — no ``request``/``get_updates_request``
        kwargs, and HTTPXRequest is not invoked at all.
        """
        from tools.send_message_tool import _send_telegram

        # Wipe every env var resolve_proxy_url() inspects so the host's
        # ambient proxy settings can't flip this test green-or-red.
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
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
        # Make sure macOS system-proxy auto-detection (scutil) can't kick in.
        monkeypatch.setattr(sys, "platform", "linux")

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
        call_args = bot_factory.call_args.args
        # token may be passed positionally or as a kwarg; either is fine.
        assert call_kwargs.get("token", call_args[0] if call_args else None) == "tok"
        assert "request" not in call_kwargs
        assert "get_updates_request" not in call_kwargs
        httpx_request_factory.assert_not_called()
        bot.send_message.assert_awaited_once()


class TestTelegramStandaloneLazyDeps:
    """The standalone Telegram send path should lazy-install missing SDK deps."""

    def test_missing_telegram_sdk_invokes_lazy_deps_then_reimports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the initial Telegram import fails, call lazy_deps.ensure for
        platform.telegram and retry the import instead of returning a manual
        pip-install error immediately.
        """
        import builtins
        from tools import send_message_tool

        for name in list(sys.modules):
            if name == "telegram" or name.startswith("telegram."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        bot_factory = MagicMock()
        httpx_request_factory = MagicMock()
        installed = False
        ensure_calls: list[tuple[str, bool]] = []

        def fake_ensure(feature: str, prompt: bool = True) -> None:
            nonlocal installed
            ensure_calls.append((feature, prompt))
            installed = True
            _install_telegram_mock_with_request(
                monkeypatch,
                bot_factory,
                httpx_request_factory,
            )

        monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)

        original_import = builtins.__import__

        def fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
            if (name == "telegram" or name.startswith("telegram.")) and not installed:
                raise ImportError("No module named 'telegram'")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        Bot, ParseMode = send_message_tool._import_telegram_send_deps()

        assert ensure_calls == [("platform.telegram", False)]
        assert Bot is bot_factory
        assert ParseMode.MARKDOWN_V2 == "MarkdownV2"
