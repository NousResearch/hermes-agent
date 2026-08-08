"""Regression tests for the standalone Telegram send path's HTTP timeouts.

The ``send_message`` tool, when invoked from a process *other than* the
gateway (agent / TUI / cron / ``hermes send``), runs ``_send_telegram``
directly with a ``telegram.Bot`` it constructs itself. Before the fix that
accompanies these tests, that Bot used PTB's default ``HTTPXRequest``
timeouts (5s across the board). Telegram can hold sendDocument/sendPhoto
responses for 30s+ when a bot is under flood control, so every standalone
media send failed with ``Timed out`` even though the upload itself
succeeded server-side.

These tests verify that the standalone path now honours the same
``HERMES_TELEGRAM_HTTP_*`` env knobs as the gateway adapter
(``plugins/platforms/telegram/adapter.py``) and matches its defaults.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.tools.test_send_message_telegram_proxy import (
    _install_telegram_mock_with_request,
    _make_bot,
)

_TIMEOUT_ENV = {
    "HERMES_TELEGRAM_HTTP_POOL_TIMEOUT": "pool_timeout",
    "HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT": "connect_timeout",
    "HERMES_TELEGRAM_HTTP_READ_TIMEOUT": "read_timeout",
    "HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT": "write_timeout",
}

_ADAPTER_DEFAULTS = {
    "pool_timeout": 8.0,
    "connect_timeout": 10.0,
    "read_timeout": 20.0,
    "write_timeout": 20.0,
}


def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe every env var resolve_proxy_url() inspects, and disable macOS
    system-proxy auto-detection, so ambient host settings can't change
    which Bot() branch runs.
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
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")


class TestSendTelegramStandaloneTimeouts:
    """The standalone ``_send_telegram`` path must construct its Bot with
    adapter-parity HTTP timeouts instead of PTB's 5s defaults.
    """

    def _run_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[MagicMock, MagicMock]:
        from tools.send_message_tool import _send_telegram

        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
        bot = _make_bot()
        bot_factory = MagicMock(return_value=bot)
        httpx_request_factory = MagicMock(side_effect=lambda **kw: MagicMock(_kw=kw))
        _install_telegram_mock_with_request(monkeypatch, bot_factory, httpx_request_factory)

        result: dict[str, Any] = asyncio.run(_send_telegram("tok", "123", "hello"))
        assert result["success"] is True
        return bot_factory, httpx_request_factory

    def test_direct_branch_uses_adapter_default_timeouts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without any env overrides, HTTPXRequest gets the gateway
        adapter's defaults (pool 8 / connect 10 / read 20 / write 20).
        """
        _clear_proxy_env(monkeypatch)
        for var in _TIMEOUT_ENV:
            monkeypatch.delenv(var, raising=False)

        bot_factory, httpx_request_factory = self._run_send(monkeypatch)

        assert "request" in bot_factory.call_args.kwargs
        assert httpx_request_factory.call_count == 1
        kw = httpx_request_factory.call_args.kwargs
        for name, expected in _ADAPTER_DEFAULTS.items():
            assert kw.get(name) == expected, (
                f"{name}: expected adapter default {expected}, got {kw.get(name)!r}"
            )

    def test_direct_branch_honours_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HERMES_TELEGRAM_HTTP_* env vars override each timeout, exactly
        as they do for the in-gateway adapter.
        """
        _clear_proxy_env(monkeypatch)
        for i, var in enumerate(_TIMEOUT_ENV):
            monkeypatch.setenv(var, str(100.0 + i))

        _bot_factory, httpx_request_factory = self._run_send(monkeypatch)

        kw = httpx_request_factory.call_args.kwargs
        for i, (var, kwarg) in enumerate(_TIMEOUT_ENV.items()):
            assert kw.get(kwarg) == 100.0 + i, (
                f"{var} not honoured: expected {100.0 + i}, got {kw.get(kwarg)!r}"
            )

    def test_direct_branch_ignores_malformed_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-numeric env value falls back to the default instead of
        crashing the send.
        """
        _clear_proxy_env(monkeypatch)
        for var in _TIMEOUT_ENV:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "not-a-number")

        _bot_factory, httpx_request_factory = self._run_send(monkeypatch)

        kw = httpx_request_factory.call_args.kwargs
        assert kw.get("read_timeout") == _ADAPTER_DEFAULTS["read_timeout"]

    def test_proxy_branch_carries_timeouts_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With TELEGRAM_PROXY set, both HTTPXRequest instances carry the
        proxy *and* the tuned timeouts.
        """
        proxy_url = "socks5://127.0.0.1:1080"
        _clear_proxy_env(monkeypatch)
        monkeypatch.setenv("TELEGRAM_PROXY", proxy_url)
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "120")

        _bot_factory, httpx_request_factory = self._run_send(monkeypatch)

        assert httpx_request_factory.call_count == 2
        for call in httpx_request_factory.call_args_list:
            assert call.kwargs.get("proxy") == proxy_url
            assert call.kwargs.get("read_timeout") == 120.0
            assert call.kwargs.get("connect_timeout") == _ADAPTER_DEFAULTS["connect_timeout"]
