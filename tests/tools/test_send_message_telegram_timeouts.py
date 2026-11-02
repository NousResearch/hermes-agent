"""Telegram media_write_timeout configurability on both send paths (#117795).

Two contracts:

1. Gateway path — ``plugins/platforms/telegram/adapter.py::_build_ptb_requests`` must
   honor ``HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT`` exactly like the five sibling
   ``HERMES_TELEGRAM_HTTP_*`` timeouts already do, defaulting to 60.0.
2. Standalone path — ``tools/send_message_senders.py::_telegram_bot`` must construct
   ``HTTPXRequest`` with the same env-overridable timeout set as the gateway path
   (connect/read/write/pool/media_write) on BOTH the proxy and direct branches, instead
   of leaving PTB's 5s/20s defaults in place (multi-MB media uploads time out).
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _install_telegram_mock(
    monkeypatch: pytest.MonkeyPatch,
    bot_factory: MagicMock,
    httpx_request_factory: MagicMock,
) -> None:
    """Stub ``telegram`` + ``telegram.request`` so Bot/HTTPXRequest construction is observable."""
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    request_mod = SimpleNamespace(HTTPXRequest=httpx_request_factory)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=lambda **_kw: SimpleNamespace(**_kw),
        constants=constants_mod,
        request=request_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)
    monkeypatch.setitem(sys.modules, "telegram.request", request_mod)


class TestGatewayMediaWriteTimeout:
    """``_build_ptb_requests`` reads HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT."""

    def _built_kwargs(self, monkeypatch: pytest.MonkeyPatch, **env: str) -> list[dict]:
        for key in ("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", "HERMES_TELEGRAM_DISABLE_FALLBACK_IPS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        from plugins.platforms.telegram import adapter as tg
        built: list[dict] = []

        def fake_httpxrequest(**kwargs):
            built.append(kwargs)
            return SimpleNamespace()

        adapter = tg.TelegramAdapter(_telegram_platform_config())
        with patch.object(tg, "HTTPXRequest", fake_httpxrequest), \
                patch.object(adapter, "_instrument_polling_request", side_effect=lambda r: r), \
                patch.object(adapter, "_fallback_ips", return_value=[]):
            asyncio.run(adapter._build_ptb_requests())
        return built

    def test_env_override_reaches_both_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built = self._built_kwargs(
            monkeypatch, HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT="120.5")
        assert len(built) == 2, "expected the (general, getUpdates) request pair"
        for kwargs in built:
            assert kwargs["media_write_timeout"] == 120.5

    def test_default_is_gateway_constant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built = self._built_kwargs(monkeypatch)
        assert len(built) == 2
        for kwargs in built:
            assert kwargs["media_write_timeout"] == 60.0

    def test_invalid_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built = self._built_kwargs(
            monkeypatch, HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT="not-a-float")
        assert len(built) == 2
        for kwargs in built:
            assert kwargs["media_write_timeout"] == 60.0


def _telegram_platform_config():
    from gateway.config import PlatformConfig
    return PlatformConfig(enabled=True, token="tok")


class TestStandaloneMediaWriteTimeout:
    """``_telegram_bot`` tunes timeouts on both proxy and direct branches."""

    def test_direct_branch_tunes_media_write_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(key, raising=False)
        bot_factory, httpx_factory = MagicMock(), MagicMock()
        _install_telegram_mock(monkeypatch, bot_factory, httpx_factory)
        from tools.send_message_senders import _telegram_bot
        bot = _telegram_bot("tok")
        assert bot is bot_factory.return_value
        assert httpx_factory.call_count >= 1, "direct branch must construct HTTPXRequest for timeouts"
        kwargs = httpx_factory.call_args.kwargs
        assert kwargs["media_write_timeout"] == 60.0, "gateway default must carry over to standalone"
        assert "proxy" not in kwargs

    def test_env_override_reaches_standalone_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", "90.0")
        bot_factory, httpx_factory = MagicMock(), MagicMock()
        _install_telegram_mock(monkeypatch, bot_factory, httpx_factory)
        from tools.send_message_senders import _telegram_bot
        _telegram_bot("tok")
        kwargs = httpx_factory.call_args.kwargs
        assert kwargs["media_write_timeout"] == 90.0

    def test_sibling_timeouts_tuned_on_direct_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", "15.0")
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "30.0")
        bot_factory, httpx_factory = MagicMock(), MagicMock()
        _install_telegram_mock(monkeypatch, bot_factory, httpx_factory)
        from tools.send_message_senders import _telegram_bot
        _telegram_bot("tok")
        kwargs = httpx_factory.call_args.kwargs
        assert kwargs["connect_timeout"] == 15.0
        assert kwargs["read_timeout"] == 30.0
        assert kwargs["write_timeout"] == 20.0
        assert kwargs["pool_timeout"] == 8.0

    def test_proxy_branch_keeps_proxy_and_gains_timeouts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_PROXY", "http://127.0.0.1:18080")
        bot_factory, httpx_factory = MagicMock(), MagicMock()
        _install_telegram_mock(monkeypatch, bot_factory, httpx_factory)
        from tools.send_message_senders import _telegram_bot
        _telegram_bot("tok")
        assert httpx_factory.call_count == 2
        for call in httpx_factory.call_args_list:
            assert call.kwargs["proxy"] == "http://127.0.0.1:18080"
            assert call.kwargs["media_write_timeout"] == 60.0

    def test_import_failure_falls_back_to_plain_bot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail-open: a broken telegram install must still return a usable plain Bot."""
        monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
        bot_factory = MagicMock()
        constants_mod = SimpleNamespace(ParseMode=SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML"))
        request_mod = SimpleNamespace(HTTPXRequest=MagicMock(side_effect=RuntimeError("boom")))
        telegram_mod = SimpleNamespace(Bot=bot_factory, constants=constants_mod, request=request_mod)
        monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
        monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)
        monkeypatch.setitem(sys.modules, "telegram.request", request_mod)
        from tools.send_message_senders import _telegram_bot
        bot = _telegram_bot("tok")
        assert bot is bot_factory.return_value


# The legacy no-proxy contract ("request not in call_kwargs" in
# tests/tools/test_send_message_telegram_proxy.py::test_no_proxy_env_uses_plain_bot)
# is superseded by #117795: the direct branch now also tunes timeouts.
