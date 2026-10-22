"""Standalone Telegram sends honour ``display.platforms.telegram.notifications``.

The gateway adapter silences non-final messages under the default "important" mode
(``_notification_kwargs``), but ``tools/send_message_senders.py::_send_telegram`` (``hermes send``
and the ``send_message`` tool) never set ``disable_notification``, so every standalone message
buzzed regardless of the mode (#131924). The ``telegram`` package is stubbed;
``_resolve_notifications_mode`` on the adapter module is patched so the mode under test is explicit.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _stub_notifications_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr("plugins.platforms.telegram.adapter._resolve_notifications_mode",
                        lambda: mode, raising=False)


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=2))
    return bot


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
        "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr(
        "gateway.platforms.base._detect_macos_system_proxy", lambda: None
    )


def _tmpfile(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"x")
    f.close()
    return f.name


def test_important_mode_silences_text_send(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    _stub_notifications_mode(monkeypatch, "important")
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    res = asyncio.run(_send_telegram("tok", "123", "status: still syncing"))
    assert res["success"] is True
    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["disable_notification"] is True


def test_all_mode_keeps_text_send_audible(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    _stub_notifications_mode(monkeypatch, "all")
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    res = asyncio.run(_send_telegram("tok", "123", "deploy finished"))
    assert res["success"] is True
    bot.send_message.assert_awaited_once()
    assert "disable_notification" not in bot.send_message.await_args.kwargs


def test_important_mode_silences_media_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    _stub_notifications_mode(monkeypatch, "important")
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    img = _tmpfile(".png")
    try:
        res = asyncio.run(_send_telegram("tok", "123", "", media_files=[(img, False)]))
        assert res["success"] is True
        bot.send_photo.assert_awaited_once()
        assert bot.send_photo.await_args.kwargs["disable_notification"] is True
    finally:
        import os
        os.unlink(img)


def test_mode_resolution_failure_fails_over_to_important(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    # Mode resolution blowing up (unreadable config, broken adapter): the sender must fail
    # over to "important" — the same fallback the gateway adapter's _build_adapter applies.
    def _boom():
        raise RuntimeError("unreadable config")

    monkeypatch.setattr("plugins.platforms.telegram.adapter._resolve_notifications_mode",
                        _boom, raising=False)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    res = asyncio.run(_send_telegram("tok", "123", "hello"))
    assert res["success"] is True
    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["disable_notification"] is True
