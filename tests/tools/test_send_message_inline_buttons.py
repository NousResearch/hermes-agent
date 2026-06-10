"""The standalone Telegram send path must attach inline keyboard buttons.

Cron deliveries fall back from the live gateway adapter to the standalone
``_send_telegram`` path when the in-process adapter is unreachable. Before
this fix that fallback dropped ``HERMES_INLINE_BUTTONS`` — the message arrived
without any buttons. These tests verify the buttons now survive the fallback.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(monkeypatch, bot_factory):
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)

    def _ikb(text="", callback_data=""):
        return SimpleNamespace(text=text, callback_data=callback_data)

    def _ikm(keyboard):
        return SimpleNamespace(inline_keyboard=keyboard)

    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=lambda **kw: SimpleNamespace(**kw),
        InlineKeyboardButton=_ikb,
        InlineKeyboardMarkup=_ikm,
        constants=constants_mod,
        request=SimpleNamespace(HTTPXRequest=MagicMock()),
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)
    monkeypatch.setitem(sys.modules, "telegram.request", telegram_mod.request)


def _make_bot():
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    return bot


def test_send_telegram_attaches_inline_keyboard(monkeypatch):
    from tools.send_message_tool import _send_telegram

    monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    buttons = [[
        {"text": "✅ Качай", "callback_data": "fam_s03_download"},
        {"text": "❌ Не надо", "callback_data": "fam_s03_cancel"},
    ]]
    asyncio.run(_send_telegram("token", "12345", "Качаю?", inline_buttons=buttons))

    bot.send_message.assert_awaited_once()
    markup = bot.send_message.await_args.kwargs.get("reply_markup")
    assert markup is not None, "reply_markup must be passed when inline_buttons are given"
    row = markup.inline_keyboard[0]
    assert [b.callback_data for b in row] == ["fam_s03_download", "fam_s03_cancel"]


def test_send_telegram_no_buttons_no_markup(monkeypatch):
    from tools.send_message_tool import _send_telegram

    monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    asyncio.run(_send_telegram("token", "12345", "plain message"))

    bot.send_message.assert_awaited_once()
    assert "reply_markup" not in bot.send_message.await_args.kwargs


def test_send_to_platform_parses_marker_from_message(monkeypatch):
    """The send_message tool path: a HERMES_INLINE_BUTTONS marker in the text is
    parsed into buttons and stripped from the visible message."""
    from gateway.config import Platform
    from tools.send_message_tool import _send_to_platform

    monkeypatch.delenv("TELEGRAM_PROXY", raising=False)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    message = (
        "Качаю?\n"
        'HERMES_INLINE_BUTTONS:[[{"text": "✅ Качай", "callback_data": "fam_s03_download"}]]'
    )
    pconfig = SimpleNamespace(token="t", extra={})
    asyncio.run(_send_to_platform(Platform.TELEGRAM, pconfig, "12345", message))

    bot.send_message.assert_awaited_once()
    kwargs = bot.send_message.await_args.kwargs
    markup = kwargs.get("reply_markup")
    assert markup is not None
    assert markup.inline_keyboard[0][0].callback_data == "fam_s03_download"
    # The raw marker must not appear in the delivered text.
    assert "HERMES_INLINE_BUTTONS" not in kwargs.get("text", "")
