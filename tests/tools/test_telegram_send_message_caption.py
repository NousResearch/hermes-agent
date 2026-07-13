"""Standalone Telegram MEDIA:<path> caption delivery.

When `hermes send --to telegram "MEDIA:/x.png This Caption"` carries a single
captionable file plus short text, the text must ride on the media bubble as the
sendPhoto/sendVideo/sendDocument ``caption`` rather than being posted as a
separate sendMessage beforehand. Longer text (> Telegram's 1024 caption cap)
falls back to a separate message. The ``telegram`` package is stubbed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    _MessageEntity = lambda **_kw: SimpleNamespace(**_kw)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=_MessageEntity,
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=2))
    bot.send_video = AsyncMock(return_value=SimpleNamespace(message_id=3))
    bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=4))
    return bot


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
        "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")


def _tmpfile(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"x")
    f.close()
    return f.name


def test_image_caption_rides_bubble_no_separate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    img = _tmpfile(".png")
    try:
        res = asyncio.run(
            _send_telegram("tok", "123", "This Caption", media_files=[(img, False)])
        )
        assert res["success"] is True
        # No separate text message; caption rides the photo.
        bot.send_message.assert_not_awaited()
        bot.send_photo.assert_awaited_once()
        assert bot.send_photo.await_args.kwargs.get("caption") == "This Caption"
    finally:
        os.unlink(img)


def test_video_caption_rides_bubble(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    vid = _tmpfile(".mp4")
    try:
        res = asyncio.run(
            _send_telegram("tok", "123", "Model unit tour", media_files=[(vid, False)])
        )
        assert res["success"] is True
        bot.send_message.assert_not_awaited()
        bot.send_video.assert_awaited_once()
        assert bot.send_video.await_args.kwargs.get("caption") == "Model unit tour"
    finally:
        os.unlink(vid)


def test_long_text_falls_back_to_separate_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    img = _tmpfile(".png")
    long_text = "x" * 1100  # over Telegram's 1024 caption cap
    try:
        res = asyncio.run(
            _send_telegram("tok", "123", long_text, media_files=[(img, False)])
        )
        assert res["success"] is True
        # Text too long for a caption — sent as its own message, photo uncaptioned.
        bot.send_message.assert_awaited()
        bot.send_photo.assert_awaited_once()
        assert not bot.send_photo.await_args.kwargs.get("caption")
    finally:
        os.unlink(img)


def test_multi_file_keeps_separate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    img = _tmpfile(".png")
    img2 = _tmpfile(".jpg")
    try:
        res = asyncio.run(
            _send_telegram("tok", "123", "two pics", media_files=[(img, False), (img2, False)])
        )
        assert res["success"] is True
        # Ambiguous caption→file association: text stays a separate message.
        bot.send_message.assert_awaited()
        assert bot.send_photo.await_count == 2
        for call in bot.send_photo.await_args_list:
            assert not call.kwargs.get("caption")
    finally:
        os.unlink(img)
        os.unlink(img2)
