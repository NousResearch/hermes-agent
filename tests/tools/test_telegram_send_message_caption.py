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
    class _InputFile:
        def __init__(self, obj, filename=None, attach=False):
            self.obj = obj
            self.filename = filename
            self.attach = attach

    class _InputMediaPhoto:
        def __init__(self, media):
            self.media = media

    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    _MessageEntity = lambda **_kw: SimpleNamespace(**_kw)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        InputFile=_InputFile,
        InputMediaPhoto=_InputMediaPhoto,
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
    # Neutralize macOS system-proxy auto-detection at its probe rather than by
    # claiming the host is Linux: this keeps the test honest on the macOS
    # runner (and on a developer's Mac), where a real scutil-configured proxy
    # would otherwise leak into the assertion.
    monkeypatch.setattr(
        "gateway.platforms.base._detect_macos_system_proxy", lambda: None
    )


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


def test_multi_file_keeps_separate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-image batches go as ONE native album (sendMediaGroup, no per-photo
    captions); the ambiguous caption→file association keeps the text a separate
    message."""
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    bot.send_media_group = AsyncMock(return_value=[
        SimpleNamespace(message_id=10, media_group_id="group-1"),
        SimpleNamespace(message_id=11, media_group_id="group-1"),
    ])
    img = _tmpfile(".png")
    img2 = _tmpfile(".jpg")
    try:
        res = asyncio.run(
            _send_telegram("tok", "123", "two pics", media_files=[(img, False), (img2, False)])
        )
        assert res["success"] is True
        # Ambiguous caption→file association: text stays a separate message.
        bot.send_message.assert_awaited()
        # One native album, not two individual sendPhoto calls.
        bot.send_media_group.assert_awaited_once()
        bot.send_photo.assert_not_awaited()
        for item in bot.send_media_group.await_args.kwargs["media"]:
            assert not item.media.filename.startswith("caption")
    finally:
        os.unlink(img)
        os.unlink(img2)
