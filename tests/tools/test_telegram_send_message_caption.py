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


def test_document_send_uses_basename_not_full_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_document receives filename=os.path.basename(media_path), not the full host path."""
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    # Create a temp file with a long path that would exceed Telegram's filename limit
    # if the full path were used as the multipart filename.
    with tempfile.TemporaryDirectory(prefix="very_long_directory_name_that_exceeds_limits_") as tmpdir:
        long_filename = "A" * 50 + ".pdf"  # 50+ char basename
        media_path = os.path.join(tmpdir, long_filename)
        with open(media_path, "wb") as f:
            f.write(b"dummy content")

        try:
            res = asyncio.run(
                _send_telegram("tok", "123", "Here is the doc", media_files=[(media_path, False)])
            )
            assert res["success"] is True
            # send_document should be called with filename=basename only
            bot.send_document.assert_awaited_once()
            call_kwargs = bot.send_document.await_args.kwargs
            assert "filename" in call_kwargs
            assert call_kwargs["filename"] == long_filename
            assert call_kwargs["filename"] != media_path  # full path NOT leaked
        finally:
            pass  # temp dir auto-cleans


def test_document_retry_thread_not_found_uses_basename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry path (thread not found) also passes filename=basename."""
    from tools.send_message_tool import _send_telegram, _is_telegram_thread_not_found

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    # Make first send_document raise "thread not found", second succeed
    call_count = {"n": 0}

    async def failing_then_ok(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate "thread not found" error
            raise Exception("Bad Request: thread not found")
        return SimpleNamespace(message_id=99)

    bot.send_document = AsyncMock(side_effect=failing_then_ok)

    with tempfile.TemporaryDirectory(prefix="long_path_") as tmpdir:
        fname = "B" * 60 + ".docx"
        media_path = os.path.join(tmpdir, fname)
        with open(media_path, "wb") as f:
            f.write(b"x")

        try:
            res = asyncio.run(
                _send_telegram("tok", "123", "caption", media_files=[(media_path, False)], thread_id="123")
            )
            assert res["success"] is True
            # Two calls: first fails, second succeeds (retry without thread_id)
            assert bot.send_document.await_count == 2
            # Both calls should have filename=basename
            for call in bot.send_document.await_args_list:
                assert call.kwargs.get("filename") == fname
                assert call.kwargs.get("filename") != media_path
        finally:
            pass


def test_document_retry_caption_parse_failure_uses_basename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry path (caption parse failure) also passes filename=basename."""
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    # First call fails with "can't parse entities" (caption parse error), second succeeds
    call_count = {"n": 0}

    async def failing_then_ok(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("Bad Request: can't parse entities: Character '$' is reserved")
        return SimpleNamespace(message_id=99)

    bot.send_document = AsyncMock(side_effect=failing_then_ok)

    with tempfile.TemporaryDirectory(prefix="long_path_") as tmpdir:
        fname = "C" * 55 + ".txt"
        media_path = os.path.join(tmpdir, fname)
        with open(media_path, "wb") as f:
            f.write(b"x")

        try:
            res = asyncio.run(
                _send_telegram("tok", "123", "Price: $100", media_files=[(media_path, False)])
            )
            assert res["success"] is True
            # Two calls: first fails, second succeeds (retry with parse_mode=None)
            assert bot.send_document.await_count == 2
            # Both calls should have filename=basename
            for call in bot.send_document.await_args_list:
                assert call.kwargs.get("filename") == fname
                assert call.kwargs.get("filename") != media_path
        finally:
            pass