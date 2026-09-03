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


def test_strict_text_topic_retries_same_thread_once_and_never_general(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    bot.send_message.side_effect = [
        RuntimeError("Message thread not found"),
        RuntimeError("transport failed after strict retry"),
    ]
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    result = asyncio.run(
        _send_telegram(
            "tok",
            "-100123",
            "private report",
            thread_id="99999",
            topic_boundary="strict",
        )
    )

    assert result["success"] is False
    assert result["raw_response"] == {
        "requested_thread_id": 99999,
        "strict_thread_failure": True,
    }
    assert "transport failed" in result["error"].lower()
    assert bot.send_message.await_count == 2
    assert [
        call.kwargs.get("message_thread_id")
        for call in bot.send_message.await_args_list
    ] == [99999, 99999]


def test_send_to_platform_propagates_strict_topic_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.config import Platform
    import tools.send_message_tool as smt

    sender = AsyncMock(return_value={"success": True, "message_id": "1"})
    monkeypatch.setattr(smt, "_send_telegram", sender)

    result = asyncio.run(
        smt._send_to_platform(
            Platform.TELEGRAM,
            SimpleNamespace(enabled=True, token="tok", extra={}),
            "-100123",
            "private report",
            thread_id="99",
            topic_boundary="strict",
        )
    )

    assert result["success"] is True
    call = sender.await_args
    assert call is not None
    assert call.kwargs["topic_boundary"] == "strict"


def test_strict_media_topic_never_retries_in_general(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    bot.send_photo.side_effect = [
        RuntimeError("Message thread not found"),
        RuntimeError("transport failed after strict retry"),
    ]
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    img = _tmpfile(".png")
    try:
        result = asyncio.run(
            _send_telegram(
                "tok",
                "-100123",
                "",
                media_files=[(img, False)],
                thread_id="99999",
                topic_boundary="strict",
            )
        )

        assert result["success"] is False
        assert result["raw_response"] == {
            "requested_thread_id": 99999,
            "strict_thread_failure": True,
        }
        assert "transport failed" in result["error"].lower()
        assert bot.send_photo.await_count == 2
        assert [
            call.kwargs["message_thread_id"]
            for call in bot.send_photo.await_args_list
        ] == [99999, 99999]
    finally:
        os.unlink(img)


def test_strict_text_success_media_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    bot.send_photo.side_effect = [
        RuntimeError("Message thread not found"),
        RuntimeError("transport failed after strict retry"),
    ]
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    img = _tmpfile(".png")
    try:
        result = asyncio.run(
            _send_telegram(
                "tok",
                "-100123",
                "x" * 1100,
                media_files=[(img, False)],
                thread_id="99999",
                topic_boundary="strict",
            )
        )

        assert bot.send_message.await_count == 1
        assert bot.send_photo.await_count == 2
        assert result["success"] is False
        assert result["raw_response"] == {
            "requested_thread_id": 99999,
            "strict_thread_failure": True,
        }
    finally:
        os.unlink(img)


def test_strict_missing_media_caption_fallback_retries_same_topic_then_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    bot.send_message.side_effect = [
        RuntimeError("Message thread not found"),
        RuntimeError("transport failed after strict retry"),
    ]
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    result = asyncio.run(
        _send_telegram(
            "tok",
            "-100123",
            "caption text",
            media_files=[("/tmp/hermes-strict-missing-photo.png", False)],
            thread_id="99999",
            topic_boundary="strict",
        )
    )

    assert bot.send_message.await_count == 2
    assert [
        call.kwargs["message_thread_id"]
        for call in bot.send_message.await_args_list
    ] == [99999, 99999]
    assert result["success"] is False
    assert result["raw_response"] == {
        "requested_thread_id": 99999,
        "strict_thread_failure": True,
    }


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
