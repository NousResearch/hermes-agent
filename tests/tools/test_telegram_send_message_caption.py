"""Standalone Telegram MEDIA:<path> caption delivery.

When `hermes send --to telegram "MEDIA:/x.png This Caption"` carries a single
captionable file plus short text, the text must ride on the media bubble as the
sendPhoto/sendVideo/sendDocument ``caption`` rather than being posted as a
separate sendMessage beforehand. Longer text (> Telegram's 1024 caption cap)
falls back to a separate message. The ``telegram`` package is stubbed.

Also covers the failure paths that keep the caption contract observable (#69252):
when the caption-fallback resend for a missing media file fails, its sanitized,
operation-prefixed error must surface in the result warnings; and when the
caption parse of an audio bubble fails, its retry must keep the same media route
(``send_audio`` for mp3/m4a), never a ``send_document`` downgrade. These live
apart from ``test_send_message_tool.py`` because that module skips wholesale
where ``python-telegram-bot`` is not installed (canonical CI installs no
messaging extras); with the stubbed SDK these regressions run in every
environment.
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


def test_caption_fallback_error_surfaces_in_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """(#69252) A short caption whose only media file is missing suppresses the text send and
    retries delivery on its own; when that fallback ALSO fails, the sanitized error must land
    in the result warnings with its operation prefix — a bare transport string is
    indistinguishable from any other media warning, and raw internals must stay redacted."""
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    async def retry_raises(*args, **kwargs):
        raise Exception('connection refused access_token="sk-live-secret-123"')

    monkeypatch.setattr("tools.send_message_senders._send_telegram_message_with_retry", retry_raises)

    res = asyncio.run(
        _send_telegram("tok", "123", "Caption for missing media",
                       media_files=[(str(tmp_path / "nonexistent.png"), False)])
    )

    # The fallback is the only delivery attempt, so nothing was delivered.
    assert "error" in res
    warnings = res.get("warnings", [])
    assert any(
        w.startswith("Telegram caption-fallback send failed for missing media: ")
        and "connection refused" in w
        for w in warnings
    ), f"expected the operation-prefixed fallback error in warnings, got {warnings}"
    # The secret in the transport error is redacted: sanitization, not a raw passthrough.
    assert all("sk-live-secret-123" not in repr(w) for w in warnings)


def test_audio_caption_parse_retry_keeps_send_audio(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """(#69252) When the caption parse of an audio bubble fails, the retry must retain the
    media route — ``send_audio`` for mp3/m4a — never downgrading to ``send_document``. The
    retry lives centrally in ``_telegram_send_one_media``, which re-sends with the same
    extension routing minus the broken parse mode."""
    from tools.send_message_senders import _telegram_send_one_media

    audio = tmp_path / "voice-note.mp3"
    audio.write_bytes(b"ID3\x03\x00" + b"\x00" * 8)
    bot = _make_bot()
    bot.send_audio = AsyncMock(side_effect=[
        Exception("can't parse entities: unclosed tag"),  # first bubble: caption parse error
        SimpleNamespace(message_id=7),  # plain-caption retry succeeds
    ])
    bot.send_document = AsyncMock()

    sent = asyncio.run(
        _telegram_send_one_media(
            bot, "123", str(audio), False,
            caption="The Caption", parse_mode="MarkdownV2", has_html=False,
            thread_kwargs={}, force_document=False)
    )

    assert sent.message_id == 7
    # Same route on the retry; no document downgrade.
    assert bot.send_audio.await_count == 2
    bot.send_document.assert_not_awaited()
    first_call, retry_call = bot.send_audio.await_args_list
    assert first_call.kwargs.get("parse_mode") == "MarkdownV2"
    # The retry drops the parse mode and keeps the plain caption.
    assert "parse_mode" not in retry_call.kwargs
    assert retry_call.kwargs.get("caption") == "The Caption"
