"""Regression coverage for Telegram document filenames (issue #67552)."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from tools.send_message_tool import _send_telegram


def _fake_bot():
    bot = SimpleNamespace()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_photo = AsyncMock()
    bot.send_video = AsyncMock()
    bot.send_voice = AsyncMock()
    bot.send_audio = AsyncMock()
    bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=2))
    return bot


def _install_telegram_mock(monkeypatch, bot):
    """Keep the reproduction independent of Telegram's network/client setup."""
    telegram_mod = ModuleType("telegram")
    telegram_mod.Bot = lambda token, **kwargs: bot
    constants_mod = ModuleType("telegram.constants")
    constants_mod.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    telegram_mod.constants = constants_mod
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def test_document_send_explicitly_uses_long_basename(tmp_path, monkeypatch):
    """The standalone sender must not expose the source path as a filename."""
    filename = "AceBill_Product_Guide_RC2_QA_Review_v1.0.2.docx"
    document = tmp_path / "nested" / filename
    document.parent.mkdir()
    document.write_bytes(b"test document")
    bot = _fake_bot()
    _install_telegram_mock(monkeypatch, bot)

    result = asyncio.run(
        _send_telegram(
            "fake-token",
            "12345",
            "",
            media_files=[(str(document), False)],
        )
    )

    assert result["success"] is True
    kwargs = bot.send_document.await_args.kwargs
    assert kwargs["filename"] == filename


@pytest.mark.parametrize(
    ("first_error", "message", "thread_id"),
    [
        (RuntimeError("Message thread not found"), "", "17585"),
        (RuntimeError("Can't parse caption entities"), "Document caption", None),
    ],
)
def test_document_retries_keep_long_basename(
    tmp_path, monkeypatch, first_error, message, thread_id
):
    """Both the thread and caption retry paths retain the safe filename."""
    filename = "AceBill_Product_Guide_RC2_QA_Review_v1.0.2.docx"
    document = tmp_path / filename
    document.write_bytes(b"test document")
    bot = _fake_bot()
    bot.send_document.side_effect = [
        first_error,
        SimpleNamespace(message_id=2),
    ]
    _install_telegram_mock(monkeypatch, bot)

    result = asyncio.run(
        _send_telegram(
            "fake-token",
            "12345",
            message,
            media_files=[(str(document), False)],
            thread_id=thread_id,
        )
    )

    assert result["success"] is True
    assert bot.send_document.await_count == 2
    for call in bot.send_document.await_args_list:
        assert call.kwargs["filename"] == filename
