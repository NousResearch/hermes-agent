"""The standalone Telegram sender (``hermes send``, the ``send_message`` tool, standalone cron
delivery) makes the gateway adapter's outbound decisions for the same text: the push-notification
mode, the invisible-only guard, the ``rich_messages`` fast-path, and a MarkdownV2 retry before
plain text when Bot API HTML rejects a tag. The ``telegram`` package is stubbed; the bot records
every call, so the assertions read what would reach the Bot API."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class BadRequest(Exception):
    """python-telegram-bot's ``BadRequest`` (matched by class name when PTB is absent)."""


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(Bot=MagicMock(return_value=bot), MessageEntity=lambda **kw: SimpleNamespace(**kw),
                                   InputMediaPhoto=lambda **kw: SimpleNamespace(**kw), constants=constants_mod)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


@pytest.fixture
def standalone(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """A temp HERMES_HOME with ``config.yaml`` content of the test's choosing, plus a recording bot."""
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY",
                "all_proxy", "HERMES_TELEGRAM_NOTIFICATIONS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=8))
    bot.do_api_request = AsyncMock(return_value={"message_id": 9})
    _install_telegram_mock(monkeypatch, bot)

    def run(message, *, config="", extra=None, notify=None, media_files=None):
        from gateway.config import Platform, PlatformConfig
        from tools.send_message_tool import _send_to_platform
        (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
        pconfig = PlatformConfig(enabled=True, token="tok", extra=extra or {})
        return asyncio.run(_send_to_platform(Platform.TELEGRAM, pconfig, "123", message, notify=notify,
                                             media_files=media_files))
    return SimpleNamespace(bot=bot, run=run, home=tmp_path)


def test_notifications_mode_reaches_every_standalone_send(standalone) -> None:
    """``important`` (default) sends silently, text and media alike; ``notify`` rings (cron's
    ``cron.delivery.notify``); ``display.platforms.telegram.notifications: all`` in config.yaml rings."""
    doc = standalone.home / "report.txt"
    doc.write_text("x", encoding="utf-8")
    assert standalone.run("status: still syncing")["success"] is True
    assert standalone.bot.send_message.await_args.kwargs["disable_notification"] is True
    assert standalone.run("report attached", media_files=[(str(doc), False)])["success"] is True
    assert standalone.bot.send_document.await_args.kwargs["disable_notification"] is True

    standalone.run("daily brief", notify=True)
    assert "disable_notification" not in standalone.bot.send_message.await_args.kwargs

    standalone.run("deploy finished", config="display:\n  platforms:\n    telegram:\n      notifications: all\n")
    assert "disable_notification" not in standalone.bot.send_message.await_args.kwargs


def test_invisible_only_text_is_never_delivered(standalone) -> None:
    result = standalone.run("\u200b\u2060\ufeff \u00ad")
    assert "error" in result
    standalone.bot.send_message.assert_not_awaited()


def test_rich_messages_opt_in_sends_tables_through_send_rich_message(standalone) -> None:
    table = "| Service | Port |\n|---|---:|\n| oMLX | 8000 |"
    result = standalone.run(table, extra={"rich_messages": True})
    assert result == {"success": True, "platform": "telegram", "chat_id": "123", "message_id": "9"}
    method, = standalone.bot.do_api_request.await_args.args
    payload = standalone.bot.do_api_request.await_args.kwargs["api_kwargs"]
    assert (method, payload["rich_message"]["markdown"], payload["disable_notification"]) == (
        "sendRichMessage", table, True)
    standalone.bot.send_message.assert_not_awaited()

    standalone.run(table)  # rich_messages off (the default): legacy MarkdownV2, table flattened
    assert standalone.bot.do_api_request.await_count == 1
    assert standalone.bot.send_message.await_args.kwargs["parse_mode"] == "MarkdownV2"


def test_rejected_html_retries_as_markdown_before_plain_text(standalone) -> None:
    """One unsupported tag beside a supported one fails the HTML parse; the Markdown still renders."""
    async def send_message(**kwargs):
        if kwargs.get("parse_mode") == "HTML":
            raise BadRequest("Bad Request: can't parse entities: unsupported start tag \"details\"")
        return SimpleNamespace(message_id=7)
    standalone.bot.send_message = AsyncMock(side_effect=send_message)

    result = standalone.run("<b>Quiz</b>\n\n- **one**\n\n<details><summary>Answers</summary>a</details>")

    assert result["success"] is True
    modes = [c.kwargs["parse_mode"] for c in standalone.bot.send_message.await_args_list]
    assert modes == ["HTML", "MarkdownV2"]
    assert "*one*" in standalone.bot.send_message.await_args.kwargs["text"]


def test_photo_batches_go_as_albums_and_mixed_batches_stay_per_file(standalone) -> None:
    """Two or more still photos ride ``sendMediaGroup`` (10 per album, in order); a batch with a
    non-photo keeps one upload per file."""
    standalone.bot.send_media_group = AsyncMock(return_value=[SimpleNamespace(message_id=11)])
    standalone.bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=12))
    photos = []
    for i in range(12):
        photo = standalone.home / f"p{i}.png"
        photo.write_bytes(b"\x89PNG\r\n\x1a\n")
        photos.append((str(photo), False))

    assert standalone.run("", media_files=photos)["success"] is True
    albums = [call.kwargs for call in standalone.bot.send_media_group.await_args_list]
    assert [len(kw["media"]) for kw in albums] == [10, 2]
    assert all(kw["disable_notification"] is True for kw in albums)
    standalone.bot.send_photo.assert_not_awaited()

    doc = standalone.home / "notes.txt"
    doc.write_text("x", encoding="utf-8")
    assert standalone.run("", media_files=photos[:2] + [(str(doc), False)])["success"] is True
    assert standalone.bot.send_media_group.await_count == 2
    assert standalone.bot.send_photo.await_count == 2
