"""Standalone send_message coverage for Telegram Web App buttons."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(
    monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock
) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)

    class WebAppInfo:
        def __init__(self, *, url: str):
            self.url = url

    class InlineKeyboardButton:
        def __init__(self, text: str, *, web_app=None):
            self.text = text
            self.web_app = web_app

    class InlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        InlineKeyboardButton=InlineKeyboardButton,
        InlineKeyboardMarkup=InlineKeyboardMarkup,
        WebAppInfo=WebAppInfo,
        MessageEntity=lambda **kwargs: SimpleNamespace(**kwargs),
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def test_standalone_send_attaches_button_after_private_chat_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.send_message_tool import _send_telegram

    bot = MagicMock()
    bot.get_chat = AsyncMock(return_value=SimpleNamespace(type="private"))
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    bot_factory = MagicMock(return_value=bot)
    _install_telegram_mock(monkeypatch, bot_factory)

    for name in (
        "TELEGRAM_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    result = asyncio.run(
        _send_telegram(
            "token",
            "12345",
            "Open the generated UI",
            web_app_button={
                "label": "Open UI",
                "url": "https://example.com/a/opaque-id",
            },
        )
    )

    assert result["success"] is True
    bot.get_chat.assert_awaited_once_with(12345)
    kwargs = bot.send_message.await_args.kwargs
    button = kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.text == "Open UI"
    assert button.web_app.url == "https://example.com/a/opaque-id"


def test_standalone_send_rejects_button_for_non_private_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.send_message_tool import _send_telegram

    bot = MagicMock()
    bot.get_chat = AsyncMock(return_value=SimpleNamespace(type="supergroup"))
    bot.send_message = AsyncMock()
    bot_factory = MagicMock(return_value=bot)
    _install_telegram_mock(monkeypatch, bot_factory)

    for name in (
        "TELEGRAM_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    result = asyncio.run(
        _send_telegram(
            "token",
            "-100123",
            "Open the app",
            web_app_button={
                "label": "Open app",
                "url": "https://example.com/app",
            },
        )
    )

    assert "only available in private chats" in result["error"]
    bot.send_message.assert_not_awaited()


def test_web_app_button_is_described_to_the_model_as_delivery_only() -> None:
    from tools.send_message_tool import SEND_MESSAGE_SCHEMA

    schema = SEND_MESSAGE_SCHEMA["parameters"]["properties"]["web_app_button"]
    assert schema["required"] == ["label", "url"]
    assert schema["additionalProperties"] is False
    description = schema["description"]
    assert "already-hosted HTTPS URL" in description
    assert "does not host or deploy" in description


@pytest.mark.parametrize(
    ("target", "button", "expected"),
    [
        (
            "slack:C123",
            {"label": "Open UI", "url": "https://example.com/app"},
            "only supported for Telegram",
        ),
        (
            "telegram:12345",
            {"label": "Open UI", "url": "http://example.com/app"},
            "must use HTTPS",
        ),
        (
            "telegram:12345",
            {"label": "Open UI", "url": "https://example.com/app", "extra": True},
            "exactly 'label' and 'url'",
        ),
    ],
)
def test_send_rejects_invalid_button_before_loading_gateway_config(
    target: str,
    button: dict,
    expected: str,
) -> None:
    from tools.send_message_tool import send_message_tool

    result = json.loads(
        send_message_tool({
            "action": "send",
            "target": target,
            "message": "Ready",
            "web_app_button": button,
        })
    )
    assert expected in result["error"]
