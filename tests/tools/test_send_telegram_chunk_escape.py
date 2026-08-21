"""Standalone _send_telegram multi-chunk MarkdownV2 escaping.

truncate_message() appends a raw " (N/M)" suffix to a chunk after
format_message() has already MarkdownV2-escaped the text. Unescaped
parentheses are invalid MarkdownV2, so without escaping the suffix,
Telegram rejects parse_mode for every chunk and the whole message falls
back to plain text -- even when no entity (e.g. an expandable blockquote)
was actually split across the chunk boundary.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot: MagicMock) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(Bot=MagicMock(return_value=bot), constants=constants_mod)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


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


def test_multi_chunk_index_suffix_is_escaped(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    _install_telegram_mock(monkeypatch, bot)

    # Force a 2-chunk split: two paragraphs each just under 4096 UTF-16 units.
    long_text = ("word " * 700).strip() + "\n\n" + ("more " * 700).strip()

    asyncio.run(_send_telegram("tok", "123", long_text))

    assert bot.send_message.await_count == 2
    for call in bot.send_message.await_args_list:
        text = call.kwargs["text"]
        # A raw, unescaped chunk-index suffix like " (1/2)" is invalid
        # MarkdownV2 (bare parentheses) and must never reach send_message.
        assert " (1/2)" not in text
        assert " (2/2)" not in text
        assert call.kwargs["parse_mode"] == "MarkdownV2"
