"""`hermes send --to telegram` must only take the HTML branch for real Telegram tags.

Two defects on the standalone sender path (`tools/send_message_senders.py`):

1. The markup heuristic matched *any* `<...>` run, so prose like `<CoT>`, `List<int>`
   or `<name>` switched the message to `parse_mode=HTML`. `format_message()` is only
   called on the Markdown branch, so the message went out unconverted, Telegram
   answered 400 `unsupported start tag`, and the plain-text fallback posted raw
   markdown (`**` instead of bold). Live repro: post 254 in @ai_n_ns, 2026-09-18.
2. `truncate_message()` appends a raw ` (1/2)` indicator whose parentheses are
   MarkdownV2-special. The gateway adapter escapes it; this sender did not, so every
   multi-chunk `hermes send` 400'd and degraded to plain text.

The `telegram` package is stubbed.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=lambda **_kw: SimpleNamespace(**_kw),
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
                "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)


# Angle-bracketed prose that is NOT a Telegram tag: token names, generics, placeholders,
# an HTML tag Telegram does not support, and a bare less-than.
@pytest.mark.parametrize("prose", ["<CoT>", "List<int>", "<name>", "<body>", "a<b",
                                   "<script>", "<em ", "</ i>"])
def test_non_telegram_angle_brackets_take_the_markdown_branch(
        monkeypatch: pytest.MonkeyPatch, prose: str) -> None:
    from tools.send_message_senders import _telegram_format

    _install_telegram_mock(monkeypatch, MagicMock())
    message = f"**Заголовок** и {prose} в тексте"
    formatted, parse_mode, has_html = _telegram_format(message)

    assert (parse_mode, has_html) == ("MarkdownV2", False)
    # The message went through format_message(): **bold** became MarkdownV2 *bold*.
    assert "*Заголовок*" in formatted and "**" not in formatted


@pytest.mark.parametrize("markup", [
    "<b>x</b>",
    "</i>",
    '<a href="https://example.com">x</a>',
    '<span class="tg-spoiler">x</span>',
    '<tg-emoji emoji-id="1">x</tg-emoji>',
    "<blockquote expandable>x</blockquote>",
    '<code class="language-python">x</code>',
    "<PRE>x</PRE>",
])
def test_telegram_tags_take_the_html_branch(
        monkeypatch: pytest.MonkeyPatch, markup: str) -> None:
    from tools.send_message_senders import _telegram_format

    _install_telegram_mock(monkeypatch, MagicMock())
    formatted, parse_mode, has_html = _telegram_format(markup)

    assert (parse_mode, has_html) == ("HTML", True)
    assert formatted == markup  # HTML is passed through untouched


def test_multi_chunk_markdown_escapes_the_chunk_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    message = "\n\n".join(f"Параграф {i} с **жирным** словом." for i in range(400))
    res = asyncio.run(_send_telegram("tok", "123", message))

    assert res["success"] is True
    sent = [call.kwargs["text"] for call in bot.send_message.await_args_list]
    assert len(sent) > 1, "test needs a multi-chunk message"
    # Parentheses are MarkdownV2-special: the indicator must ride escaped, never raw.
    assert sent[0].endswith(r" \(1/%d\)" % len(sent))
    assert not any(chunk.endswith(")") and not chunk.endswith(r"\)") for chunk in sent)
    assert all(call.kwargs["parse_mode"] == "MarkdownV2" for call in bot.send_message.await_args_list)
