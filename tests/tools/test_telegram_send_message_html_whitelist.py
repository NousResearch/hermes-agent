"""`hermes send --to telegram` must only take the HTML branch for real Telegram tags.

The markup heuristic on the standalone sender (`hermes_agent/tools/send_message_senders.py`) matched
*any* `<...>` run, so prose like `<CoT>`, `List<int>` or `<name>` switched the message to
`parse_mode=HTML`. `format_message()` only runs on the Markdown branch, so the message went
out unconverted, Telegram answered 400 `unsupported start tag`, and the plain-text fallback
posted raw markdown (`**` instead of bold).

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
    monkeypatch.setattr("hermes_agent.gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr("hermes_agent.gateway.platforms.base._detect_macos_system_proxy", lambda: None)


# Angle-bracketed prose that is NOT a Telegram tag: token names, generics, placeholders,
# an HTML tag Telegram does not support, and a bare less-than.
@pytest.mark.parametrize("prose", ["<CoT>", "List<int>", "<name>", "<body>", "a<b",
                                   "<script>", "<em ", "</ i>"])
def test_non_telegram_angle_brackets_take_the_markdown_branch(
        monkeypatch: pytest.MonkeyPatch, prose: str) -> None:
    from hermes_agent.plugins.platforms.telegram.standalone import _telegram_format

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
    from hermes_agent.plugins.platforms.telegram.standalone import _telegram_format

    _install_telegram_mock(monkeypatch, MagicMock())
    formatted, parse_mode, has_html = _telegram_format(markup)

    assert (parse_mode, has_html) == ("HTML", True)
    assert formatted == markup  # HTML is passed through untouched


@pytest.mark.parametrize("message, sent", [
    # A model-invented tag beside a real one is escaped, so the real tag still renders.
    ("<b>Status</b>: restart <service> now", "<b>Status</b>: restart &lt;service&gt; now"),
    ('<a href="https://x.y/?a=1&amp;b=2">l</a> 5 > 3 & up',
     '<a href="https://x.y/?a=1&amp;b=2">l</a> 5 &gt; 3 &amp; up'),
])
def test_html_send_escapes_markup_telegram_does_not_support(
        monkeypatch: pytest.MonkeyPatch, message: str, sent: str) -> None:
    from hermes_agent.plugins.platforms.telegram.standalone import _send_telegram

    _no_proxy(monkeypatch)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    assert asyncio.run(_send_telegram("tok", "123", message))["success"] is True
    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["parse_mode"] == "HTML"
    assert bot.send_message.await_args.kwargs["text"] == sent


@pytest.mark.parametrize("code", ["`<b>x</b>`", "``<b>x</b>``", "`` a ` <b>x</b> ``"])
def test_tag_shown_only_as_markdown_code_keeps_the_markdown_branch(
        monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    from hermes_agent.plugins.platforms.telegram.standalone import _telegram_format

    _install_telegram_mock(monkeypatch, MagicMock())
    formatted, parse_mode, has_html = _telegram_format(f"**Note**: example {code} here")

    assert (parse_mode, has_html) == ("MarkdownV2", False)
    assert formatted.startswith("*Note*") and "**" not in formatted


def test_a_quoted_gt_inside_an_attribute_does_not_break_the_tag(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_agent.plugins.platforms.telegram.standalone import _telegram_format

    _install_telegram_mock(monkeypatch, MagicMock())
    markup = '<a href="https://x.y/?q=a>b">link</a> and <b>bold</b>'
    assert _telegram_format(markup) == (markup, "HTML", True)
