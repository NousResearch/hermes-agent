"""Standalone Telegram MarkdownV2 chunk-indicator escaping (#74004).

When ``_send_telegram()`` chunks a long formatted MarkdownV2 message,
``truncate_message()`` appends a raw `` (1/N)`` suffix *after*
``format_message()`` has already escaped the text.  The parentheses are
MarkdownV2-reserved and must be escaped, and the indicator must be separated
from a trailing code fence — otherwise Telegram rejects the parse and silently
falls back to plain text.

The live gateway adapter already applies both transformations; this test
verifies the standalone send path matches.
"""

from __future__ import annotations

import asyncio
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Test helpers (same mocking pattern as test_telegram_send_message_caption.py)
# ---------------------------------------------------------------------------

def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    return bot


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
        "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")


def _long_markdown_message(repeats: int = 250) -> str:
    """Build markdown that exceeds 4096 UTF-16 units *after* MarkdownV2 formatting."""
    return "\n".join(f"**bold** item {i} text content here" for i in range(repeats))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chunk_indicators_have_escaped_parens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-chunk MarkdownV2 messages must escape () in the (N/M) indicator."""
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    message = _long_markdown_message()
    result = asyncio.run(_send_telegram("tok", "123", message))
    assert result["success"] is True

    calls = bot.send_message.await_args_list
    assert len(calls) >= 2, f"Expected chunking into 2+ messages, got {len(calls)}"

    for idx, call in enumerate(calls):
        text = call.kwargs.get("text", "")
        assert call.kwargs.get("parse_mode") == "MarkdownV2"
        # The (N/M) indicator must NOT have unescaped parens.
        assert not re.search(r" \(\d+/\d+\)$", text), (
            f"Chunk {idx} has UNESCAPED chunk indicator: ...{text[-40:]!r}"
        )
        # The (N/M) indicator MUST have escaped parens.
        assert re.search(r" \\\(\d+/\d+\\\)$", text), (
            f"Chunk {idx} missing ESCAPED chunk indicator: ...{text[-40:]!r}"
        )


def test_single_chunk_no_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short message must not receive a chunk indicator at all."""
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    result = asyncio.run(_send_telegram("tok", "123", "**short** message"))
    assert result["success"] is True

    assert bot.send_message.await_count == 1
    text = bot.send_message.await_args.kwargs.get("text", "")
    # No chunk indicator of any kind.
    assert not re.search(r"\(\d+/\d+\)", text), (
        f"Single chunk should have no indicator: {text!r}"
    )


def test_html_mode_chunked_no_paren_escaping(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTML-mode chunked messages must NOT get MarkdownV2 backslash escaping."""
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    # <b> triggers HTML mode; 600 repetitions exceed 4096 chars → chunking.
    message = "<b>bold</b> line text content padding " * 600
    result = asyncio.run(_send_telegram("tok", "123", message))
    assert result["success"] is True

    calls = bot.send_message.await_args_list
    assert len(calls) >= 2, f"Expected chunking, got {len(calls)} calls"

    for idx, call in enumerate(calls):
        text = call.kwargs.get("text", "")
        assert call.kwargs.get("parse_mode") == "HTML"
        # In HTML mode parens are NOT special → must stay unescaped.
        assert not re.search(r" \\\(\d+/\d+\\\)$", text), (
            f"HTML chunk {idx} should NOT backslash-escape parens: ...{text[-40:]!r}"
        )


def test_chunked_code_fence_indicator_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk indicator that lands on a code-fence line must be on its own line.

    When truncate_message() splits inside a fenced code block it closes the
    fence and appends the (N/M) indicator.  The indicator must NOT remain on the
    same line as the closing ``` — it must be separated to its own line so
    Telegram treats the fence as a clean close.
    """
    from tools.send_message_tool import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    # A long fenced code block that will be split across chunks.
    code_line = "x = data.get('key', default_value)"
    message = "```python\n" + "\n".join(code_line for _ in range(200)) + "\n```"
    result = asyncio.run(_send_telegram("tok", "123", message))
    assert result["success"] is True

    calls = bot.send_message.await_args_list
    assert len(calls) >= 2, f"Expected chunking, got {len(calls)} calls"

    for idx, call in enumerate(calls):
        text = call.kwargs.get("text", "")
        # No line should be "``` (N/M)" or "``` \\(N/M\\)" — the indicator
        # must never sit on the same line as a closing code fence.
        for line in text.split("\n"):
            assert not re.match(r"^```\s*", line) or "(" not in line, (
                f"Chunk {idx} has indicator glued to closing fence: {line!r}"
            )
