from __future__ import annotations

from tools.send_message_tool import _message_uses_telegram_html


def test_plain_code_like_angle_brackets_are_not_treated_as_html() -> None:
    assert not _message_uses_telegram_html("RSS fetch failed in <urlopen>; falling back to cache")


def test_supported_telegram_html_tags_are_treated_as_html() -> None:
    assert _message_uses_telegram_html("<b>Important</b> <a href='https://example.com'>link</a>")


def test_unsupported_html_like_tags_are_not_treated_as_html() -> None:
    assert not _message_uses_telegram_html("<script>alert('x')</script>")
