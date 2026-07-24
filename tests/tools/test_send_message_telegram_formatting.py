from __future__ import annotations

from tools.send_message_tool import _message_uses_telegram_html


def test_plain_code_like_angle_brackets_are_not_treated_as_html() -> None:
    assert not _message_uses_telegram_html("RSS fetch failed in <urlopen>; falling back to cache")


def test_supported_telegram_html_tags_are_treated_as_html() -> None:
    assert _message_uses_telegram_html("<b>Important</b> <a href='https://example.com'>link</a>")


def test_supported_opening_tag_with_attributes_is_treated_as_html() -> None:
    assert _message_uses_telegram_html("<a href='https://example.com'>link")


def test_unsupported_html_like_tags_are_not_treated_as_html() -> None:
    assert not _message_uses_telegram_html("<script>alert('x')</script>")


def test_mixed_supported_and_unsupported_tags_fall_back_to_markdown_v2() -> None:
    # Even a single unsupported tag-like token forces MarkdownV2 fallback.
    # This pins the all() semantics: one bad tag → safe mode, no partial HTML.
    assert not _message_uses_telegram_html("<b>Error</b> parsing <urlopen>")
