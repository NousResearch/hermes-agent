"""Regression tests for issue #28639.

``delegate_task`` crashed with ``AttributeError: 'list' object has no
attribute 'lstrip'`` when a tool message ``content`` was a list (e.g.
OpenAI-style multimodal ``[{"type": "text", "text": "..."}]`` parts).
These tests cover both the low-level error detector and the
``_normalize_tool_content`` helper used at the tool_trace call site.
"""

from __future__ import annotations

from tools.delegate_tool import (
    _looks_like_error_output,
    _normalize_tool_content,
)


# --- _normalize_tool_content -------------------------------------------------


def test_normalize_none_returns_empty_string():
    assert _normalize_tool_content(None) == ""


def test_normalize_string_passthrough():
    assert _normalize_tool_content("hello") == "hello"


def test_normalize_list_of_text_parts_joins_text():
    content = [
        {"type": "text", "text": "first line"},
        {"type": "text", "text": "second line"},
    ]
    assert _normalize_tool_content(content) == "first line\nsecond line"


def test_normalize_list_with_non_text_part_is_serialized_not_crashed():
    content = [
        {"type": "text", "text": "see image"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]
    out = _normalize_tool_content(content)
    assert "see image" in out
    # non-text part must not raise and must contribute *something* to the
    # serialized output so byte counts stay non-zero.
    assert "image_url" in out


def test_normalize_list_of_strings():
    assert _normalize_tool_content(["a", "b"]) == "a\nb"


def test_normalize_dict_with_text_key():
    assert _normalize_tool_content({"text": "ok"}) == "ok"


def test_normalize_dict_without_text_key_json_encodes():
    out = _normalize_tool_content({"foo": "bar"})
    assert "foo" in out and "bar" in out


# --- _looks_like_error_output ------------------------------------------------


def test_looks_like_error_output_accepts_list_content():
    # This is the exact crash from the issue traceback: a list-form
    # tool message content used to raise AttributeError on .lstrip().
    content = [{"type": "text", "text": "error: boom"}]
    assert _looks_like_error_output(content) is True


def test_looks_like_error_output_list_with_plain_text_not_error():
    content = [{"type": "text", "text": "all good here"}]
    assert _looks_like_error_output(content) is False


def test_looks_like_error_output_empty_list_not_error():
    assert _looks_like_error_output([]) is False


def test_looks_like_error_output_dict_json_error_key():
    # dict content surfaced via normalization should still be detected.
    assert _looks_like_error_output({"error": "nope"}) is True


def test_looks_like_error_output_string_still_works():
    # Regression guard: the original string fast-path must keep working.
    assert _looks_like_error_output("Traceback (most recent call last):") is True
    assert _looks_like_error_output("just a normal log line") is False
