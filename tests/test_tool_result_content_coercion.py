"""Regression tests for tool-result ``content`` coercion (issue #31435).

OpenAI Chat Completions requires ``role: "tool"`` message ``content`` to be a
string. Plugin tool handlers that return ``Dict[str, Any]`` previously got
persisted as a raw dict, which strict upstreams reject with HTTP 400
(Z.ai 1210, Manifest fallback_exhausted). ``make_tool_result_message`` now
normalizes via ``_coerce_tool_result_content``: strings and multimodal
results pass through; any other value is JSON-encoded.
"""
import json

from agent.tool_dispatch_helpers import (
    _coerce_tool_result_content,
    make_tool_result_message,
)


def test_dict_content_is_json_stringified():
    """The reported failure mode: a plugin returning a dict must not reach
    the wire as a dict."""
    content = {"definitions": [{"name": "wf"}], "count": 1}
    msg = make_tool_result_message("list_workflows", content, "call_1")
    assert isinstance(msg["content"], str)
    assert json.loads(msg["content"]) == content


def test_string_content_passes_through_unchanged():
    msg = make_tool_result_message("terminal", "$ ls\nfile.txt", "call_2")
    assert msg["content"] == "$ ls\nfile.txt"


def test_none_content_becomes_empty_string():
    """A handler returning None (silent success) must become "" — not the
    literal "null" — so strict providers don't reject a null content field."""
    assert _coerce_tool_result_content(None) == ""
    msg = make_tool_result_message("noop_tool", None, "call_none")
    assert msg["content"] == ""


def test_multimodal_content_is_preserved():
    """Multimodal envelopes must NOT be flattened — providers that support
    multipart tool messages consume the list directly."""
    multimodal = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "ok"}],
        "text_summary": "ok",
    }
    msg = make_tool_result_message("computer_use", multimodal, "call_3")
    assert msg["content"] is multimodal


def test_non_serializable_falls_back_to_str():
    class Weird:
        def __repr__(self):
            return "<weird>"

    out = _coerce_tool_result_content(Weird())
    assert isinstance(out, str)


def test_list_content_is_json_stringified():
    """A non-content-part list (e.g. ``[1, 2, 3]``) is not wire-valid and
    gets JSON-stringified."""
    out = _coerce_tool_result_content([1, 2, 3])
    assert out == "[1, 2, 3]"


def test_content_part_list_passes_through():
    """An OpenAI-style content-part list (multimodal content array) is already
    wire-valid for multimodal providers and passes through unchanged so the
    list structure stays intact for vision adapters."""
    parts = [{"type": "text", "text": "page contents"}]
    out = _coerce_tool_result_content(parts)
    assert out is parts


def test_make_tool_result_message_preserves_other_fields():
    msg = make_tool_result_message("toolx", {"ok": True}, "call_4")
    assert msg["role"] == "tool"
    assert msg["name"] == "toolx"
    assert msg["tool_name"] == "toolx"
    assert msg["tool_call_id"] == "call_4"
