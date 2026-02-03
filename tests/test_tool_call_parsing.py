from __future__ import annotations

from atropos.tools.base import ToolCall


def test_parse_tool_call_json_wrapper() -> None:
    text = '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>'
    calls = ToolCall.parse_from_text(text)
    assert len(calls) == 1
    assert calls[0].name == "terminal"
    assert calls[0].arguments == {"command": "pwd"}


def test_parse_tool_call_glm_style() -> None:
    text = '<tool_call>terminal{"command":"ls -la"}</tool_call>'
    calls = ToolCall.parse_from_text(text)
    assert len(calls) == 1
    assert calls[0].name == "terminal"
    assert calls[0].arguments == {"command": "ls -la"}


def test_parse_tool_call_missing_close_tag() -> None:
    text = '<tool_call>terminal{"command":"echo hi"}'
    calls = ToolCall.parse_from_text(text)
    assert calls == []


def test_parse_tool_call_strips_accidental_xml() -> None:
    text = '<tool_call>terminal{"command":"ls -la"}</arg_value></tool_call>'
    calls = ToolCall.parse_from_text(text)
    assert calls == []
