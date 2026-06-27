"""Regression tests for recovering tool calls a model emitted as TEXT.

Root cause (verified empirically 2026-06-11): qwen3-coder:30b via Ollama emits
a tool call inside assistant *content* ~33% of the time instead of via the
structured ``tool_calls`` field. The captured real leak — reproduced against
the live Ollama endpoint Hermes uses — is the Qwen-Agent XML form:

    <function=write_file>
    <parameter=path>
    hello.py
    </parameter>
    <parameter=content>
    print('hello world')
    </parameter>
    </function>
    </tool_call>

When the provider template-parser misses this, ``tool_calls`` is empty and the
turn silently ends as a text response, dropping the intended edit. These tests
lock the recovery contract so the normal dispatch path runs instead.
"""
import json

from agent.tool_call_recovery import recover_tool_calls_from_text

VALID = {"write_file", "read_file", "terminal"}

# The EXACT text captured from the live qwen3-coder:30b probe (not synthetic).
REAL_LEAK = (
    "<function=write_file>\n"
    "<parameter=path>\n"
    "hello.py\n"
    "</parameter>\n"
    "<parameter=content>\n"
    "print('hello world')\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def _args(call):
    return json.loads(call.function.arguments)


def test_recovers_real_captured_qwen_agent_xml_leak():
    calls, cleaned = recover_tool_calls_from_text(REAL_LEAK, VALID)
    assert len(calls) == 1
    c = calls[0]
    assert c.function.name == "write_file"
    assert c.type == "function"
    assert c.id and c.call_id  # shaped like an OpenAI tool_call object
    assert _args(c) == {"path": "hello.py", "content": "print('hello world')"}
    # The leaked XML (including the stray trailing </tool_call>) is consumed.
    assert "<function" not in cleaned
    assert "parameter" not in cleaned
    assert "tool_call" not in cleaned
    assert cleaned.strip() == ""


def test_recovers_tool_call_json_tag_form():
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
    calls, cleaned = recover_tool_calls_from_text(text, VALID)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert _args(calls[0]) == {"path": "a.txt"}
    assert "tool_call" not in cleaned


def test_recovers_python_dict_single_quoted_tool_call():
    # Hermes' own system prompt documents the single-quoted python-dict form.
    text = "<tool_call>\n{'name': 'read_file', 'arguments': {'path': 'a.txt'}}\n</tool_call>"
    calls, _ = recover_tool_calls_from_text(text, VALID)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert _args(calls[0]) == {"path": "a.txt"}


def test_gating_unknown_tool_name_is_not_recovered():
    text = "<function=rm_rf_everything>\n<parameter=path>/</parameter>\n</function>"
    calls, cleaned = recover_tool_calls_from_text(text, VALID)
    assert calls == []
    # Untouched — we never consume spans we didn't recover.
    assert cleaned == text


def test_prose_mentioning_a_tool_is_not_executed():
    text = "I will use the write_file tool to create hello.py for you."
    calls, cleaned = recover_tool_calls_from_text(text, VALID)
    assert calls == []
    assert cleaned == text


def test_surrounding_prose_is_preserved_when_cleaning():
    text = (
        "Sure, creating it now.\n"
        "<function=read_file><parameter=path>a.txt</parameter></function>\n"
        "Let me know if that works."
    )
    calls, cleaned = recover_tool_calls_from_text(text, VALID)
    assert len(calls) == 1
    assert "Sure, creating it now." in cleaned
    assert "Let me know if that works." in cleaned
    assert "<function" not in cleaned


def test_multiple_function_blocks_recovered():
    text = (
        "<function=read_file><parameter=path>a.txt</parameter></function>"
        "<function=read_file><parameter=path>b.txt</parameter></function>"
    )
    calls, _ = recover_tool_calls_from_text(text, VALID)
    assert len(calls) == 2
    assert _args(calls[0]) == {"path": "a.txt"}
    assert _args(calls[1]) == {"path": "b.txt"}
    assert calls[0].id != calls[1].id  # distinct call ids


def test_function_block_with_no_parameters_is_empty_args():
    text = "<function=terminal></function>"
    calls, _ = recover_tool_calls_from_text(text, VALID)
    assert len(calls) == 1
    assert _args(calls[0]) == {}


def test_no_structured_calls_when_already_present_is_caller_concern():
    # Recovery is a no-op on plain text with no tool-call markup.
    calls, cleaned = recover_tool_calls_from_text("just a normal answer", VALID)
    assert calls == []
    assert cleaned == "just a normal answer"


def test_empty_and_none_input():
    assert recover_tool_calls_from_text("", VALID) == ([], "")
    assert recover_tool_calls_from_text(None, VALID) == ([], "")
