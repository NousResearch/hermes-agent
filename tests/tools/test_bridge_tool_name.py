"""The deferred-tool bridge's name must not collide with a tool-call delimiter (#110442).

Qwen-family templates (and the Hermes function-calling format) wrap native calls in
``<tool_call>...</tool_call>``; a bridge named ``tool_call`` read as that wrapper, so models
called it with the inner tool's arguments and the turn died on the identical-call guardrail.
These pin the contract: the advertised name is not a delimiter tag, nothing the model reads
names the old bridge, and transcripts recorded before the rename still read back.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools.tool_search_catalog import (
    BRIDGE_TOOL_NAMES,
    LEGACY_TOOL_CALL_NAMES,
    TOOL_CALL_NAME,
    TOOL_CALL_NAMES,
)

# The old bridge name as a word, but not the model's own <tool_call> tag.
_OLD_BRIDGE_WORD = re.compile(r"(?<![<\w/.])tool_call(?![\w>])")
_REPO = Path(__file__).resolve().parents[2]


def test_bridge_names_are_not_tool_call_delimiter_tags():
    from agent.agent_runtime_helpers import _TOOL_CALL_TAG_NAMES
    from hermes_cli.cli_render import _TOOL_CALL_TAGS

    assert TOOL_CALL_NAME == "invoke_tool"
    assert not BRIDGE_TOOL_NAMES & set(_TOOL_CALL_TAG_NAMES)
    assert not BRIDGE_TOOL_NAMES & set(_TOOL_CALL_TAGS)
    assert TOOL_CALL_NAMES == {TOOL_CALL_NAME} | LEGACY_TOOL_CALL_NAMES
    assert TOOL_CALL_NAME not in LEGACY_TOOL_CALL_NAMES


def test_model_facing_bridge_text_never_names_the_old_bridge():
    from agent.onboarding_text import _CONNECTORS
    from model_tools import _VAULT_NO_PASSWORD_NOTE
    from tools.tool_search import bridge_tool_schemas
    from tools.tool_search_validation import normalize_tool_call_entries

    texts = [_CONNECTORS, _VAULT_NO_PASSWORD_NOTE]
    for schema in bridge_tool_schemas(deferred_count=3, connections_granted=True):
        texts.append(json.dumps(schema))
    for bad in ({}, {"calls": "not json"}, {"calls": []}, {"calls": [1]}, {"calls": [{"arguments": {}}]},
                {"calls": [{"name": TOOL_CALL_NAME, "arguments": {}}]}):
        _entries, error = normalize_tool_call_entries(bad)
        assert error, bad
        texts.append(error)
    for text in texts:
        assert not _OLD_BRIDGE_WORD.search(text), text


@pytest.mark.parametrize("bridge", sorted(TOOL_CALL_NAMES))
def test_transcript_todo_calls_are_recognised_under_either_bridge_name(bridge):
    from tools.todo_tool import is_todo_tool_call

    call = {"id": "c1", "function": {"name": bridge, "arguments": json.dumps(
        {"calls": [{"name": "todo_list", "arguments": {"todos": []}}]})}}
    assert is_todo_tool_call(call)
    other = {"id": "c2", "function": {"name": bridge, "arguments": json.dumps(
        {"calls": [{"name": "read_file", "arguments": {"path": "todo.txt"}}]})}}
    assert not is_todo_tool_call(other)


def test_transcript_bridge_calls_label_the_same_under_either_name():
    from tools.tool_labels import labels_for_call

    args = {"calls": [{"name": "todo_list", "arguments": {"todos": []}}]}
    assert labels_for_call(TOOL_CALL_NAME, args) == labels_for_call("tool_call", args)
    assert labels_for_call(TOOL_CALL_NAME, args)


def test_every_locale_labels_the_bridge_under_its_current_name():
    block = re.compile(r"\n  bridge_generating:\n((?:    .*\n)+)")
    seen = 0
    for path in sorted((_REPO / "locales").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        match = block.search(text)
        if not match:
            continue
        seen += 1
        keys = re.findall(r"^    (\w+):", match.group(1), re.MULTILINE)
        assert TOOL_CALL_NAME in keys, path.name
        assert "tool_call" not in keys, path.name
    assert seen


def test_generating_phrase_for_the_bridge_is_a_translation_not_a_bare_key():
    from agent.display import bridge_generating_phrase

    phrase = bridge_generating_phrase(TOOL_CALL_NAME)
    assert phrase and "bridge_generating" not in phrase
