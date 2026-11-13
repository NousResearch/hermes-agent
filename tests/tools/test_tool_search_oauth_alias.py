"""A deferred tool the Anthropic OAuth wire renames must resolve under the wire name too.

On an OAuth session the deferred-tool catalog shown to the model carries the aliased name
(``session_search`` -> ``chat_history_lookup``), but ``normalize_response`` only reverses
``tool_use`` block names, not names passed as ``tool_describe`` / ``tool_call`` arguments.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.anthropic_adapter import _OAUTH_TOOL_NAME_ALIASES
from tools.tool_search import (
    ToolSearchConfig, dispatch_tool_describe, resolve_underlying_call,
)


def _td(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": f"{name} tool",
        "parameters": {"type": "object", "properties": {}}}}


def _defer(*names: str):
    return patch("tools.tool_search.load_config_readonly",
                 return_value=ToolSearchConfig.from_raw({"defer": list(names)}))


@pytest.mark.parametrize("real", sorted(_OAUTH_TOOL_NAME_ALIASES))
def test_tool_describe_resolves_wire_alias_of_deferred_tool(real):
    wire = _OAUTH_TOOL_NAME_ALIASES[real]
    with _defer(real):
        result = json.loads(dispatch_tool_describe(
            {"names": [wire]}, current_tool_defs=[_td(real), _td("read_file")]))
    assert wire in result["tools"]
    assert "not_found" not in result


@pytest.mark.parametrize("real", sorted(_OAUTH_TOOL_NAME_ALIASES))
def test_tool_call_resolves_wire_alias_of_deferred_tool(real):
    wire = _OAUTH_TOOL_NAME_ALIASES[real]
    with _defer(real):
        name, args, err = resolve_underlying_call(
            {"name": wire, "arguments": {"q": 1}}, [real, "read_file"])
    assert (name, args, err) == (real, {"q": 1}, None)


def test_real_name_still_resolves_for_describe_and_call():
    with _defer("session_search"):
        described = json.loads(dispatch_tool_describe(
            {"names": ["session_search"]}, current_tool_defs=[_td("session_search")]))
        name, _, err = resolve_underlying_call(
            {"name": "session_search", "arguments": {}}, ["session_search"])
    assert "session_search" in described["tools"]
    assert (name, err) == ("session_search", None)


def test_wire_alias_of_a_tool_that_is_not_deferred_stays_unresolved():
    with _defer("todo_list"):
        described = json.loads(dispatch_tool_describe(
            {"names": ["chat_history_lookup"]}, current_tool_defs=[_td("todo_list")]))
        _, _, err = resolve_underlying_call(
            {"name": "chat_history_lookup", "arguments": {}}, ["todo_list"])
    assert described["not_found"] == ["chat_history_lookup"]
    assert err
