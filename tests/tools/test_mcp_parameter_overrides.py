"""Tests for ``tools.parameter_overrides`` on MCP servers.

A server can enforce a constraint its published input schema leaves out (Beeper Desktop's
``search_messages`` rejects ``limit`` above 20 with ``400 Invalid input`` while declaring a bare
integer). The override merges JSON-schema keys onto the declared parameter so the model sees
the bound, without mutating the server's own schema object (it is also written to the cache).
"""
import logging
from types import SimpleNamespace

from tools.mcp_tool_registration import _tool_candidates


def _search_tool():
    return SimpleNamespace(
        name="search_messages",
        description="Search messages.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Maximum number of messages to return."},
            },
            "required": [],
        },
        annotations={"readOnlyHint": True},
    )


def _candidates(tool, config):
    return _tool_candidates("beeper", [tool], lambda _name: True, 30.0, config=config)


def test_override_merges_keys_onto_declared_parameter():
    tool = _search_tool()
    config = {"tools": {"parameter_overrides": {"search_messages": {"limit": {"minimum": 1, "maximum": 20}}}}}

    [candidate] = _candidates(tool, config)

    assert candidate.schema["parameters"]["properties"]["limit"] == {
        "type": "integer", "description": "Maximum number of messages to return.", "minimum": 1, "maximum": 20}
    assert candidate.schema["parameters"]["properties"]["query"] == {"type": "string"}
    assert "maximum" not in tool.inputSchema["properties"]["limit"]


def test_no_override_leaves_schema_unchanged():
    [plain] = _candidates(_search_tool(), {})
    [other_tool] = _candidates(_search_tool(), {"tools": {"parameter_overrides": {"search_chats": {"limit": {"maximum": 5}}}}})

    assert plain.schema == other_tool.schema
    assert "maximum" not in plain.schema["parameters"]["properties"]["limit"]


def test_override_for_undeclared_parameter_is_skipped_with_warning(caplog):
    config = {"tools": {"parameter_overrides": {"search_messages": {"page_size": {"maximum": 20}}}}}

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        [candidate] = _candidates(_search_tool(), config)

    assert "page_size" not in candidate.schema["parameters"]["properties"]
    assert "search_messages.page_size does not match the tool's schema" in caplog.text


def test_callers_without_config_get_the_served_schema():
    [candidate] = _tool_candidates("beeper", [_search_tool()], lambda _name: True, 30.0)

    assert candidate.schema["parameters"]["properties"]["limit"] == {
        "type": "integer", "description": "Maximum number of messages to return."}
