"""Rehydrated history tool rows carry the card kind a tool declared via `ui_card`."""

import json

from tools.registry import registry
from tui_gateway import server


def _history(tool_name: str) -> list[dict]:
    return [
        {"role": "user", "content": "fan out"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps({"tasks": [{"goal": "x"}]})},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": json.dumps({"results": []})},
    ]


def _tool_rows(tool_name: str) -> list[dict]:
    return [m for m in server._history_to_messages(_history(tool_name)) if m["role"] == "tool"]


def test_declared_ui_card_lands_on_history_tool_row():
    registry.register(
        name="plugin_fanout", toolset="test_ui_card",
        schema={"name": "plugin_fanout", "description": "x", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: "{}", ui_card="delegate",
    )
    try:
        rows = _tool_rows("plugin_fanout")
    finally:
        registry.deregister("plugin_fanout")
    assert len(rows) == 1
    assert rows[0]["name"] == "plugin_fanout"
    assert rows[0]["card"] == "delegate"


def test_undeclared_tool_history_row_has_no_card():
    rows = _tool_rows("plain_unknown_tool")
    assert len(rows) == 1
    assert "card" not in rows[0]


def test_card_stored_with_the_result_survives_without_the_registry():
    """A cold load (another process, a profile without the plugin) reads the stamp, not the registry."""
    history = _history("gone_plugin_tool")
    history[2]["display_metadata"] = {"tool_result_metadata": {"card": "delegate"}}
    rows = [m for m in server._history_to_messages(history) if m["role"] == "tool"]
    assert rows[0]["card"] == "delegate"
    history[2]["display_metadata"] = json.dumps({"tool_result_metadata": {"card": "delegate"}})
    rows = [m for m in server._history_to_messages(history) if m["role"] == "tool"]
    assert rows[0]["card"] == "delegate"
