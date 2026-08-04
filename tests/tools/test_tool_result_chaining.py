"""Self-check for tool result chaining (#78061): a tool can consume a prior
tool's output via {{tool_result:<call_id>.<field>}} without the model re-emitting it."""

from types import SimpleNamespace

from agent.agent_runtime_helpers import (
    remember_tool_result,
    resolve_tool_result_refs,
    tool_result_reference_hint,
)


def test_chaining():
    agent = SimpleNamespace()
    # A prior tool produced structured output.
    remember_tool_result(agent, "call_1", '{"data":"abc","nested":{"x":7},"items":["a","b"]}')

    # Passing it to a later tool by reference resolves to the stored value.
    assert resolve_tool_result_refs(agent, {"image_url": "{{tool_result:call_1.data}}"}) == {"image_url": "abc"}
    assert resolve_tool_result_refs(agent, {"path": "{{tool_result:call_1.nested.x}}"}) == {"path": "7"}
    assert resolve_tool_result_refs(agent, {"index": "{{tool_result:call_1.items.1}}"}) == {"index": "b"}
    # Nested containers resolve to compact JSON strings (args stay strings).
    assert resolve_tool_result_refs(agent, {"cfg": "{{tool_result:call_1.nested}}"}) == {"cfg": '{"x": 7}'}

    # The model-facing hint advertises the reference.
    hint = tool_result_reference_hint("call_1", '{"ok":true}')
    assert "{{tool_result:call_1.field}}" in hint
    # Non-structured results get no hint.
    assert tool_result_reference_hint("call_1", "plain text") == ""
    # Unknown call id leaves the reference untouched (no crash).
    assert resolve_tool_result_refs(agent, {"x": "{{tool_result:missing.a}}"}) == {"x": "{{tool_result:missing.a}}"}
    # Plain strings pass through.
    assert resolve_tool_result_refs(agent, {"x": "hello"}) == {"x": "hello"}
    print("test_chaining passed")


if __name__ == "__main__":
    test_chaining()