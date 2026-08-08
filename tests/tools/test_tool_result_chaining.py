"""Self-check for tool result chaining (#78061): a tool can consume a prior
tool's output via {{tool_result:<call_id>.<field>}} without the model re-emitting it."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
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

    # The model-facing hint advertises a usable reference path.
    hint = tool_result_reference_hint("call_1", '{"ok":true}')
    assert "{{tool_result:call_1.ok}}" in hint
    # MCP-style wrapper {"result": {...}} reveals the nested path.
    hint = tool_result_reference_hint("call_1", '{"result": {"data": "x"}}')
    assert "{{tool_result:call_1.result.data}}" in hint
    # Non-structured results get no hint.
    assert tool_result_reference_hint("call_1", "plain text") == ""
    # Unknown call id leaves the reference untouched (no crash).
    assert resolve_tool_result_refs(agent, {"x": "{{tool_result:missing.a}}"}) == {"x": "{{tool_result:missing.a}}"}
    # Plain strings pass through.
    assert resolve_tool_result_refs(agent, {"x": "hello"}) == {"x": "hello"}
    print("test_chaining passed")


def test_mcp_double_encoded_result_resolves():
    """MCP tool results arrive double-encoded: the wrapper stores
    {"result": "<json-string>", "structuredContent": {"result": "<json-string>"}}
    (tools/mcp_tool.py). The walk must re-parse string leaves so chained
    paths like .result.data resolve against the real payload.
    """
    agent = SimpleNamespace()
    # Real MCP shape from tools/mcp_tool.py: text blocks joined into a str,
    # then json.dumps'ed; structuredContent may wrap the same string.
    remember_tool_result(agent, "call_mcp", json.dumps({
        "result": '{"folders": ["a", "b"], "count": 2}',
        "structuredContent": {"result": '{"folders": ["a", "b"], "count": 2}'},
    }))

    # .result.folders walks into the string leaf, re-parses, and resolves.
    assert resolve_tool_result_refs(
        agent, {"image_url": "{{tool_result:call_mcp.result.folders}}"}
    ) == {"image_url": '["a", "b"]'}
    assert resolve_tool_result_refs(
        agent, {"image_url": "{{tool_result:call_mcp.result.count}}"}
    ) == {"image_url": "2"}
    # structuredContent branch resolves the same way.
    assert resolve_tool_result_refs(
        agent, {"image_url": "{{tool_result:call_mcp.structuredContent.result.folders}}"}
    ) == {"image_url": '["a", "b"]'}
    # Unresolvable refs pass through untouched (contract preserved).
    assert resolve_tool_result_refs(
        agent, {"x": "{{tool_result:call_mcp.result.missing}}"}
    ) == {"x": "{{tool_result:call_mcp.result.missing}}"}
    # Non-JSON string leaf: pass through untouched, no crash.
    remember_tool_result(agent, "call_plain", json.dumps({"result": "not json"}))
    assert resolve_tool_result_refs(
        agent, {"x": "{{tool_result:call_plain.result.nope}}"}
    ) == {"x": "{{tool_result:call_plain.result.nope}}"}
    # The hint prefers structuredContent (the machine-oriented branch) and
    # reveals a working path for the double-encoded shape.
    hint = tool_result_reference_hint("call_mcp", json.dumps({
        "result": '{"folders": ["a", "b"], "count": 2}',
        "structuredContent": {"result": '{"folders": ["a", "b"], "count": 2}'},
    }))
    assert "{{tool_result:call_mcp.structuredContent.result.field}}" in hint


def _tc(name="echo_tool", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def test_sequential_path_resolves_refs_before_dispatch(agent):
    """The default single-call path (sequential) must resolve refs.

    Regression guard for the review finding that resolution only fired on
    the concurrent path (invoke_tool) while the sequential executor passed
    {{tool_result:...}} through verbatim. A single tool call routes to
    execute_tool_calls_sequential, so the dispatched args must be resolved.
    """
    seen = {}

    def fake_handle(name, args, task_id, **kwargs):
        seen["args"] = args
        return json.dumps({"ok": True})

    remember_tool_result(agent, "call_abc123", '{"data": "iVBORw0KGgoAAA"}')

    msg = SimpleNamespace(
        content="",
        tool_calls=[_tc(arguments='{"image_url": "{{tool_result:call_abc123.data}}"}')],
    )
    messages = []

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls(msg, messages, "task-1")

    assert seen["args"] == {"image_url": "iVBORw0KGgoAAA"}, seen
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert tool_msgs
    # Content stays a parseable JSON string (JSON contract); the ref hint
    # rides as a sibling key, injected for the model only at the wire.
    tool_msg = tool_msgs[-1]
    assert json.loads(tool_msg["content"]) == {"ok": True}
    assert "{{tool_result:" in tool_msg.get("_tool_result_hint", "")
    # The wire-time injection (ChatCompletionsTransport.convert_messages)
    # folds the sibling key into the model-visible content.
    from agent.transports.chat_completions import ChatCompletionsTransport
    wire = ChatCompletionsTransport().convert_messages([tool_msg], model="test-model")
    assert "{{tool_result:" in wire[0]["content"]
    assert tool_msg["tool_call_id"] in wire[0]["content"]


if __name__ == "__main__":
    test_chaining()
