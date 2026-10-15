"""Progress labels agree with the bridge payloads accepted by real dispatch."""

import copy
import json

import pytest

from tools.registry import registry
from tools.tool_labels import labels_for_call
from tools.tool_search_catalog import TOOL_CALL_NAME


@pytest.fixture
def echo_tool():
    name = "mcp__label_contract__echo"
    seen = []

    def handler(arguments, **kwargs):
        seen.append(arguments)
        return json.dumps({"success": True, "value": arguments["query"]})

    registry.register(
        name=name,
        toolset="mcp-label-contract",
        schema={"name": name, "description": "Echo a synthetic test query.", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }},
        handler=handler,
    )
    yield name, seen
    registry.deregister(name)


@pytest.mark.parametrize("shape", ["array", "object", "encoded-array", "encoded-object", "legacy"])
@pytest.mark.parametrize("encoded_arguments", [False, True])
def test_labels_match_dispatch_for_accepted_envelopes(echo_tool, shape, encoded_arguments):
    from model_tools import handle_function_call

    name, seen = echo_tool
    arguments = {"query": "नमस्ते, hello"}
    entry = {"name": name, "arguments": json.dumps(arguments) if encoded_arguments else arguments}
    envelopes = {
        "array": {"calls": [entry]},
        "object": {"calls": entry},
        "encoded-array": {"calls": json.dumps([entry])},
        "encoded-object": {"calls": json.dumps(entry)},
        "legacy": entry,
    }
    payload = envelopes[shape]
    original = copy.deepcopy(payload)
    result = json.loads(handle_function_call(
        TOOL_CALL_NAME, payload, enabled_toolsets=["mcp-label-contract"],
    ))
    assert result.get("success") is True
    assert seen == [arguments]
    expected = labels_for_call(TOOL_CALL_NAME, {"calls": [{"name": name, "arguments": arguments}]})
    actual = labels_for_call(TOOL_CALL_NAME, payload)
    assert [label.name for label in actual] == [name]
    assert actual == expected
    assert seen == [arguments]  # Building labels must not dispatch another call.
    assert payload == original


@pytest.mark.parametrize("shape", ["invalid-json", "invalid-arguments", "mixed-batch"])
def test_rejected_envelopes_do_not_advertise_an_underlying_call(echo_tool, shape):
    from model_tools import handle_function_call

    name, seen = echo_tool
    entry = {"name": name, "arguments": {"query": "synthetic"}}
    payload = {
        "invalid-json": {"name": name, "calls": "not-json", "arguments": entry["arguments"]},
        "invalid-arguments": {"calls": [{"name": name, "arguments": ["not-an-object"]}]},
        "mixed-batch": {"calls": [entry, {"arguments": {"query": "missing-name"}}]},
    }[shape]
    original = copy.deepcopy(payload)
    result = json.loads(handle_function_call(
        TOOL_CALL_NAME, payload, enabled_toolsets=["mcp-label-contract"],
    ))
    assert result.get("error")
    assert seen == []
    labels = labels_for_call(TOOL_CALL_NAME, payload)
    assert labels
    assert all(not label.name and not label.preview for label in labels)
    assert seen == []
    assert payload == original
