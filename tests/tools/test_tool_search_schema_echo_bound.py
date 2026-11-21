"""A ``tool_call`` validation error must stay small however large the tool's schema is (#155742).

``_validation_error`` echoed the tool's ENTIRE ``parameters`` object beside the failure
detail, so a plugin tool with a many-branch discriminated schema answered ~33 KB per
failed call (measured on the report's tool: 33,537 bytes for a 33,883-byte schema) — and
a model that retries adds another copy to the conversation each time. The echo is now
trimmed to the branch the call failed on, or replaced by a note pointing at
``tool_describe``, once it would pass the schema-echo budget; small schemas are echoed
verbatim exactly as before.
"""

import json

import pytest

# One failed call must fit comfortably in a few KB of model context.
CALL_BUDGET = 6000
# Matches the reported plugin tool: tens of KB, one tool with many `action` branches.
LARGE_SCHEMA_CHARS = 20000


def _large_params(branches: int = 40, filler: int = 600):
    """One tool, many ``action`` branches — the shape that made the echo 33 KB."""
    properties = {"action": {"type": "string",
                             "enum": [f"branch_{i}" for i in range(branches)]}}
    branches_if_then = []
    for i in range(branches):
        properties[f"branch_{i}"] = {
            "type": "object",
            "description": "x" * filler,
            "properties": {"fields": {"type": "array", "items": {"type": "string"}}},
            "additionalProperties": False,
        }
        branches_if_then.append({
            "if": {"properties": {"action": {"const": f"branch_{i}"}}},
            "then": {"required": [f"branch_{i}"]},
        })
    return {"type": "object", "properties": properties,
            "required": ["action"], "allOf": branches_if_then}


def _register(name: str, params: dict, toolset: str = "mcp-probe-echo-budget"):
    from tools.registry import registry

    def _handler(args, task_id=None, **kw):
        return json.dumps({"ok": True, "args": args})

    registry.register(
        name=name,
        handler=_handler,
        schema={"name": name, "description": f"desc {name}", "parameters": params},
        toolset=toolset,
    )
    return name


def _payload(err: str | None) -> dict:
    assert err is not None, "a missing-required/failing call must return a validation error"
    assert len(err) < CALL_BUDGET, f"validation error is {len(err)} chars: {err[:400]}"
    return json.loads(err)


def test_missing_required_on_a_large_schema_echoes_only_the_needed_property():
    from tools.tool_search import validate_deferred_call_args

    params = _large_params()
    assert len(json.dumps(params)) > LARGE_SCHEMA_CHARS
    name = _register("mcp_echo_large_required", params)

    payload = _payload(validate_deferred_call_args(name, {}))

    assert "action" in payload["error"]
    echoed = payload["parameters"]
    # The branch the call failed on, not the whole discriminated schema.
    assert echoed["required"] == ["action"]
    assert list(echoed["properties"]) == ["action"]
    assert "tool_describe" in echoed["note"]
    assert "x" * 600 not in json.dumps(payload)


def test_failing_branch_on_a_large_schema_echoes_that_branch_only():
    from tools.tool_search import validate_deferred_call_args

    name = _register("mcp_echo_large_branch", _large_params())

    payload = _payload(
        validate_deferred_call_args(name, {"action": "branch_1", "branch_1": {"fields": "nope"}}))

    assert "branch_1" in payload["error"]
    echoed = payload["parameters"]
    assert list(echoed["properties"]) == ["branch_1"]
    assert echoed["properties"]["branch_1"]["properties"]["fields"]["type"] == "array"
    # Every *other* branch — and their 600-char descriptions — stays out of the payload.
    assert "branch_2" not in json.dumps(echoed)


@pytest.mark.parametrize("branches", [2, 40])
def test_error_size_does_not_track_schema_size(branches):
    from tools.tool_search import validate_deferred_call_args

    name = _register(f"mcp_echo_size_{branches}", _large_params(branches=branches))
    err = validate_deferred_call_args(name, {"action": "nope"})
    _payload(err)


def test_small_schema_echo_is_unchanged():
    from tools.tool_search import validate_deferred_call_args

    params = {
        "type": "object",
        "properties": {"document_id": {"type": "string", "description": "Doc id"}},
        "required": ["document_id"],
    }
    name = _register("mcp_echo_small_unchanged", params)

    err = validate_deferred_call_args(name, {})
    assert err is not None
    payload = json.loads(err)

    assert payload["parameters"] == params
    assert payload["hint"] == "Retry tool_call with 'arguments' matching the parameters schema above."


def test_flattened_probe_message_is_bounded_too(monkeypatch):
    """The provider-native lane flattens the probe payload into one string; it reads the
    same bounded ``parameters``, so it must stay small for the same reason."""
    from agent import tool_executor

    name = _register("mcp_echo_large_flattened", _large_params())
    monkeypatch.setattr(tool_executor, "_tool_search_scoped_names",
                        lambda agent: frozenset({name}))

    _, _, block = tool_executor._unwrap_tool_search_call(
        None, "tool_call", {"name": name, "arguments": {}}, flatten_probe=True)

    assert block is not None
    assert len(block) < CALL_BUDGET, f"flattened block is {len(block)} chars"
    assert "tool_describe" in block
