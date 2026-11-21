"""Regression for #123345: type-less anyOf/oneOf union args were skipped by coercion.

The coercion loop only read ``prop_schema["type"]``; a union without a top-level
``type`` (terminal ``notify: anyOf[boolean, array]``, a non-nullable MCP
``Union[bool, list[str]]``, or a plugin tool registered with a raw ``anyOf``) yielded
``expected=None`` and was skipped, so string-typed args reached the handler uncoerced.
Nullable MCP ``Optional[X]`` unions are collapsed to ``{type: X, nullable: true}`` by
``_normalize_mcp_input_schema`` before registration, so they already coerce as a plain
type; the synthetic ``anyOf[X, null]`` fixtures below exercise the helper's branch
handling for unions that survive normalization (non-nullable MCP / raw plugin schemas).

These tests drive the real dispatch chokepoint (``handle_function_call`` ->
``coerce_tool_args`` -> terminal handler) with the executor stubbed (no command is ever
spawned), plus the class invariants: $ref branches, cyclic/dangling refs, string-union
preservation, nullable desugar to the equivalent type-list, and exact large-integer parsing.
"""

import json

import pytest

import model_tools
import tools.terminal_tool  # registers the real terminal schema
from tools.arg_coercion import coerce_tool_args
from tools.registry import registry


# ── dispatch-level contract: real terminal schema, real handler, stubbed executor ──


@pytest.fixture
def captured_terminal(monkeypatch):
    """Stub the terminal executor so dispatch reaches it without spawning anything."""
    captured = {}

    def _fake_executor(**kw):
        captured.update(kw)
        return json.dumps({"status": "ok", "pid": 1})

    monkeypatch.setattr("tools.terminal_tool.terminal_tool", _fake_executor)
    return captured


def _dispatch(args):
    # skip hooks/middleware: coercion runs before them anyway; isolate coercion+handler.
    return model_tools.handle_function_call(
        "terminal", args, task_id="t",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )


class TestNotifyDispatchContract:
    def test_background_notify_true_string_notifies_on_exit(self, captured_terminal):
        _dispatch({"command": "echo hi", "background": True, "notify": "true"})
        assert captured_terminal.get("notify_on_complete") is True

    def test_background_notify_false_string_stays_silent(self, captured_terminal):
        _dispatch({"command": "echo hi", "background": True, "notify": "false"})
        assert captured_terminal.get("notify_on_complete") is False

    def test_background_notify_json_array_string_becomes_watch_patterns(self, captured_terminal):
        _dispatch({"command": "echo hi", "background": True, "notify": '["ready", "done"]'})
        assert captured_terminal.get("watch_patterns") == ["ready", "done"]
        assert captured_terminal.get("notify_on_complete") is False

    def test_foreground_notify_false_string_is_a_silent_noop(self, captured_terminal):
        # The string "false" is truthy: before the fix the foreground gate wrongly
        # refused it. Coerced to False it passes the gate and runs as a no-op notify.
        result = json.loads(_dispatch({"command": "echo hi", "notify": "false"}))
        assert "error" not in result
        assert captured_terminal.get("notify_on_complete") is False
        assert captured_terminal.get("background") is False

    def test_foreground_notify_true_string_still_refused(self, captured_terminal):
        result = json.loads(_dispatch({"command": "echo hi", "notify": "true"}))
        assert "background" in result["error"]
        assert captured_terminal == {}  # refused at the gate, executor never ran

    @pytest.mark.parametrize("bad", ["maybe", "1", "yes", "on", "tru"])
    def test_background_unrecognized_notify_fails_closed(self, captured_terminal, bad):
        # One boolean truth table (_coerce_boolean: true/false only). Shell-ish tokens
        # are NOT booleans; the handler refuses them rather than silently notifying.
        result = json.loads(_dispatch({"command": "echo hi", "background": True, "notify": bad}))
        assert "notify must be" in result["error"]
        assert captured_terminal == {}

    def test_background_native_bool_and_list_unchanged(self, captured_terminal):
        _dispatch({"command": "echo hi", "background": True, "notify": True})
        assert captured_terminal.get("notify_on_complete") is True
        captured_terminal.clear()
        _dispatch({"command": "echo hi", "background": True, "notify": ["ready"]})
        assert captured_terminal.get("watch_patterns") == ["ready"]


# ── class invariants of the shared-layer fix (synthetic schemas) ──


@pytest.fixture
def coerce_with(monkeypatch):
    """Coerce args against a synthetic schema via the real chokepoint (monkeypatch-restored)."""
    def _coerce(properties, args, defs=None):
        params = {"type": "object", "properties": properties}
        if defs is not None:
            params["$defs"] = defs
        schema = {"name": "probe", "parameters": params}
        monkeypatch.setattr(registry, "get_schema", lambda name: schema)
        return coerce_tool_args("probe", dict(args))
    return _coerce


class TestUnionClassInvariants:
    def test_ref_branches_resolve_per_branch(self, coerce_with):
        props = {"p": {"anyOf": [{"$ref": "#/$defs/b"}, {"$ref": "#/$defs/a"}]}}
        defs = {"b": {"type": "boolean"}, "a": {"type": "array", "items": {"type": "string"}}}
        assert coerce_with(props, {"p": "true"}, defs)["p"] is True
        assert coerce_with(props, {"p": '["x"]'}, defs)["p"] == ["x"]

    def test_cyclic_ref_left_alone_without_crash(self, coerce_with):
        props = {"p": {"anyOf": [{"$ref": "#/$defs/self"}, {"type": "integer"}]}}
        defs = {"self": {"$ref": "#/$defs/self"}}
        assert coerce_with(props, {"p": "5"}, defs)["p"] == "5"

    def test_dangling_ref_left_alone(self, coerce_with):
        props = {"p": {"anyOf": [{"$ref": "#/$defs/nope"}, {"type": "integer"}]}}
        assert coerce_with(props, {"p": "5"}, defs={})["p"] == "5"

    def test_string_union_is_not_reinterpreted(self, coerce_with):
        props = {"p": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}
        assert coerce_with(props, {"p": "007"})["p"] == "007"
        assert coerce_with(props, {"p": "42"})["p"] == "42"

    def test_typed_const_branch_left_alone(self, coerce_with):
        # A branch with a plain type AND const only permits that member; coercing "true"
        # to True would fabricate a value the const:false branch disallows. Leave it alone.
        props = {"p": {"anyOf": [{"type": "boolean", "const": False}, {"type": "array"}]}}
        assert coerce_with(props, {"p": "true"})["p"] == "true"

    def test_typed_enum_branch_left_alone(self, coerce_with):
        # Same for a typed enum: "5" is not a member of [1, 2, 3]; do not coerce it to 5.
        props = {"p": {"anyOf": [{"type": "integer", "enum": [1, 2, 3]}, {"type": "null"}]}}
        assert coerce_with(props, {"p": "5"})["p"] == "5"

    def test_enum_on_the_union_itself_left_alone(self, coerce_with):
        props = {"p": {"anyOf": [{"type": "integer"}, {"type": "null"}], "enum": [1, 2, 3, None]}}
        assert coerce_with(props, {"p": "5"})["p"] == "5"

    def test_const_on_the_union_itself_left_alone(self, coerce_with):
        props = {"p": {"anyOf": [{"type": "boolean"}, {"type": "array"}], "const": False}}
        assert coerce_with(props, {"p": "true"})["p"] == "true"

    @pytest.mark.parametrize("branch", [
        {"type": "integer", "allOf": [{"enum": [1, 2, 3]}]},
        {"type": "integer", "anyOf": [{"const": 1}, {"const": 2}]},
        {"type": "integer", "oneOf": [{"const": 1}, {"const": 2}]},
        {"type": "integer", "not": {"const": 5}},
        {"type": "boolean", "if": {}, "then": {"const": False}},
    ])
    def test_typed_branch_with_composition_left_alone(self, coerce_with, branch):
        props = {"p": {"anyOf": [branch, {"type": "null"}]}}
        assert coerce_with(props, {"p": "5"})["p"] == "5"

    @pytest.mark.parametrize("node", [
        {"anyOf": [{"type": "integer"}, {"type": "null"}], "allOf": [{"enum": [1, 2, 3]}]},
        {"anyOf": [{"type": "boolean"}, {"type": "array"}], "not": {"const": True}},
        {"anyOf": [{"type": "integer"}, {"type": "null"}], "oneOf": [{"const": 1}, {"const": 2}]},
        {"anyOf": [{"type": "integer"}], "oneOf": [{"type": "boolean"}]},
    ])
    def test_node_level_composition_left_alone(self, coerce_with, node):
        # Composition sibling to anyOf on the union node is the same fabrication class as a
        # constrained branch, one level up; the node guard bails symmetrically.
        assert coerce_with({"p": node}, {"p": "5"})["p"] == "5"

    def test_ref_target_const_left_alone(self, coerce_with):
        props = {"p": {"anyOf": [{"$ref": "#/$defs/f"}, {"type": "array"}]}}
        assert coerce_with(props, {"p": "true"}, {"f": {"type": "boolean", "const": False}})["p"] == "true"

    def test_constrained_branch_valid_member_also_left_alone(self, coerce_with):
        # Known cost of the presence guard: a valid member is not repaired either.
        props = {"p": {"anyOf": [{"type": "boolean", "const": False}, {"type": "array"}]}}
        assert coerce_with(props, {"p": "false"})["p"] == "false"
        assert coerce_with(props, {"p": '["a"]'})["p"] == '["a"]'

    def test_range_constrained_branch_matches_plain_type(self, coerce_with):
        # Parity: range keywords do not stop coercion on the union or the plain path.
        union = {"p": {"anyOf": [{"type": "integer", "minimum": 10}, {"type": "null"}]}}
        plain = {"p": {"type": "integer", "minimum": 10}}
        for raw in ("5", "15"):
            assert coerce_with(union, {"p": raw})["p"] == coerce_with(plain, {"p": raw})["p"] == int(raw)

    def test_non_nullable_union_coerces(self, coerce_with):
        # The real surviving class: a non-nullable union (Pydantic Union[bool, list[str]]
        # -> anyOf[boolean, array]) is NOT collapsed by MCP normalization, so it coerces.
        props = {"p": {"anyOf": [{"type": "boolean"}, {"type": "array", "items": {"type": "string"}}]}}
        assert coerce_with(props, {"p": "true"})["p"] is True
        assert coerce_with(props, {"p": '["a"]'})["p"] == ["a"]

    def test_nullable_integer_union_coerces_and_null_becomes_none(self, coerce_with):
        # Synthetic anyOf[integer, null] (a raw plugin schema; real MCP Optional[int] is
        # normalized to type:integer + nullable hint before coercion).
        props = {"p": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}
        assert coerce_with(props, {"p": "5"})["p"] == 5
        assert coerce_with(props, {"p": "null"})["p"] is None

    def test_nullable_array_union_does_not_wrap_bare_string(self, coerce_with):
        # desugar to type-list: anyOf[array, null] behaves like type:[array, null],
        # which leaves a bare string intact (no single-element wrap).
        props = {"p": {"anyOf": [{"type": "array"}, {"type": "null"}]}}
        assert coerce_with(props, {"p": "abc"})["p"] == "abc"

    def test_pure_single_branch_array_wraps_bare_string(self, coerce_with):
        # anyOf[array] collapses to scalar "array" -> behaves like type: array (wrap).
        props = {"p": {"anyOf": [{"type": "array"}]}}
        assert coerce_with(props, {"p": "abc"})["p"] == ["abc"]

    def test_enum_branch_left_alone(self, coerce_with):
        props = {"p": {"anyOf": [{"enum": ["a", "b"]}, {"type": "integer"}]}}
        assert coerce_with(props, {"p": "5"})["p"] == "5"

    def test_allof_wrapper_left_alone(self, coerce_with):
        props = {"p": {"allOf": [{"type": "integer"}]}}
        assert coerce_with(props, {"p": "5"})["p"] == "5"

    def test_oneof_union_coerces(self, coerce_with):
        props = {"p": {"oneOf": [{"type": "boolean"}, {"type": "array"}]}}
        assert coerce_with(props, {"p": "true"})["p"] is True

    def test_nullable_integer_union_preserves_snowflake_precision(self, coerce_with):
        # A 19-digit Discord snowflake must survive exactly; a float round-trip
        # would corrupt it (1550136568038035536 -> 1550136568038035456). Guards the
        # integer branch newly coerced via a type-less union (and plain type:integer).
        props = {"p": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}
        out = coerce_with(props, {"p": "1550136568038035536"})["p"]
        assert out == 1550136568038035536
        assert isinstance(out, int)

    def test_nullable_list_union_coerces_json_string(self, coerce_with):
        props = {"p": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}}
        assert coerce_with(props, {"p": '["a", "b"]'})["p"] == ["a", "b"]

    def test_native_container_union_matches_type_list(self, coerce_with):
        # Deliberate parity: native-container normalization under a union behaves exactly
        # like the equivalent type-list (the native-container guard is unchanged). Both
        # leave a JSON-encoded inner string intact here; the fix does not silently diverge.
        inner = {"type": "object", "properties": {"x": {"type": "array", "items": {"type": "string"}}}}
        native = [{"x": '["a"]'}]
        via_union = coerce_with({"p": {"anyOf": [{"type": "array", "items": inner}, {"type": "null"}]}}, {"p": native})["p"]
        via_type_list = coerce_with({"p": {"type": ["array", "null"], "items": inner}}, {"p": native})["p"]
        assert via_union == via_type_list


    def test_plain_integer_preserves_snowflake_precision(self, coerce_with):
        # Pre-existing precision bug, proven red on base through the public chokepoint:
        # a plain type:integer corrupted large ids via float before this fix. This is the
        # most frequent case (not a union), framed honestly as a second defect of the class.
        out = coerce_with({"p": {"type": "integer"}}, {"p": "1550136568038035536"})["p"]
        assert out == 1550136568038035536
        assert isinstance(out, int)

    def test_plain_number_preserves_precision(self, coerce_with):
        # A number property also parses exact integer literals (no float rounding past 2**53).
        out = coerce_with({"p": {"type": "number"}}, {"p": "9007199254740993"})["p"]
        assert out == 9007199254740993
