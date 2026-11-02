"""Root union fields remain callable through the model-facing MCP catalog."""
import copy
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

import model_tools
from tools.mcp_tool_schema import _convert_mcp_schema
from tools.registry import registry
from tools.tool_search import dispatch_tool_describe


@pytest.fixture
def register_server_tool():
    names = []

    def register(raw):
        schema = _convert_mcp_schema("union_contract", SimpleNamespace(
            name=f"call_{len(names)}", description="Exercise a root union", inputSchema=raw))
        name = schema["name"]
        validator = Draft202012Validator(raw)

        def server(args, **kwargs):
            validator.validate(args)
            return json.dumps(args)

        registry.register(name=name, toolset="mcp-union_contract", schema=schema, handler=server)
        names.append(name)
        return name

    yield register
    for name in names:
        registry.deregister(name)
    model_tools._clear_tool_defs_cache()


def _catalog(name):
    kwargs = {"enabled_toolsets": ["mcp-union_contract"], "skip_tool_search_assembly": True}
    definitions = model_tools.get_tool_definitions(**kwargs)
    cached = model_tools.get_tool_definitions(**kwargs, quiet_mode=True)
    hit = model_tools.get_tool_definitions(**kwargs, quiet_mode=True)
    schema = next(d["function"]["parameters"] for d in definitions if d["function"]["name"] == name)
    for tools in (cached, hit):
        assert next(d["function"]["parameters"] for d in tools if d["function"]["name"] == name) == schema
    described = json.loads(dispatch_tool_describe({"names": [name]}, current_tool_defs=hit))
    assert described["tools"][name]["parameters"] == schema
    return schema


def test_root_union_fields_reach_the_original_server_validator(register_server_tool):
    for union in ("anyOf", "oneOf"):
        raw = {"type": "object", "required": ["tag"], "$defs": {"Count": {"type": "integer"}}, union: [
            {"properties": {"tag": {"type": "string"}, "kind": {"const": "rows"}, "count": {"$ref": "#/$defs/Count"},
                            "rows/list": {"type": "array", "items": {
                                "type": "object", "properties": {"a/b": {"type": "string"}, "a_b": {"type": "string"}},
                                "required": ["a/b", "a_b"]}}},
             "required": ["kind", "count", "rows/list"]},
            {"properties": {"tag": {"type": "string"}, "kind": {"const": "text"}, "count": {"$ref": "#/$defs/Count"},
                            "message": {"type": "string"}}, "required": ["kind", "count", "message"]}]}
        original = copy.deepcopy(raw)
        name = register_server_tool(raw)
        exposed = _catalog(name)
        assert {"kind", "count", "rows_list", "message"} <= exposed["properties"].keys()
        assert set(exposed["required"]) == {"tag", "kind", "count"}
        assert union not in exposed
        args = {"tag": "required", "kind": "rows", "count": "2", "rows_list": '[{"a_b_2":"wire slash","a_b":"wire underscore"}]'}
        result = json.loads(model_tools.handle_function_call(name, args))
        assert result == {"tag": "required", "kind": "rows", "count": 2, "rows/list": [{"a/b": "wire slash", "a_b": "wire underscore"}]}
        assert raw == original


def test_root_union_conflicts_keep_string_and_nested_wire_interpretations(register_server_tool):
    raw = {"type": "object", "oneOf": [
        {"properties": {"kind": {"const": "text"}, "value": {"type": "string"}}, "required": ["kind", "value"]},
        {"properties": {"kind": {"const": "rows"}, "value": {"type": "array", "items": {
            "type": "object", "properties": {"wire/key": {"type": "string"}}, "required": ["wire/key"]}}},
         "required": ["kind", "value"]}]}
    name = register_server_tool(raw)
    exposed = _catalog(name)
    assert {branch["type"] for branch in exposed["properties"]["value"]["anyOf"]} == {"array", "string"}
    for args, expected in [
        ({"kind": "text", "value": '[{"wire_key":"literal"}]'}, {"kind": "text", "value": '[{"wire_key":"literal"}]'}),
        ({"kind": "rows", "value": [{"wire_key": "native"}]}, {"kind": "rows", "value": [{"wire/key": "native"}]}),
    ]:
        assert json.loads(model_tools.handle_function_call(name, args)) == expected

    for alternatives in (
        [{"type": "object", "properties": {"a/b": {"type": "string"}}},
         {"type": "object", "properties": {"a b": {"type": "string"}}}],
        [{"type": "object", "properties": {"a/b": {"type": "string"}}}, {}],
    ):
        raw = {"type": "object", "anyOf": [{"properties": {"payload": {"anyOf": alternatives}}}]}
        name = register_server_tool(raw)
        assert "payload" in _catalog(name)["properties"]
        args = {"payload": {"a_b": "ambiguous"}}
        assert json.loads(model_tools.handle_function_call(name, copy.deepcopy(args))) == args

    item = {"type": "object", "properties": {"wire/key": {"type": "string"}}, "required": ["wire/key"]}
    raw = {"type": "object", "anyOf": [
        {"properties": {"rows": {"type": "array", "items": item, "minItems": 1},
                        "a/b": {"type": "string"}}},
        {"properties": {"rows": {"type": "array", "items": item, "maxItems": 3},
                        "a_b": {"type": "string"}}}]}
    name = register_server_tool(raw)
    exposed = _catalog(name)
    assert exposed["properties"]["rows"]["type"] == "array"
    args = {"rows": ['{"wire_key":"nested string"}'], "a_b_2": "slash", "a_b": "underscore"}
    assert json.loads(model_tools.handle_function_call(name, args)) == {
        "rows": [{"wire/key": "nested string"}], "a/b": "slash", "a_b": "underscore"}

    raw = {"type": "object", "anyOf": [{"properties": {
        "template": item, "payload": {"$ref": "#/anyOf/0/properties/template"}}}]}
    name = register_server_tool(raw)
    exposed = _catalog(name)
    Draft202012Validator(exposed).validate({"payload": {"wire_key": "through ref"}})
    assert json.loads(model_tools.handle_function_call(name, {"payload": '{"wire_key":"through ref"}'})) == {
        "payload": {"wire/key": "through ref"}}

    raw = {"type": "object", "$defs": {"Rows": {"anyOf": [
        {"type": "array"}, {"$ref": "#/$defs/Rows", "description": "recursive alternative"}]}},
        "anyOf": [{"properties": {"rows": {"$ref": "#/$defs/Rows"}}}]}
    name = register_server_tool(raw)
    assert "rows" in _catalog(name)["properties"]
    assert json.loads(model_tools.handle_function_call(name, {"rows": []})) == {"rows": []}


def test_nested_allof_containers_retain_dispatch_coercion(register_server_tool):
    from tools.arg_coercion import coerce_tool_args

    for kind, encoded, decoded in (("array", "[]", []), ("object", "{}", {})):
        for constraint in (
            {"allOf": [{"type": kind}]},
            {"allOf": [{}, {"anyOf": [{"type": kind}, {"type": "null"}]}]},
        ):
            schema = {"type": "object", "properties": {"payload": {
                "type": "object", "properties": {"rows": constraint}, "required": ["rows"]}},
                "required": ["payload"]}
            original = copy.deepcopy(schema)
            name = register_server_tool(schema)
            args = {"payload": {"rows": encoded}}
            fixed = coerce_tool_args(name, copy.deepcopy(args))
            assert fixed == {"payload": {"rows": decoded}}
            assert json.loads(model_tools.handle_function_call(name, args)) == fixed
            assert schema == original

    schema = {"type": "object", "properties": {"payload": {"type": "object", "properties": {
        "rows": {"allOf": [{}, {"anyOf": [{"type": "array"}, {"type": "string"}]}]}}}}}
    name = register_server_tool(schema)
    args = {"payload": {"rows": "[]"}}
    assert coerce_tool_args(name, copy.deepcopy(args)) == args
    assert json.loads(model_tools.handle_function_call(name, args)) == args


def test_root_owned_fields_skip_discarded_projection(monkeypatch):
    from tools import schema_sanitizer as sanitizer

    schema = {"type": "object", "properties": {"rows": {"type": "string"}}, "anyOf": [
        {"properties": {"rows": {"type": "array", "items": {"type": "integer"}}}, "required": ["rows"]},
        {"properties": {"rows": {"type": "array", "items": {"type": "string"}}}, "required": ["rows"]}]}
    original = copy.deepcopy(schema)

    def discarded_projection(choices):
        pytest.fail("Root-owned fields must not transform discarded alternatives")

    monkeypatch.setattr(sanitizer, "_project_choices", discarded_projection)
    projected = sanitizer.preserve_root_union_properties(schema)
    assert projected["properties"] == schema["properties"]
    assert projected["required"] == ["rows"]
    assert schema == original
