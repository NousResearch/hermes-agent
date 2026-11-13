"""MCP references stay complete or become portable, never partially erased.

Real imports and conversion/describe paths; connector transport is injected offline.
"""
import copy
import json
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden in MCP schema regression tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _schema(defs=None, **properties):
    schema = {"type": "object", "properties": properties, "required": []}
    if defs is not None:
        schema["$defs"] = defs
    return schema


def _resolvable_cases():
    yield "basic", _schema({"A": {"type": "string"}}, x={"$ref": "#/$defs/A"}), {"x": {"type": "string"}}
    yield "legacy", {"type": "object", "definitions": {"A": {"type": "integer"}}, "properties": {"x": {"$ref": "#/definitions/A"}}, "required": []}, {"x": {"type": "integer"}}
    yield "literal-names", _schema({"A": {"type": "string"}}, definitions={"$ref": "#/$defs/A"}, **{"$defs": {"type": "integer"}}), {"definitions": {"type": "string"}, "$defs": {"type": "integer"}}
    yield "nested-scope", _schema({"A": {"type": "string"}}, x={"$ref": "#/$defs/A"}, nested={"type": "object", "$defs": {"A": {"type": "integer"}}, "properties": {"y": {"$ref": "#/properties/nested/$defs/A"}}, "required": []}), {"x": {"type": "string"}, "nested": {"type": "object", "properties": {"y": {"type": "integer"}}, "required": []}}
    yield "escaped", _schema({"A/B~C": {"type": "string"}}, x={"$ref": "#/$defs/A~1B~0C"}), {"x": {"type": "string"}}
    yield "percent", _schema({"A B": {"type": "integer"}}, x={"$ref": "#/$defs/A%20B"}), {"x": {"type": "integer"}}
    yield "array-pointer", _schema({"A": {"allOf": [{"type": "string"}]}}, x={"$ref": "#/$defs/A/allOf/0"}), {"x": {"type": "string"}}
    yield "siblings", _schema({"A": {"type": "string", "description": "old"}}, x={"$ref": "#/$defs/A", "description": "new", "minLength": 2}), {"x": {"type": "string", "description": "new", "minLength": 2}}
    yield "sibling-ref", _schema({"A": {"type": "array"}, "B": {"type": "integer"}}, x={"$ref": "#/$defs/A", "items": {"$ref": "#/$defs/B"}}), {"x": {"type": "array", "items": {"type": "integer"}}}
    yield "boolean", _schema({"A": False}, x={"$ref": "#/$defs/A"}), {"x": False}
    yield "long-chain", _schema({**{f"A{i}": {"$ref": f"#/$defs/A{i+1}"} for i in range(30)}, "A30": {"type": "integer"}}, x={"$ref": "#/$defs/A0"}), {"x": {"type": "integer"}}
    yield "sibling-conflict", _schema({"A": {"type": "string", "maxLength": 4}}, x={"$ref": "#/$defs/A", "maxLength": 8}), {"x": {"allOf": [{"type": "string", "maxLength": 4}, {"maxLength": 8}]}}
    yield "boolean-siblings", _schema({"A": False}, x={"$ref": "#/$defs/A", "description": "never"}), {"x": {"allOf": [False, {"description": "never"}]}}
    yield "shared", _schema({"A": {"type": "string"}}, x={"$ref": "#/$defs/A"}, y={"$ref": "#/$defs/A"}), {"x": {"type": "string"}, "y": {"type": "string"}}
    yield "pattern-and-dependent", _schema({"A": {"type": "string"}}, x={"type": "object", "patternProperties": {"definitions": {"$ref": "#/$defs/A"}}, "dependentSchemas": {"$defs": {"properties": {"y": {"$ref": "#/$defs/A"}}}}}), None
    yield "annotation-data", _schema({"A": {"type": "object", "default": {"$ref": "literal", "definitions": 1}}}, x={"$ref": "#/$defs/A"}), None
    yield "unused-ref", {"type": "object", "properties": {}, "$defs": {"A": {"type": "string"}, "B": {"$ref": "#/$defs/A"}}, "required": []}, {}
    yield "normalization", _schema({"A": {"anyOf": [{"const": "a", "type": "string"}, {"const": "b", "type": "string"}, {"type": "null"}]}}, x={"$ref": "#/$defs/A"}), None


def _fallback_cases():
    for name, ref in [("dangling", "#/definitions/Missing"), ("external", "https://example.invalid/schema"), ("anchor", "#Named"), ("root-cycle", "#")]:
        yield name, {"type": "object", "definitions": {"A": {"type": "string"}}, "properties": {"x": {"$ref": ref}}, "required": []}
    yield "unused-cycle", _schema({"A": {"$ref": "#/$defs/A"}})
    yield "cycle", _schema({"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}, x={"$ref": "#/$defs/A"})
    yield "mixed", _schema({"A": {"type": "string"}}, x={"$ref": "#/$defs/A"}, y={"$ref": "#/$defs/Missing"})
    yield "resource-scope", _schema({"A": {"type": "string"}}, x={"$id": "other.json", "$defs": {"A": {"type": "integer"}}, "$ref": "#/$defs/A"})
    yield "budget", _schema({"A": {"type": "string"}}, **{f"x{i}": {"$ref": "#/$defs/A"} for i in range(300)})
    yield "dynamic-ref", _schema({"A": {"type": "string"}}, x={"$dynamicRef": "#node"})
    yield "bad-pointer", _schema({"A~2B": {"type": "string"}}, x={"$ref": "#/$defs/A~2B"})


def _through_path(path, schema):
    from tools.mcp_tool_schema import _convert_mcp_schema, _normalize_mcp_input_schema
    from tools.tool_search import ToolSearchConfig, dispatch_tool_describe
    from tools.registry import registry

    if path == "normalize":
        return _normalize_mcp_input_schema(schema)
    if path == "convert":
        return _convert_mcp_schema("offline", SimpleNamespace(name="probe", description="offline", inputSchema=schema))["parameters"]
    name = "mcp__offline__probe" if path == "local-describe" else "connectors__offline__probe"
    fn = {"name": name, "description": "offline", "parameters": schema}
    definitions = [{"type": "function", "function": fn}]
    if path == "local-describe":
        registry.register(name=name, schema=fn, toolset="mcp-offline", handler=lambda args, **kw: "{}")
    else:
        definitions = [{"type": "function", "function": {"name": "manage_connections", "parameters": {}}}]
    def describe(names):
        assert names == [name]
        return SimpleNamespace(payload={"tools": {name: fn}}, failure=None)
    result = dispatch_tool_describe({"names": [name]}, current_tool_defs=definitions,
                                    config=ToolSearchConfig.from_raw({"enabled": "on"}), connector_describe=describe)
    return json.loads(result)["tools"][name]["parameters"]


def _assert_schema_refs_absent(node):
    if not isinstance(node, dict):
        return
    assert "$ref" not in node
    assert "$defs" not in node
    assert "definitions" not in node
    for key, value in node.items():
        if key in {"default", "const", "enum", "examples"}:
            continue
        if key in {"properties", "patternProperties", "dependentSchemas", "dependencies"} and isinstance(value, dict):
            for child in value.values():
                _assert_schema_refs_absent(child)
        elif isinstance(value, list):
            for child in value:
                _assert_schema_refs_absent(child)
        else:
            _assert_schema_refs_absent(value)


@pytest.mark.parametrize("path", ["normalize", "convert", "local-describe", "remote-describe"])
@pytest.mark.parametrize("case,schema,properties", list(_resolvable_cases()), ids=lambda v: v if isinstance(v, str) else None)
def test_resolvable_references_are_portable_without_changing_inputs(path, case, schema, properties):
    before = copy.deepcopy(schema)
    result = _through_path(path, schema)
    assert schema == before
    _assert_schema_refs_absent(result)
    if properties is not None:
        assert result["properties"] == properties
    if case == "annotation-data":
        assert result["properties"]["x"]["default"]["$ref"] == "literal"
    if case == "normalization" and path in {"normalize", "convert"}:
        assert result["properties"]["x"]["enum"] == ["a", "b"]
        assert result["properties"]["x"]["nullable"] is True
    if case != "normalization":
        from jsonschema import Draft202012Validator

        original_validator = Draft202012Validator(before)
        portable_validator = Draft202012Validator(result)
        for value in [None, False, 1, "a", "abcdef", [], [1], {}, {"y": 1}]:
            instance = {"x": value, "y": value, "nested": {"y": value}}
            assert portable_validator.is_valid(instance) == original_validator.is_valid(instance)
    result["properties"]["__output_mutation_probe"] = {"type": "string"}
    assert schema == before


@pytest.mark.parametrize("path", ["normalize", "convert", "local-describe", "remote-describe"])
@pytest.mark.parametrize("case,schema", list(_fallback_cases()), ids=lambda v: v if isinstance(v, str) else None)
def test_unsupported_references_preserve_the_entire_document(path, case, schema):
    before = copy.deepcopy(schema)
    assert _through_path(path, schema) == before
    assert schema == before
