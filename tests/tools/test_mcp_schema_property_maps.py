"""Schema keywords in keyed maps and instance data are not schema nodes."""

from copy import deepcopy

import pytest
from jsonschema import Draft7Validator
from mcp.types import Tool as SDKMCPTool

from tools.mcp_tool_schema import _convert_mcp_schema, _normalize_mcp_input_schema
from tools.schema_sanitizer import (
    sanitize_property_key, sanitize_tool_schemas, strip_pattern_and_format, strip_slash_enum,
)


@pytest.mark.parametrize("data", [
    {"properties": {"required": "value"}, "required": ["missing"],
     "definitions": {"type": "literal"}},
    {"anyOf": [{"const": "a"}, {"const": "b"}]},
    {"oneOf": [{"type": "string"}, {"type": "null"}],
     "default": {"$ref": "#/definitions/literal", "default": "keep"}},
    {"type": "object", "required": ["missing"], "properties": {"type": "object"}},
    [{"type": ["number", "string"], "pattern": "keep", "format": "keep",
      "enum": ["vendor/model"]}, "object", ["string"]],
    {"type": "string", "pattern": "keep", "format": "keep", "enum": ["vendor/model"]},
])
@pytest.mark.parametrize("placement", ["root", "items", "allOf", "definitions", "$defs"])
def test_conversion_preserves_schema_maps_and_instance_data(placement, data):
    names = ["properties", "required", "type", "definitions", "$defs", "items"]
    obj = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in names},
        "required": names,
        "additionalProperties": False,
        "default": deepcopy(data),
        "examples": [deepcopy(data)],
    }
    instance = {name: "value" for name in names}
    if placement == "root":
        schema, valid, invalid = obj, instance, {}
    elif placement == "items":
        schema = {"type": "array", "items": obj, "minItems": 1}
        valid, invalid = [instance], [{}]
    elif placement == "allOf":
        schema, valid, invalid = {"allOf": [obj]}, instance, {}
    else:
        schema = {placement: {"definitions": obj},
                  "$ref": f"#/{placement}/definitions"}
        valid, invalid = instance, {}
    # Exercise literal objects as constraints too, not only annotations.
    schema = {"type": "object", "properties": {
        "payload": schema, "literal": {"enum": [data], "const": data},
        "tree": {"$ref": "#/definitions/Node"}},
        "required": ["payload", "literal"],
        "definitions": {"Node": {"type": "object", "properties": {
            "next": {"$ref": "#/definitions/Node"}}}}}
    if placement in ("definitions", "$defs"):
        definitions = schema["properties"]["payload"].pop(placement)
        definitions.update(schema.pop("definitions"))
        schema[placement] = definitions
        schema["properties"]["payload"]["$ref"] = f"#/{placement}/definitions"
        schema["properties"]["tree"]["$ref"] = f"#/{placement}/Node"
        definitions["Node"]["properties"]["next"]["$ref"] = f"#/{placement}/Node"
    original = deepcopy(schema)
    Draft7Validator.check_schema(schema)
    assert Draft7Validator(schema).is_valid({"payload": valid, "literal": data})
    tool = SDKMCPTool(name="schema_map_probe", inputSchema=deepcopy(schema))
    converted = _convert_mcp_schema("synthetic", tool)
    tools = [{"type": "function", "function": converted}]
    before = deepcopy(tools)
    final = sanitize_tool_schemas(tools)[0]["function"]["parameters"]
    assert tools == before
    for result in (_normalize_mcp_input_schema(schema), converted["parameters"], final):
        Draft7Validator.check_schema(result)
        validator = Draft7Validator(result)
        expected_names = [sanitize_property_key(name) for name in names] if result is final else names
        expected_instance = dict(zip(expected_names, instance.values()))
        expected_valid = [expected_instance] if placement == "items" else expected_instance
        assert validator.is_valid({"payload": expected_valid, "literal": data, "tree": {"next": {}}})
        assert not validator.is_valid({"payload": invalid, "literal": data})
        assert not validator.is_valid({"payload": expected_valid, "literal": {}})
        payload = result["properties"]["payload"]
        if placement == "items":
            payload = payload["items"]
        elif placement == "allOf":
            payload = payload["allOf"][0]
        elif placement in ("definitions", "$defs"):
            payload = result["$defs"]["definitions"]
        assert payload["properties"] == {name: {"type": "string"} for name in expected_names}
        assert payload["required"] == expected_names
        assert payload["default"] == data
        assert payload["examples"] == [data]
        assert result["properties"]["literal"] == {"enum": [data], "const": data}
        assert result["$defs"]["Node"]["properties"]["next"] == {
            "$ref": "#/$defs/Node"}
    # Recovery strips schema constraints only, never similarly named literal keys
    # or properties in a keyword-keyed map. Exercise both accepted wire formats.
    recovery_schema = deepcopy(final)
    recovery_schema["properties"]["pattern"] = {"type": "string", "pattern": "x", "format": "uri"}
    recovery_schema["properties"]["enum"] = {"type": "string", "enum": ["vendor/model"]}
    recovery = [{"function": {"parameters": deepcopy(recovery_schema)}},
                {"parameters": deepcopy(recovery_schema)}]
    for strip, expected_count in ((strip_pattern_and_format, 4), (strip_slash_enum, 2)):
        returned, count = strip(recovery)
        assert returned is recovery
        assert count == expected_count
        for entry in recovery:
            params = entry.get("function", entry)["parameters"]
            assert params["properties"]["literal"] == final["properties"]["literal"]
            assert params["properties"]["payload"] == final["properties"]["payload"]
            assert "pattern" in params["properties"] and "enum" in params["properties"]
    assert schema == original
    assert tool.model_dump(by_alias=True)["inputSchema"] == original


@pytest.mark.parametrize("keyword,container", [
    ("properties", "map"), ("patternProperties", "map"),
    ("definitions", "map"), ("$defs", "map"),
    ("dependentSchemas", "map"), ("dependencies", "map"),
    ("items", "single"), ("items", "list"),
    ("additionalItems", "single"), ("additionalProperties", "single"),
    ("contains", "single"), ("propertyNames", "single"),
    ("not", "single"), ("if", "single"), ("then", "single"), ("else", "single"),
    ("allOf", "list"), ("anyOf", "list"), ("oneOf", "list"),
    ("prefixItems", "list"), ("unevaluatedItems", "single"),
    ("unevaluatedProperties", "single"), ("contentSchema", "single"),
    *[(name, "constraint") for name in (
        "required", "explicit", "aliases", "conditional", "allOf-peer",
        "anyOf-peer", "oneOf-peer", "dependencies", "not-scalar")],
    *[(order, "type-conjunction") for order in (
        "type-first", "anyof-first", "allof-type-first", "allof-anyof-first")],
])
def test_repairs_only_schema_children(keyword, container):
    if container == "type-conjunction":
        parts = [("type", ["number", "string"]), ("anyOf", [
            {"type": "number", "minimum": 5},
            {"type": "string", "minLength": 3},
            {"type": "boolean"},
        ])]
        if keyword.endswith("anyof-first"):
            parts.reverse()
        constraint = dict(parts)
        valid, invalid = [6, "long"], [1, "x", True, False, None]
        if keyword.startswith("allof-"):
            constraint["allOf"] = [{"not": {"enum": [6, "long"]}}]
            valid, invalid = [7, "longer"], [*invalid, 6, "long"]
        schema = {"type": "object", "properties": {"payload": constraint},
                  "required": ["payload"]}
        original = deepcopy(schema)
        tool = SDKMCPTool(name="conjunction_probe", inputSchema=deepcopy(schema))
        converted = _convert_mcp_schema("synthetic", tool)
        wrapped = [{"type": "function", "function": converted}]
        before = deepcopy(wrapped)
        final = sanitize_tool_schemas(wrapped)[0]["function"]["parameters"]
        for result in (schema, _normalize_mcp_input_schema(schema),
                       converted["parameters"], final):
            Draft7Validator.check_schema(result)
            validator = Draft7Validator(result)
            assert all(validator.is_valid({"payload": value}) for value in valid)
            assert all(not validator.is_valid({"payload": value}) for value in invalid)
        assert wrapped == before
        assert schema == original
        assert tool.model_dump(by_alias=True)["inputSchema"] == original
        return
    if container == "constraint":
        # Required names may be declared by a parent/peer, or not declared at all.
        constraints = {
            "required": ({"required": ["query"]}, [{"query": "synthetic"}], [{}]),
            "explicit": ({"type": "object", "required": ["query"]},
                         [{"query": "synthetic"}], [{}]),
            "aliases": ({"allOf": [
                {"not": {"required": ["limit", "pageLimit"]}},
                {"not": {"required": ["page", "searchAfter"]}}]},
                [{}, {"query": "synthetic"}, {"limit": 1}, {"pageLimit": 1},
                 {"page": 1}, {"searchAfter": []}],
                [{"limit": 1, "pageLimit": 1}, {"page": 1, "searchAfter": []}]),
            "conditional": ({"if": {"required": ["page"]},
                             "then": {"required": ["limit"]},
                             "else": {"required": ["searchAfter"]}},
                            [{"page": 1, "limit": 1}, {"searchAfter": []}],
                            [{}, {"page": 1}, {"limit": 1}]),
            "allOf-peer": ({"allOf": [{"properties": {"query": {"type": "string"}}},
                                      {"required": ["query"]}]},
                           [{"query": "synthetic"}], [{}, {"query": 1}]),
            "anyOf-peer": ({"anyOf": [{"required": ["limit"]}, {"required": ["page"]}]},
                           [{"limit": 1}, {"page": 1}, {"page": 1, "limit": 1}], [{}]),
            "oneOf-peer": ({"oneOf": [{"required": ["limit"]}, {"required": ["page"]}]},
                           [{"limit": 1}, {"page": 1}], [{}, {"page": 1, "limit": 1}]),
            "dependencies": ({"dependencies": {"page": {"required": ["limit"]}}},
                             [{}, {"page": 1, "limit": 1}], [{"page": 1}]),
            # Inferring object from required changes negation on non-objects too.
            "not-scalar": ({"not": {"required": ["query"]}}, [{}], ["text", []]),
        }
        constraint, valid, invalid = constraints[keyword]
        schema = {"type": "object", "properties": {name: {} for name in (
            "query", "limit", "pageLimit", "page", "searchAfter")}, **constraint}
        if keyword in ("required", "explicit", "not-scalar"):
            schema = constraint
        original = deepcopy(schema)
        tool = SDKMCPTool(name="required_probe", inputSchema=deepcopy(schema))
        for result in (schema, _normalize_mcp_input_schema(schema),
                       _convert_mcp_schema("synthetic", tool)["parameters"]):
            Draft7Validator.check_schema(result)
            validator = Draft7Validator(result)
            assert all(validator.is_valid(value) for value in valid)
            assert all(not validator.is_valid(value) for value in invalid)
            # Nest so Codex's documented top-level combinator stripping does not
            # remove the constraint being checked by the final sanitizer.
            tools = [{"type": "function", "function": {"name": "probe", "parameters": {
                "type": "object", "properties": {"payload": result}}}}]
            before = deepcopy(tools)
            final = sanitize_tool_schemas(tools)[0]["function"]["parameters"]["properties"]["payload"]
            assert all(Draft7Validator(final).is_valid(value) for value in valid)
            assert all(not Draft7Validator(final).is_valid(value) for value in invalid)
            assert tools == before
        assert schema == original
        assert tool.model_dump(by_alias=True)["inputSchema"] == original
        return
    literal = {"anyOf": [{"const": "a"}, {"const": "b"}]}
    child = {"properties": {"required": {"type": "string"},
                            "empty": {"type": "object", "required": ["gone"]},
                            "choice": deepcopy(literal),
                            "optional": {"oneOf": [{"type": "string"}, {"type": "null"}]}},
             "default": literal, "examples": [literal],
             "required": ["required", "missing"]}
    value = {"properties": child, "boolean": False} if container == "map" else (
        [child, False] if container == "list" else child)
    schema = {"type": "object", keyword: value}
    if keyword == "dependencies":
        value["required"] = ["properties"]
    original = deepcopy(schema)
    tool = SDKMCPTool(name="repair_probe", inputSchema=deepcopy(schema))
    normalized = _normalize_mcp_input_schema(schema)
    wrapped = [{"function": {"name": "probe", "parameters": {
        "type": "object", "properties": {"payload": normalized}}}}]
    before = deepcopy(wrapped)
    final = sanitize_tool_schemas(wrapped)[0]["function"]["parameters"]["properties"]["payload"]
    assert wrapped == before
    for result in (normalized, _convert_mcp_schema("synthetic", tool)["parameters"], final):
        Draft7Validator.check_schema(result)
        value_out = result["$defs" if keyword == "definitions" else keyword]
        repaired = value_out["properties"] if container == "map" else (
            value_out[0] if container == "list" else value_out)
        assert repaired["type"] == "object"
        assert repaired["required"] == ["required", "missing"]
        assert repaired["properties"]["empty"] == {
            "type": "object", "properties": {}, "required": ["gone"]}
        assert repaired["default"] == literal
        assert repaired["examples"] == [literal]
        assert repaired["properties"]["choice"] == {"type": "string", "enum": ["a", "b"]}
        assert repaired["properties"]["optional"] == {"type": "string", "nullable": True}
        assert Draft7Validator(repaired).is_valid({"required": "value", "missing": 1, "choice": "a"})
        assert not Draft7Validator(repaired).is_valid({"required": "value"})
        assert not Draft7Validator(repaired).is_valid({"required": "value", "missing": 1, "empty": {}})
        assert not Draft7Validator(repaired).is_valid({"required": "value", "missing": 1, "choice": "c"})
        assert not Draft7Validator(repaired).is_valid({})
        assert not Draft7Validator(repaired).is_valid({"required": 7})
        if container == "map":
            assert value_out.keys() == value.keys()
            assert value_out["boolean"] is False
        elif container == "list":
            assert value_out[1] is False
        if keyword == "dependencies":
            assert value_out["required"] == ["properties"]
    assert schema == original
    assert tool.model_dump(by_alias=True)["inputSchema"] == original
