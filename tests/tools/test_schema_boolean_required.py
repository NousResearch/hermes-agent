import copy

from jsonschema import Draft202012Validator
from tools.schema_sanitizer import sanitize_tool_schemas


def test_boolean_required_preserves_parent_intent():
    for required in ([], ["existing"]):
        schema = {"type": "object", "required": required, "properties": {
            "existing": {"type": "string"},
            "nested": {"type": "array", "items": {"properties": {
                "url path": {"required": True},
                "flag": {"type": "boolean", "required": False},
            }}},
        }}
        original = copy.deepcopy(schema)
        result = sanitize_tool_schemas([{"function": {"name": "probe", "parameters": schema}}])[0]["function"]["parameters"]
        Draft202012Validator.check_schema(result)
        item = result["properties"]["nested"]["items"]
        assert item["required"] == ["url_path"]
        assert result.get("required", []) == required
        assert schema == original


def test_boolean_required_does_not_modify_literal_data():
    literal = {"type": "object", "required": True, "properties": {"x": {"required": False}}}
    prop = {key: copy.deepcopy(literal) for key in ("const", "default", "example", "x-metadata")}
    prop.update(enum=[literal], examples=[literal])
    schema = {"type": "object", "properties": {"value": prop}}
    result = sanitize_tool_schemas([{"function": {"name": "probe", "parameters": schema}}])[0]["function"]["parameters"]
    assert result["properties"]["value"] == prop
