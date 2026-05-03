import json
from tools.memory_tool import MEMORY_SCHEMA


def test_memory_schema_requires_content_and_old_text_for_replace_action():
    schema = MEMORY_SCHEMA["parameters"]
    assert schema["required"] == ["action", "target"]

    all_of = schema.get("allOf")
    assert all_of, "memory schema should use conditional requirements"

    replace_requirements = [
        branch["then"].get("required", [])
        for branch in all_of
        if branch.get("if", {}).get("properties", {}).get("action", {}).get("const") == "replace"
    ]
    assert replace_requirements == [["old_text", "content"]]


def test_memory_schema_requires_content_for_add_action():
    add_requirements = [
        branch["then"].get("required", [])
        for branch in MEMORY_SCHEMA["parameters"].get("allOf", [])
        if branch.get("if", {}).get("properties", {}).get("action", {}).get("const") == "add"
    ]
    assert add_requirements == [["content"]]


def test_memory_schema_requires_old_text_for_remove_action():
    remove_requirements = [
        branch["then"].get("required", [])
        for branch in MEMORY_SCHEMA["parameters"].get("allOf", [])
        if branch.get("if", {}).get("properties", {}).get("action", {}).get("const") == "remove"
    ]
    assert remove_requirements == [["old_text"]]


def test_memory_schema_is_json_serializable():
    json.dumps(MEMORY_SCHEMA)
