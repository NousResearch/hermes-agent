"""MCP tool metadata reaches the model on every API call: invisible TAG chars and bidi controls in
the description and in every free-text schema string must not survive registration, while the
strings the server matches calls against (property names, enum values) stay exact and a clean
schema stays the same object (schemas are frozen wire bytes for prompt caching)."""

import json
from types import SimpleNamespace

from tools.mcp_tool_schema import _convert_mcp_schema, strip_schema_annotations

HIDDEN = "\U000e0069\U000e0067\u202e\u2066"  # TAG 'i','g' + RLO + LRI: invisible to the user
FAMILY = "\U0001f468\u200d\U0001f469\u200d\U0001f467"  # ZWJ emoji sequence: must survive


def _invisible(text: str) -> bool:
    return any("\U000e0000" <= c <= "\U000e007f" or c in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
               for c in text)


def test_tool_schema_strips_hidden_free_text_but_keeps_the_call_contract():
    tagged_key, tagged_choice = f"q{HIDDEN}", f"fast{HIDDEN}"
    tool = SimpleNamespace(name="lookup", description=f"Look up{HIDDEN} a record {FAMILY}", inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "title": f"Q{HIDDEN}", "description": f"Search text{HIDDEN}",
                      "default": f"d{HIDDEN}",
                      # Instance data: a keyword-named key inside it is plain data, not the keyword.
                      "examples": [{"propertyName": f"x{HIDDEN}", "enum": f"y{HIDDEN}"}]},
            "mode": {"anyOf": [{"type": "string", "enum": [tagged_choice, "slow"],
                                "description": f"speed{HIDDEN}"}, {"type": "null"}]},
            tagged_key: {"type": "string", "description": f"legacy{HIDDEN}"},
        },
        "required": ["query"],
        "dependentRequired": {"query": [tagged_key]},
        "discriminator": {"propertyName": tagged_key},
        # draft-07: an array value names properties, an object value is a sub-schema.
        "dependencies": {"mode": [tagged_key], "query": {"description": f"dep7{HIDDEN}"}},
        # An argument literally named "enum" in a name->schema map is a name, not the keyword.
        "dependentSchemas": {"enum": {"description": f"dep{HIDDEN}"}},
    })
    schema = _convert_mcp_schema("srv", tool)
    props = schema["parameters"]["properties"]
    free_text = [schema["description"], props["query"]["title"], props["query"]["description"],
                 props["query"]["default"], props["mode"]["description"], props[tagged_key]["description"],
                 schema["parameters"]["dependentSchemas"]["enum"]["description"],
                 schema["parameters"]["dependencies"]["query"]["description"],
                 *props["query"]["examples"][0].values()]
    assert not any(_invisible(text) for text in free_text)
    assert props["query"]["description"] == "Search text" and FAMILY in schema["description"]
    assert tagged_key in props and props["mode"]["enum"] == [tagged_choice, "slow"]
    # Strings that name a property stay byte-identical to that property's key.
    assert schema["parameters"]["dependentRequired"]["query"] == [tagged_key]
    assert schema["parameters"]["discriminator"]["propertyName"] == tagged_key
    assert schema["parameters"]["dependencies"]["mode"] == [tagged_key]
    json.dumps(schema)


def test_clean_schema_is_returned_unchanged():
    schema = {"type": "object", "properties": {"q": {"type": "string", "description": f"emoji {FAMILY}",
                                                     "enum": ["a", "b"]}}, "required": ["q"]}
    assert strip_schema_annotations(schema) is schema
