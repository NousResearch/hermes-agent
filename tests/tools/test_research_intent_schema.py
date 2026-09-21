"""Model-facing schema contract for explicit research intent labels."""


def test_collection_tool_schemas_expose_optional_research_intent_without_mutating_input():
    from model_tools import _apply_dynamic_schemas

    definitions = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "terminal",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    rendered = _apply_dynamic_schemas(definitions)
    search = rendered[0]["function"]["parameters"]
    assert search["properties"]["research_intent"]["type"] == "string"
    assert "research_intent" not in search["required"]
    assert "research_intent" not in definitions[0]["function"]["parameters"]["properties"]
    assert "research_intent" not in rendered[1]["function"]["parameters"]["properties"]
