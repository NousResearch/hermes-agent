from __future__ import annotations

import json

from agent.prompt_overhead import PromptOverheadModes


def _tool_def(name: str, description: str, parameter_description: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": parameter_description,
                    }
                },
                "required": ["query"],
            },
        },
    }


def test_tool_schema_modes_use_separate_cache_entries(monkeypatch):
    import model_tools

    description = "Perform a deliberately verbose operation. " + "details " * 30
    parameter_description = "A deliberately verbose parameter. " + "details " * 20
    calls = []
    state = {"mode": "minimal"}

    def compute(*args, **kwargs):
        calls.append((args, kwargs))
        return [_tool_def("cache_mode_tool", description, parameter_description)]

    monkeypatch.setattr(model_tools, "_compute_tool_definitions", compute)
    monkeypatch.setattr(
        model_tools,
        "get_prompt_overhead_modes",
        lambda: PromptOverheadModes(
            tool_schema_mode=state["mode"], platform="test-platform"
        ),
    )
    model_tools._clear_tool_defs_cache()
    try:
        minimal = model_tools.get_tool_definitions(quiet_mode=True)
        state["mode"] = "full"
        full = model_tools.get_tool_definitions(quiet_mode=True)
    finally:
        model_tools._clear_tool_defs_cache()

    assert len(calls) == 2
    assert (
        "description" not in minimal[0]["function"]["parameters"]["properties"]["query"]
    )
    assert full[0]["function"]["description"] == description
    assert (
        full[0]["function"]["parameters"]["properties"]["query"]["description"]
        == parameter_description
    )


def test_tool_describe_keeps_full_session_scoped_available_schema(monkeypatch):
    import model_tools
    from tools.registry import registry

    full_description = (
        "Perform a connected-service operation. FULL_SCHEMA_DESCRIPTION_SENTINEL"
    )
    parameter_description = (
        "The complete query syntax. FULL_PARAMETER_DESCRIPTION_SENTINEL"
    )

    def register(name, toolset, *, available=True):
        registry.register(
            name=name,
            toolset=toolset,
            handler=lambda args, **kwargs: json.dumps({"ok": True}),
            check_fn=lambda: available,
            schema=_tool_def(name, full_description, parameter_description)["function"],
        )

    register("mcp_prompt_visible", "mcp-prompt-visible")
    register("mcp_prompt_out_of_scope", "mcp-prompt-other")
    register("mcp_prompt_unavailable", "mcp-prompt-visible", available=False)
    monkeypatch.setattr(
        model_tools,
        "get_prompt_overhead_modes",
        lambda: PromptOverheadModes(
            tool_schema_mode="minimal", platform="test-platform"
        ),
    )
    model_tools._clear_tool_defs_cache()

    visible = json.loads(
        model_tools.handle_function_call(
            function_name="tool_describe",
            function_args={"name": "mcp_prompt_visible"},
            enabled_toolsets=["mcp-prompt-visible"],
        )
    )
    out_of_scope = json.loads(
        model_tools.handle_function_call(
            function_name="tool_describe",
            function_args={"name": "mcp_prompt_out_of_scope"},
            enabled_toolsets=["mcp-prompt-visible"],
        )
    )
    unavailable = json.loads(
        model_tools.handle_function_call(
            function_name="tool_describe",
            function_args={"name": "mcp_prompt_unavailable"},
            enabled_toolsets=["mcp-prompt-visible"],
        )
    )

    assert visible["description"] == full_description
    assert (
        visible["parameters"]["properties"]["query"]["description"]
        == parameter_description
    )
    assert "not currently available" in out_of_scope["error"]
    assert "not currently available" in unavailable["error"]
