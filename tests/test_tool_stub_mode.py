"""Tests for tools.stub_mode — _apply_stub_mode() and the stub_mode param on
get_tool_definitions().

Invariants:
- Core tools (_STUB_MODE_FULL_TOOLS) keep full schemas (parameters present with properties).
- Non-core tools are reduced to a minimal stub: {name, description, parameters: {type:object, properties:{}}}.
- The description of a stubbed tool is annotated with the delegation hint.
- stub_mode=False (default) leaves all schemas intact.
- stub_mode is cache-keyed independently — a full call and a stub call never share a cached object.
- _apply_stub_mode is safe on an empty list.
- Subagents (platform="subagent") always receive full schemas regardless of config.
- stub_mode bypasses Tool Search assembly so the full catalog is available to stub.
"""
from __future__ import annotations

import pytest

import model_tools
from model_tools import _apply_stub_mode, _STUB_MODE_FULL_TOOLS

_MINIMAL_STUB_PARAMS = {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, *, extra_param: bool = True) -> dict:
    fn: dict = {"name": name, "description": f"Tool {name}."}
    if extra_param:
        fn["parameters"] = {
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
            "required": ["arg1"],
        }
    return {"type": "function", "function": fn}


def _is_stub(tool: dict) -> bool:
    """True if this tool entry carries only the minimal stub parameters."""
    params = tool["function"].get("parameters", {})
    return params == _MINIMAL_STUB_PARAMS


def _is_full(tool: dict) -> bool:
    """True if the tool has a non-trivial parameters schema (real properties)."""
    params = tool["function"].get("parameters", {})
    return bool(params.get("properties"))


# Pick one known core tool and one known non-core tool for parametrised tests.
_CORE_TOOL = "delegate_task"
_NON_CORE_TOOL = "terminal"


# ---------------------------------------------------------------------------
# _apply_stub_mode unit tests
# ---------------------------------------------------------------------------

class TestApplyStubMode:
    def test_empty_list_is_safe(self):
        assert _apply_stub_mode([]) == []

    def test_core_tool_keeps_full_schema(self):
        tool = _make_tool(_CORE_TOOL)
        result = _apply_stub_mode([tool])
        assert len(result) == 1
        assert _is_full(result[0]), "Core tool must retain its full parameters schema."

    def test_non_core_tool_is_reduced_to_stub(self):
        tool = _make_tool(_NON_CORE_TOOL)
        result = _apply_stub_mode([tool])
        assert len(result) == 1
        fn = result[0]["function"]
        assert fn["name"] == _NON_CORE_TOOL
        assert fn["description"]
        # Stub must carry a minimal valid parameters shape — not the real one.
        assert fn.get("parameters") == _MINIMAL_STUB_PARAMS, (
            "Stubbed tool must have a minimal valid parameters shape, not the real schema."
        )

    def test_stub_description_includes_delegation_hint(self):
        tool = _make_tool(_NON_CORE_TOOL)
        result = _apply_stub_mode([tool])
        desc = result[0]["function"]["description"]
        assert "delegate_task" in desc, (
            "Stubbed description must hint that the tool must be invoked via delegate_task."
        )

    def test_mixed_tools_split_correctly(self):
        """Core tools keep full schemas; non-core tools carry minimal stub params."""
        core = [_make_tool(name) for name in _STUB_MODE_FULL_TOOLS]
        non_core = [_make_tool("fake_non_core_tool")]
        result = _apply_stub_mode(core + non_core)
        for item in result:
            name = item["function"]["name"]
            if name in _STUB_MODE_FULL_TOOLS:
                assert _is_full(item), f"{name} must keep full schema"
            else:
                assert item["function"].get("parameters") == _MINIMAL_STUB_PARAMS, (
                    f"{name} must carry minimal stub parameters"
                )

    def test_type_field_preserved(self):
        """The 'type': 'function' wrapper must survive stubbing."""
        tool = _make_tool(_NON_CORE_TOOL)
        result = _apply_stub_mode([tool])
        assert result[0].get("type") == "function"

    def test_tool_without_parameters_field_is_safe(self):
        """A tool that already has no parameters field must not crash."""
        tool = _make_tool(_NON_CORE_TOOL, extra_param=False)
        result = _apply_stub_mode([tool])
        fn = result[0]["function"]
        assert fn["name"] == _NON_CORE_TOOL
        # Still gets the minimal stub parameters.
        assert fn.get("parameters") == _MINIMAL_STUB_PARAMS

    def test_stub_parameters_is_valid_json_schema(self):
        """Minimal stub parameters must satisfy strict provider schema validation."""
        tool = _make_tool(_NON_CORE_TOOL)
        result = _apply_stub_mode([tool])
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)


# ---------------------------------------------------------------------------
# get_tool_definitions(stub_mode=...) integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    model_tools._tool_defs_cache.clear()
    yield
    model_tools._tool_defs_cache.clear()


class TestGetToolDefinitionsStubMode:
    def test_stub_mode_false_returns_full_schemas(self):
        tools = model_tools.get_tool_definitions(quiet_mode=True, stub_mode=False)
        non_core = [
            t for t in tools
            if t["function"]["name"] not in _STUB_MODE_FULL_TOOLS
        ]
        assert any(_is_full(t) for t in non_core), (
            "stub_mode=False must leave full schemas intact."
        )

    def test_stub_mode_true_strips_non_core_parameters(self):
        tools = model_tools.get_tool_definitions(quiet_mode=True, stub_mode=True)
        for t in tools:
            name = t["function"]["name"]
            if name not in _STUB_MODE_FULL_TOOLS:
                assert not _is_full(t), (
                    f"stub_mode=True: {name} must not expose real parameter properties."
                )
                assert t["function"].get("parameters") == _MINIMAL_STUB_PARAMS, (
                    f"stub_mode=True: {name} must carry minimal valid parameters."
                )

    def test_stub_mode_true_preserves_core_tools(self):
        tools = model_tools.get_tool_definitions(quiet_mode=True, stub_mode=True)
        by_name = {t["function"]["name"]: t for t in tools}
        for core_name in _STUB_MODE_FULL_TOOLS:
            if core_name in by_name:
                assert _is_full(by_name[core_name]), (
                    f"stub_mode=True: core tool {core_name} must retain full parameters."
                )

    def test_stub_and_full_use_separate_cache_entries(self):
        """stub_mode=True and stub_mode=False must not share a cache slot."""
        model_tools.get_tool_definitions(quiet_mode=True, stub_mode=False)
        model_tools.get_tool_definitions(quiet_mode=True, stub_mode=True)
        assert len(model_tools._tool_defs_cache) == 2, (
            "stub_mode must be part of the cache key — full and stub results must not overwrite each other."
        )

    def test_stub_mode_skip_tool_search_assembly_respected(self):
        """When skip_tool_search_assembly=True, stub mode is not applied
        (that path serves the bridge catalog reader, not the main agent)."""
        tools_bridge = model_tools.get_tool_definitions(
            quiet_mode=True,
            stub_mode=True,
            skip_tool_search_assembly=True,
        )
        non_core_full = [
            t for t in tools_bridge
            if t["function"]["name"] not in _STUB_MODE_FULL_TOOLS
            and _is_full(t)
        ]
        assert non_core_full, (
            "skip_tool_search_assembly=True must bypass stub mode — "
            "the bridge reader needs full schemas."
        )

    def test_stub_mode_returns_all_tools_not_proxied(self):
        """stub_mode bypasses Tool Search assembly — the returned list must not
        contain tool_search / tool_describe / tool_call proxy entries."""
        tools = model_tools.get_tool_definitions(quiet_mode=True, stub_mode=True)
        tool_names = {t["function"]["name"] for t in tools}
        proxy_names = {"tool_search", "tool_describe", "tool_call"}
        assert not (tool_names & proxy_names), (
            "stub_mode must bypass Tool Search so proxy entries don't replace real tool stubs."
        )
