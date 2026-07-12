"""Tests for tools/reasoning_effort_tool.py — validation and dispatch."""

import json

from tools.reasoning_effort_tool import (
    REASONING_EFFORT_SCHEMA,
    check_reasoning_effort_requirements,
    reasoning_effort_tool,
)


class TestReasoningEffortTool:
    def test_requirements_always_met(self):
        assert check_reasoning_effort_requirements() is True

    def test_missing_level_errors(self):
        result = json.loads(reasoning_effort_tool("", callback=lambda *a, **k: None))
        assert result["error"] == "level is required"

    def test_invalid_level_errors_with_valid_levels(self):
        result = json.loads(
            reasoning_effort_tool("galaxy-brain", callback=lambda *a, **k: None)
        )
        assert "invalid level" in result["error"]
        assert "none" in result["valid_levels"]
        assert "medium" in result["valid_levels"]

    def test_no_callback_errors(self):
        result = json.loads(reasoning_effort_tool("high"))
        assert "not available" in result["error"]

    def test_level_is_normalized_before_dispatch(self):
        seen = {}

        def _cb(parsed, *, level, persist):
            seen.update(parsed=parsed, level=level, persist=persist)
            return {"success": True}

        reasoning_effort_tool("  HIGH  ", callback=_cb)
        assert seen["level"] == "high"
        assert seen["parsed"] == {"enabled": True, "effort": "high"}
        assert seen["persist"] is False

    def test_none_parses_to_disabled(self):
        seen = {}

        def _cb(parsed, *, level, persist):
            seen.update(parsed=parsed)
            return {"success": True}

        reasoning_effort_tool("none", callback=_cb)
        assert seen["parsed"] == {"enabled": False}

    def test_persist_flag_forwarded(self):
        seen = {}

        def _cb(parsed, *, level, persist):
            seen.update(persist=persist)
            return {"success": True}

        reasoning_effort_tool("low", persist=True, callback=_cb)
        assert seen["persist"] is True

    def test_dict_result_serialized(self):
        result = reasoning_effort_tool(
            "low", callback=lambda parsed, *, level, persist: {"success": True, "level": level}
        )
        assert json.loads(result) == {"success": True, "level": "low"}

    def test_string_result_passthrough(self):
        result = reasoning_effort_tool(
            "low", callback=lambda parsed, *, level, persist: '{"raw": true}'
        )
        assert result == '{"raw": true}'

    def test_callback_exception_becomes_error(self):
        def _cb(parsed, *, level, persist):
            raise RuntimeError("boom")

        result = json.loads(reasoning_effort_tool("low", callback=_cb))
        assert "Failed to set reasoning effort: boom" in result["error"]

    def test_schema_levels_match_parser(self):
        from hermes_constants import VALID_REASONING_EFFORTS

        schema_levels = REASONING_EFFORT_SCHEMA["parameters"]["properties"]["level"]["enum"]
        assert schema_levels == ["none", *VALID_REASONING_EFFORTS]

    def test_registered_in_registry(self):
        # Other tests may reset the global registry; reload the module so its
        # import-time registry.register(...) runs against the current registry.
        import importlib

        import tools.reasoning_effort_tool as _mod
        from tools.registry import registry

        if registry.get_entry("reasoning_effort") is None:
            importlib.reload(_mod)

        entry = registry.get_entry("reasoning_effort")
        assert entry is not None
        assert entry.toolset == "reasoning"
