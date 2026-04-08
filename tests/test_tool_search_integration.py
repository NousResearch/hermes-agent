"""Integration tests for tool search deferred loading pipeline.

Covers should_defer_tools(), _estimate_tool_tokens(), and
build_tool_catalog_prompt() working together.
"""

import json

from model_tools import should_defer_tools, _estimate_tool_tokens
from agent.prompt_builder import build_tool_catalog_prompt


class TestShouldDeferTools:
    def test_auto_defers_when_over_threshold(self):
        assert should_defer_tools(5000, 32768, "auto", 0.10) is True

    def test_auto_does_not_defer_under_threshold(self):
        assert should_defer_tools(1000, 200000, "auto", 0.10) is False

    def test_always_mode(self):
        assert should_defer_tools(1, 999999, "always", 0.10) is True

    def test_never_mode(self):
        assert should_defer_tools(999999, 1, "never", 0.10) is False

    def test_zero_context_length(self):
        assert should_defer_tools(5000, 0, "auto", 0.10) is False

    def test_boundary_exactly_at_threshold(self):
        # 3277 / 32768 ≈ 0.1000 — right at threshold, should not defer
        assert should_defer_tools(3276, 32768, "auto", 0.10) is False

    def test_boundary_just_over(self):
        assert should_defer_tools(3277, 32768, "auto", 0.10) is True


class TestEstimateToolTokens:
    def test_empty_list(self):
        assert _estimate_tool_tokens([]) == 0

    def test_rough_estimate(self):
        defs = [
            {"type": "function", "function": {"name": "test", "description": "x" * 400}},
        ]
        tokens = _estimate_tool_tokens(defs)
        assert tokens > 0
        # ~400 chars desc + overhead, /4 ≈ ~120 tokens
        assert 80 < tokens < 300


class TestBuildToolCatalogPrompt:
    def test_empty_catalog(self):
        assert build_tool_catalog_prompt([]) == ""

    def test_groups_by_toolset(self):
        catalog = [
            {"name": "read_file", "description": "Read a file", "toolset": "file"},
            {"name": "write_file", "description": "Write a file", "toolset": "file"},
            {"name": "web_search", "description": "Search the web", "toolset": "web"},
        ]
        prompt = build_tool_catalog_prompt(catalog)
        assert "file:" in prompt
        assert "web:" in prompt
        assert "read_file" in prompt
        assert "web_search" in prompt

    def test_excludes_internal_toolsets(self):
        catalog = [
            {"name": "tool_search", "description": "Search tools", "toolset": "_tool_search"},
            {"name": "real_tool", "description": "Real", "toolset": "core"},
        ]
        prompt = build_tool_catalog_prompt(catalog)
        assert "_tool_search" not in prompt
        assert "real_tool" in prompt

    def test_truncates_long_descriptions(self):
        catalog = [
            {"name": "verbose_tool", "description": "A" * 200, "toolset": "misc"},
        ]
        prompt = build_tool_catalog_prompt(catalog)
        assert "..." in prompt

    def test_contains_usage_instructions(self):
        catalog = [
            {"name": "t", "description": "test", "toolset": "s"},
        ]
        prompt = build_tool_catalog_prompt(catalog)
        assert "tool_details" in prompt
        assert "tool_search" in prompt

    def test_available_tools_header(self):
        catalog = [
            {"name": "t", "description": "test", "toolset": "s"},
        ]
        prompt = build_tool_catalog_prompt(catalog)
        assert "## Available Tools" in prompt
        assert "<available_tools>" in prompt
