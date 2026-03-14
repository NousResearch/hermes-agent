"""
Tests for environments/tool_call_parsers/ — client-side tool call parsers.

These parsers extract structured tool_calls from raw model output text.
Used in Phase 2 (VLLM/generate) where the server returns raw tokens.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from environments.tool_call_parsers import (
        ParseResult,
        ToolCallParser,
        get_parser,
        list_parsers,
    )
except ImportError:
    pytest.skip("atroposlib not installed", allow_module_level=True)


# ─── Registry tests ─────────────────────────────────────────────────────

class TestParserRegistry:
    def test_list_parsers_returns_nonempty(self):
        parsers = list_parsers()
        assert len(parsers) > 0

    def test_hermes_parser_registered(self):
        parsers = list_parsers()
        assert "hermes" in parsers

    def test_get_parser_returns_instance(self):
        parser = get_parser("hermes")
        assert isinstance(parser, ToolCallParser)

    def test_get_parser_unknown_raises(self):
        with pytest.raises(KeyError):
            get_parser("nonexistent_parser_xyz")

    def test_all_registered_parsers_instantiate(self):
        """Every registered parser should be importable and instantiable."""
        for name in list_parsers():
            parser = get_parser(name)
            assert isinstance(parser, ToolCallParser)
            assert hasattr(parser, "parse")


# ─── Hermes parser tests ────────────────────────────────────────────────

class TestHermesParser:
    @pytest.fixture
    def parser(self):
        return get_parser("hermes")

    def test_no_tool_call(self, parser):
        text = "Hello, I can help you with that."
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = '<tool_call>{"name": "terminal", "arguments": {"command": "ls -la"}}</tool_call>'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "terminal"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["command"] == "ls -la"

    def test_tool_call_with_surrounding_text(self, parser):
        text = 'Let me check that for you.\n<tool_call>{"name": "terminal", "arguments": {"command": "pwd"}}</tool_call>'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "terminal"
        # Content should have the surrounding text
        if content is not None:
            assert "check that" in content or content.strip() != ""

    def test_multiple_tool_calls(self, parser):
        text = (
            '<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>\n'
            '<tool_call>{"name": "read_file", "arguments": {"path": "test.py"}}</tool_call>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = {tc.function.name for tc in tool_calls}
        assert "terminal" in names
        assert "read_file" in names

    def test_tool_call_ids_are_unique(self, parser):
        text = (
            '<tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>\n'
            '<tool_call>{"name": "terminal", "arguments": {"command": "pwd"}}</tool_call>'
        )
        _, tool_calls = parser.parse(text)
        assert tool_calls is not None
        ids = [tc.id for tc in tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs must be unique"

    def test_empty_string(self, parser):
        content, tool_calls = parser.parse("")
        assert tool_calls is None

    def test_malformed_json_in_tool_call(self, parser):
        text = '<tool_call>not valid json</tool_call>'
        content, tool_calls = parser.parse(text)
        # Should either return None tool_calls or handle gracefully
        # (implementation may vary — some parsers return error tool calls)

    def test_truncated_tool_call(self, parser):
        """Test handling of unclosed tool_call tag (model truncated mid-generation)."""
        text = '<tool_call>{"name": "terminal", "arguments": {"command": "ls -la"}'
        content, tool_calls = parser.parse(text)
        # Parser should handle truncated output gracefully
        # Either parse it successfully or return None


# ─── Parse result contract tests (applies to ALL parsers) ───────────────

class TestParseResultContract:
    """Ensure all parsers conform to the ParseResult contract."""

    @pytest.fixture(params=["hermes"])  # Add more as needed
    def parser(self, request):
        return get_parser(request.param)

    def test_returns_tuple_of_two(self, parser):
        result = parser.parse("hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_tools_returns_none_tool_calls(self, parser):
        content, tool_calls = parser.parse("Just plain text, no tools.")
        assert tool_calls is None
        assert content is not None

    def test_tool_calls_are_proper_objects(self, parser):
        """When tool calls are found, they should be ChatCompletionMessageToolCall objects."""
        # Use hermes format since that's universal
        text = '<tool_call>{"name": "terminal", "arguments": {"command": "echo hi"}}</tool_call>'
        content, tool_calls = parser.parse(text)
        if tool_calls is not None:
            for tc in tool_calls:
                assert hasattr(tc, "id")
                assert hasattr(tc, "function")
                assert hasattr(tc.function, "name")
                assert hasattr(tc.function, "arguments")
                assert tc.id is not None
                assert isinstance(tc.function.name, str)
                assert isinstance(tc.function.arguments, str)


# ─── DeepSeek V3 parser tests ───────────────────────────────────────────

class TestDeepSeekV3Parser:
    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_v3")

    def test_no_tool_call(self, parser):
        text = "Hello, how can I help you?"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = (
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n'
            '```json\n{"city": "London"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["city"] == "London"

    def test_multiple_tool_calls(self, parser):
        text = (
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n'
            '```json\n{"city": "London"}\n```<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n'
            '```json\n{"timezone": "UTC"}\n```<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"
        names = [tc.function.name for tc in tool_calls]
        assert "get_weather" in names
        assert "get_time" in names

    def test_tool_call_with_preceding_text(self, parser):
        text = (
            'Let me check that for you.\n'
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>terminal\n'
            '```json\n{"command": "ls"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
