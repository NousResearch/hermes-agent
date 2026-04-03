"""Tests for tool call validation to prevent session poisoning."""

import json
import pytest
from unittest.mock import MagicMock


class TestToolCallValidation:
    """Test _is_valid_tool_call and _sanitize_tool_calls_for_strict_api."""

    def _make_tool_call(self, name, arguments):
        """Create a mock tool call object."""
        tc = MagicMock()
        tc.function = MagicMock()
        tc.function.name = name
        tc.function.arguments = arguments
        return tc

    def test_valid_tool_call(self):
        """Valid tool calls should pass validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("read_file", '{"path": "/tmp/test.txt"}')
        assert Agent._is_valid_tool_call(tc) is True

    def test_empty_function_name(self):
        """Empty function name should fail validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("", '{"key": "value"}')
        assert Agent._is_valid_tool_call(tc) is False

    def test_whitespace_function_name(self):
        """Whitespace-only function name should fail validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("   ", '{"key": "value"}')
        assert Agent._is_valid_tool_call(tc) is False

    def test_none_function_name(self):
        """None function name should fail validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call(None, '{"key": "value"}')
        assert Agent._is_valid_tool_call(tc) is False

    def test_empty_arguments_string(self):
        """Empty arguments string should fail validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("test_tool", "")
        assert Agent._is_valid_tool_call(tc) is False

    def test_invalid_json_arguments(self):
        """Invalid JSON arguments should fail validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("test_tool", "{invalid json}")
        assert Agent._is_valid_tool_call(tc) is False

    def test_array_arguments(self):
        """Array arguments (not object) should fail validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("test_tool", '["item1", "item2"]')
        assert Agent._is_valid_tool_call(tc) is False

    def test_primitive_arguments(self):
        """Primitive arguments should fail validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("test_tool", '"just a string"')
        assert Agent._is_valid_tool_call(tc) is False

    def test_empty_object_arguments(self):
        """Empty object arguments {} should pass validation."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("test_tool", '{}')
        assert Agent._is_valid_tool_call(tc) is True

    def test_none_arguments(self):
        """None arguments should pass validation (provider may omit)."""
        from run_agent import AIAgent as Agent

        tc = self._make_tool_call("test_tool", None)
        assert Agent._is_valid_tool_call(tc) is True

    def test_missing_function_attribute(self):
        """Missing function attribute should fail validation."""
        from run_agent import AIAgent as Agent

        tc = MagicMock(spec=[])  # No function attribute
        assert Agent._is_valid_tool_call(tc) is False


class TestSanitizeToolCallsForStrictAPI:
    """Test the API sanitization filters malformed tool calls."""

    def test_filters_empty_function_name(self):
        """Malformed tool calls with empty name should be filtered."""
        from run_agent import AIAgent as Agent

        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "", "arguments": "{}"}},
                {"id": "2", "function": {"name": "valid_tool", "arguments": "{}"}},
            ]
        }
        result = Agent._sanitize_tool_calls_for_strict_api(api_msg)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "valid_tool"

    def test_filters_invalid_json_arguments(self):
        """Tool calls with invalid JSON arguments should be filtered."""
        from run_agent import AIAgent as Agent

        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "bad_tool", "arguments": "not json"}},
                {"id": "2", "function": {"name": "good_tool", "arguments": '{"x": 1}'}},
            ]
        }
        result = Agent._sanitize_tool_calls_for_strict_api(api_msg)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "good_tool"

    def test_filters_array_arguments(self):
        """Tool calls with array arguments should be filtered."""
        from run_agent import AIAgent as Agent

        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "array_args", "arguments": '["a", "b"]'}},
                {"id": "2", "function": {"name": "obj_args", "arguments": '{"a": "b"}'}},
            ]
        }
        result = Agent._sanitize_tool_calls_for_strict_api(api_msg)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "obj_args"

    def test_strips_extra_keys(self):
        """Extra keys like call_id should be stripped."""
        from run_agent import AIAgent as Agent

        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "call_id": "extra_1",
                    "response_item_id": "extra_2",
                    "function": {"name": "test", "arguments": "{}"}
                },
            ]
        }
        result = Agent._sanitize_tool_calls_for_strict_api(api_msg)
        assert len(result["tool_calls"]) == 1
        assert "call_id" not in result["tool_calls"][0]
        assert "response_item_id" not in result["tool_calls"][0]
        assert result["tool_calls"][0]["id"] == "1"

    def test_preserves_valid_tool_calls(self):
        """Valid tool calls should be preserved unchanged (except stripped keys)."""
        from run_agent import AIAgent as Agent

        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "read", "arguments": '{"file": "x.txt"}'}},
                {"id": "2", "type": "function", "function": {"name": "write", "arguments": '{"file": "y.txt", "content": "hi"}'}},
            ]
        }
        result = Agent._sanitize_tool_calls_for_strict_api(api_msg)
        assert len(result["tool_calls"]) == 2

    def test_handles_empty_tool_calls(self):
        """Empty tool_calls list should be handled gracefully."""
        from run_agent import AIAgent as Agent

        api_msg = {"role": "assistant", "tool_calls": []}
        result = Agent._sanitize_tool_calls_for_strict_api(api_msg)
        assert result["tool_calls"] == []

    def test_handles_missing_tool_calls(self):
        """Missing tool_calls key should be handled gracefully."""
        from run_agent import AIAgent as Agent

        api_msg = {"role": "assistant", "content": "Hello"}
        result = Agent._sanitize_tool_calls_for_strict_api(api_msg)
        assert "tool_calls" not in result or result.get("tool_calls") is None
