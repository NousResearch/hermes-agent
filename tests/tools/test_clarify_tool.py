"""Tests for tools/clarify_tool.py - Interactive clarifying questions."""

import json
from typing import List, Optional

import pytest

from tools.clarify_tool import (
    clarify_tool,
    check_clarify_requirements,
    MAX_CHOICES,
    CLARIFY_SCHEMA,
)


class TestClarifyToolBasics:
    """Basic functionality tests for clarify_tool."""

    def test_simple_question_with_callback(self):
        """Should return user response for simple question."""
        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            assert question == "What color?"
            assert choices is None
            return "blue"

        result = json.loads(clarify_tool("What color?", callback=mock_callback))
        assert result["question"] == "What color?"
        assert result["choices_offered"] is None
        assert result["user_response"] == "blue"

    def test_question_with_choices(self):
        """Should pass choices to callback and return response."""
        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            assert question == "Pick a number"
            assert choices == ["1", "2", "3"]
            return "2"

        result = json.loads(clarify_tool(
            "Pick a number",
            choices=["1", "2", "3"],
            callback=mock_callback
        ))
        assert result["question"] == "Pick a number"
        assert result["choices_offered"] == ["1", "2", "3"]
        assert result["user_response"] == "2"

    def test_empty_question_returns_error(self):
        """Should return error for empty question."""
        result = json.loads(clarify_tool("", callback=lambda q, c: "ignored"))
        assert "error" in result
        assert "required" in result["error"].lower()

    def test_whitespace_only_question_returns_error(self):
        """Should return error for whitespace-only question."""
        result = json.loads(clarify_tool("   \n\t  ", callback=lambda q, c: "ignored"))
        assert "error" in result

    def test_no_callback_returns_error(self):
        """Should return error when no callback is provided."""
        result = json.loads(clarify_tool("What do you want?"))
        assert "error" in result
        assert "not available" in result["error"].lower()


class TestClarifyToolChoicesValidation:
    """Tests for choices parameter validation."""

    def test_choices_trimmed_to_max(self):
        """Should trim choices to MAX_CHOICES."""
        choices_passed = []

        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            choices_passed.extend(choices or [])
            return "picked"

        many_choices = ["a", "b", "c", "d", "e", "f", "g"]
        clarify_tool("Pick one", choices=many_choices, callback=mock_callback)

        assert len(choices_passed) == MAX_CHOICES

    def test_empty_choices_become_none(self):
        """Empty choices list should become None (open-ended)."""
        choices_received = ["marker"]

        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            choices_received.clear()
            if choices is not None:
                choices_received.extend(choices)
            return "answer"

        clarify_tool("Open question?", choices=[], callback=mock_callback)
        assert choices_received == []  # Was cleared, nothing added

    def test_choices_with_only_whitespace_stripped(self):
        """Whitespace-only choices should be stripped out."""
        choices_received = []

        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            choices_received.extend(choices or [])
            return "answer"

        clarify_tool("Pick", choices=["valid", "  ", "", "also valid"], callback=mock_callback)
        assert choices_received == ["valid", "also valid"]

    def test_invalid_choices_type_returns_error(self):
        """Non-list choices should return error."""
        result = json.loads(clarify_tool(
            "Question?",
            choices="not a list",  # type: ignore
            callback=lambda q, c: "ignored"
        ))
        assert "error" in result
        assert "list" in result["error"].lower()

    def test_choices_converted_to_strings(self):
        """Non-string choices should be converted to strings."""
        choices_received = []

        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            choices_received.extend(choices or [])
            return "answer"

        clarify_tool("Pick", choices=[1, 2, 3], callback=mock_callback)  # type: ignore
        assert choices_received == ["1", "2", "3"]


class TestClarifyToolCallbackHandling:
    """Tests for callback error handling."""

    def test_callback_exception_returns_error(self):
        """Should return error if callback raises exception."""
        def failing_callback(question: str, choices: Optional[List[str]]) -> str:
            raise RuntimeError("User cancelled")

        result = json.loads(clarify_tool("Question?", callback=failing_callback))
        assert "error" in result
        assert "Failed to get user input" in result["error"]
        assert "User cancelled" in result["error"]

    def test_callback_receives_stripped_question(self):
        """Callback should receive trimmed question."""
        received_question = []

        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            received_question.append(question)
            return "answer"

        clarify_tool("  Question with spaces  \n", callback=mock_callback)
        assert received_question[0] == "Question with spaces"

    def test_user_response_stripped(self):
        """User response should be stripped of whitespace."""
        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            return "  response with spaces  \n"

        result = json.loads(clarify_tool("Q?", callback=mock_callback))
        assert result["user_response"] == "response with spaces"


class TestCheckClarifyRequirements:
    """Tests for the requirements check function."""

    def test_always_returns_true(self):
        """clarify tool has no external requirements."""
        assert check_clarify_requirements() is True


class TestClarifySchema:
    """Tests for the OpenAI function-calling schema."""

    def test_schema_name(self):
        """Schema should have correct name."""
        assert CLARIFY_SCHEMA["name"] == "clarify"

    def test_schema_has_description(self):
        """Schema should have a description."""
        assert "description" in CLARIFY_SCHEMA
        assert len(CLARIFY_SCHEMA["description"]) > 50

    def test_schema_question_required(self):
        """Question parameter should be required."""
        assert "question" in CLARIFY_SCHEMA["parameters"]["required"]

    def test_schema_choices_optional(self):
        """Choices parameter should be optional."""
        assert "choices" not in CLARIFY_SCHEMA["parameters"]["required"]

    def test_schema_choices_max_items(self):
        """Schema should specify max items for choices."""
        choices_spec = CLARIFY_SCHEMA["parameters"]["properties"]["choices"]
        assert choices_spec.get("maxItems") == MAX_CHOICES

    def test_max_choices_is_four(self):
        """MAX_CHOICES constant should be 4."""
        assert MAX_CHOICES == 4

    def test_schema_multi_select_optional(self):
        """multi_select should not be in required list."""
        assert "multi_select" not in CLARIFY_SCHEMA["parameters"]["required"]

    def test_schema_multi_select_is_boolean(self):
        """multi_select should be a boolean parameter."""
        ms_spec = CLARIFY_SCHEMA["parameters"]["properties"].get("multi_select")
        assert ms_spec is not None
        assert ms_spec["type"] == "boolean"

    def test_schema_multi_select_default_false(self):
        """multi_select should default to false (not in required)."""
        # The model should treat it as false when omitted
        assert "multi_select" not in CLARIFY_SCHEMA["parameters"]["required"]


class TestClarifyToolMultiSelect:
    """Tests for multi_select (checkbox) support added to clarify_tool."""

    def test_multi_select_false_keeps_existing_behavior(self):
        """When multi_select=False, user_response should be a single string."""
        def mock_callback(question, choices):
            return "blue"

        result = json.loads(clarify_tool(
            "What color?",
            choices=["red", "blue", "green"],
            multi_select=False,
            callback=mock_callback,
        ))
        assert result["user_response"] == "blue"
        assert isinstance(result["user_response"], str)

    def test_multi_select_true_returns_list(self):
        """When multi_select=True, user_response should be a list of strings."""
        def mock_callback(question, choices):
            return "red, blue"

        result = json.loads(clarify_tool(
            "Which colors?",
            choices=["red", "blue", "green"],
            multi_select=True,
            callback=mock_callback,
        ))
        assert result["user_response"] == ["red", "blue"]
        assert isinstance(result["user_response"], list)

    def test_multi_select_single_choice_still_list(self):
        """Even a single selection should be a list when multi_select=True."""
        def mock_callback(question, choices):
            return "red"

        result = json.loads(clarify_tool(
            "Which color?",
            choices=["red", "blue"],
            multi_select=True,
            callback=mock_callback,
        ))
        assert result["user_response"] == ["red"]
        assert isinstance(result["user_response"], list)

    def test_multi_select_with_json_array_response(self):
        """Callback can return a JSON array string for multi-select."""
        def mock_callback(question, choices):
            return '["red", "blue"]'

        result = json.loads(clarify_tool(
            "Which colors?",
            choices=["red", "blue", "green"],
            multi_select=True,
            callback=mock_callback,
        ))
        assert result["user_response"] == ["red", "blue"]

    def test_multi_select_no_choices_falls_back_to_single_string(self):
        """When choices is None, multi_select has no effect on response type."""
        def mock_callback(question, choices):
            return "free form answer"

        result = json.loads(clarify_tool(
            "What do you think?",
            multi_select=True,
            callback=mock_callback,
        ))
        # Without choices, falls back to single string response
        assert result["user_response"] == "free form answer"
        assert isinstance(result["user_response"], str)

    def test_multi_select_default_is_false(self):
        """Default multi_select should be False (backward compatible)."""
        def mock_callback(question, choices):
            return "picked"

        result = json.loads(clarify_tool(
            "Pick one",
            choices=["a", "b"],
            callback=mock_callback,
        ))
        assert result["user_response"] == "picked"
        assert isinstance(result["user_response"], str)

    def test_multi_select_callback_receives_flag(self):
        """Callback should receive multi_select keyword argument when supported."""
        received_flag = []

        def mock_callback(question, choices, **kwargs):
            received_flag.append(kwargs.get("multi_select"))
            return "a, b"

        clarify_tool(
            "Pick",
            choices=["a", "b", "c"],
            multi_select=True,
            callback=mock_callback,
        )
        assert received_flag == [True]

    def test_multi_select_backward_compatible_callback(self):
        """Callback that does not accept multi_select keyword should still work."""
        def mock_callback(question, choices):
            return "a, b"

        result = json.loads(clarify_tool(
            "Pick",
            choices=["a", "b", "c"],
            multi_select=True,
            callback=mock_callback,
        ))
        assert result["user_response"] == ["a", "b"]

    def test_multi_select_empty_selection_returns_empty_list(self):
        """Empty response should produce empty list when multi_select=True."""
        def mock_callback(question, choices):
            return ""

        result = json.loads(clarify_tool(
            "Which?",
            choices=["a", "b"],
            multi_select=True,
            callback=mock_callback,
        ))
        assert result["user_response"] == []

    def test_multi_select_whitespace_choices_stripped(self):
        """Individual selections should be stripped of whitespace."""
        def mock_callback(question, choices):
            return "  a , b ,  c  "

        result = json.loads(clarify_tool(
            "Which?",
            choices=["a", "b", "c"],
            multi_select=True,
            callback=mock_callback,
        ))
        assert result["user_response"] == ["a", "b", "c"]

    def test_multi_select_choices_offered_preserved(self):
        """choices_offered should match what was passed in, not the response."""
        def mock_callback(question, choices):
            return "red, blue"

        result = json.loads(clarify_tool(
            "Which?",
            choices=["red", "blue", "green"],
            multi_select=True,
            callback=mock_callback,
        ))
        assert result["choices_offered"] == ["red", "blue", "green"]

    def test_multi_select_max_choices_enforced(self):
        """MAX_CHOICES enforcement should still work with multi_select."""
        choices_passed = []

        def mock_callback(question, choices):
            choices_passed.extend(choices or [])
            return "a, b, c, d"

        many_choices = ["a", "b", "c", "d", "e", "f"]
        clarify_tool(
            "Pick some",
            choices=many_choices,
            multi_select=True,
            callback=mock_callback,
        )
        assert len(choices_passed) == MAX_CHOICES
