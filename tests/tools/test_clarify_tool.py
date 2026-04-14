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


class TestClarifyToolQuestionsMode:
    """Tests for the advanced multi-question mode."""

    def test_questions_mode_routes_to_callback(self):
        received = {}
        def mock_callback(question, choices=None, questions=None):
            received["questions"] = questions
            return json.dumps({"deploy_target": {"selected": ["staging"], "freeform": None}})

        questions = [
            {
                "header": "deploy_target",
                "question": "Deploy to which env?",
                "options": [
                    {"label": "staging", "description": "Test env", "recommended": True},
                    {"label": "production", "description": "Prod env"},
                ],
            }
        ]
        result = json.loads(clarify_tool(questions=questions, callback=mock_callback))
        assert "responses" in result
        assert received["questions"] == questions

    def test_questions_mode_ignores_question_and_choices(self):
        received = {}
        def mock_callback(question, choices=None, questions=None):
            received["questions"] = questions
            received["question"] = question
            return json.dumps({"q1": {"selected": [], "freeform": "hi"}})

        questions = [{"header": "q1", "question": "What?", "allowFreeformInput": True}]
        clarify_tool(
            question="ignored question",
            choices=["ignored"],
            questions=questions,
            callback=mock_callback,
        )
        assert received["questions"] is not None
        assert received["question"] is None

    def test_questions_empty_list_returns_error(self):
        result = json.loads(clarify_tool(questions=[], callback=lambda *a, **kw: ""))
        assert "error" in result

    def test_questions_missing_header_returns_error(self):
        result = json.loads(clarify_tool(
            questions=[{"question": "No header"}],
            callback=lambda *a, **kw: "",
        ))
        assert "error" in result

    def test_questions_missing_question_text_returns_error(self):
        result = json.loads(clarify_tool(
            questions=[{"header": "h1"}],
            callback=lambda *a, **kw: "",
        ))
        assert "error" in result

    def test_questions_duplicate_headers_returns_error(self):
        result = json.loads(clarify_tool(
            questions=[
                {"header": "dup", "question": "Q1"},
                {"header": "dup", "question": "Q2"},
            ],
            callback=lambda *a, **kw: "",
        ))
        assert "error" in result

    def test_questions_header_max_length_enforced(self):
        result = json.loads(clarify_tool(
            questions=[{"header": "x" * 51, "question": "Q?"}],
            callback=lambda *a, **kw: "",
        ))
        assert "error" in result


class TestClarifySchemaQuestionsMode:
    def test_schema_has_questions_property(self):
        assert "questions" in CLARIFY_SCHEMA["parameters"]["properties"]

    def test_schema_questions_is_array(self):
        q = CLARIFY_SCHEMA["parameters"]["properties"]["questions"]
        assert q["type"] == "array"

    def test_schema_questions_items_have_header_and_question(self):
        items = CLARIFY_SCHEMA["parameters"]["properties"]["questions"]["items"]
        assert "header" in items["properties"]
        assert "question" in items["properties"]
        assert set(items["required"]) == {"header", "question"}

    def test_schema_questions_not_required(self):
        assert "questions" not in CLARIFY_SCHEMA["parameters"]["required"]
