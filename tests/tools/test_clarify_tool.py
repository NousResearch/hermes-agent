"""Tests for tools/clarify_tool.py - Interactive clarifying questions."""

import json
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

from agent.gemini_schema import sanitize_gemini_tool_parameters

from tools.clarify_tool import (
    clarify_tool,
    check_clarify_requirements,
    MAX_CHOICES,
    CLARIFY_SCHEMA,
    CLARIFY_CUSTOM_RESPONSE_PREFIX,
    CLARIFY_OPTION_RESPONSE_PREFIX,
    _flatten_choice,
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


class TestClarifyDictChoices:
    """Dict-shaped choices must be unwrapped to user-facing text at the source.

    LLMs sometimes emit [{"description": "..."}] instead of bare strings. The
    naive str(c) coercion leaked the Python dict repr onto every surface (CLI
    panel, Discord buttons, Telegram list) AND returned it verbatim as the
    user's answer. _flatten_choice normalises at the one platform-agnostic
    entry point so the whole class is fixed in one place.
    """

    def test_flatten_unwraps_label_first(self):
        assert _flatten_choice({"label": "Short", "description": "Long"}) == "Short"

    def test_flatten_unwraps_description_when_no_label(self):
        assert _flatten_choice({"description": "A loose layout"}) == "A loose layout"

    def test_flatten_unwrap_order_label_over_description(self):
        assert _flatten_choice({"description": "verbose", "label": "tight"}) == "tight"

    def test_flatten_drops_name_value_only_dict(self):
        # name/value are component-shaped fields, not user-facing labels —
        # picking them would leak raw enum values / short model ids.
        assert _flatten_choice({"name": "tight", "value": "x"}) == ""

    def test_flatten_prefers_canonical_key_over_name(self):
        assert _flatten_choice({"name": "tight", "description": "Tight desc"}) == "Tight desc"

    def test_flatten_drops_keyless_dict(self):
        assert _flatten_choice({"foo": "bar", "n": 1}) == ""

    def test_flatten_passthrough_string_and_scalar(self):
        assert _flatten_choice("plain") == "plain"
        assert _flatten_choice(7) == "7"
        assert _flatten_choice(None) == ""

    def test_dict_choices_reach_callback_as_clean_text(self):
        """The whole point: the UI callback never sees a dict repr."""
        seen = []

        def cb(question, choices):
            seen.extend(choices or [])
            return choices[0]

        result = json.loads(clarify_tool(
            "Pick a layout",
            choices=[
                {"choice": "Tight", "description": "Tight, covers all 3 points"},
                {"description": "Loose layout"},
                {"name": "modelid", "value": "abc"},  # dropped, not leaked
                "A plain string choice",
            ],
            callback=cb,
        ))  # type: ignore
        assert seen == [
            "Tight, covers all 3 points",
            "Loose layout",
            "A plain string choice",
        ]
        # and the resolved answer is clean text, not a dict repr
        assert result["user_response"] == "Tight, covers all 3 points"
        assert "{" not in result["user_response"]
        assert all("{" not in c for c in result["choices_offered"])

    def test_structured_choices_preserve_description_and_canonical_selection(self):
        """One structured source must drive readable UI text and the selected value."""
        seen = {}

        def cb(question, choices):
            seen["question"] = question
            seen["choices"] = choices
            # Desktop responds with the stable option id instead of display text.
            return "safe"

        result = json.loads(clarify_tool(
            "Which rollout should we use?",
            context="Production currently has no canary traffic.",
            recommendation="Choose Safe because it preserves rollback capacity.",
            choices=[
                {
                    "id": "fast",
                    "label": "Fast",
                    "description": "Deploy to every instance immediately.",
                    "value": "fast-rollout",
                },
                {
                    "id": "safe",
                    "label": "Safe",
                    "description": "Canary first, then expand after verification.",
                    "value": "safe-rollout",
                },
            ],
            callback=cb,
        ))

        assert seen["question"] == (
            "Which rollout should we use?\n\n"
            "Context: Production currently has no canary traffic.\n\n"
            "Recommendation: Choose Safe because it preserves rollback capacity."
        )
        assert seen["choices"] == [
            "Fast — Deploy to every instance immediately.",
            "Safe — Canary first, then expand after verification.",
        ]
        assert result["question"] == "Which rollout should we use?"
        assert result["context"] == "Production currently has no canary traffic."
        assert result["recommendation"] == "Choose Safe because it preserves rollback capacity."
        assert result["options"][1] == {
            "id": "safe",
            "index": 2,
            "label": "Safe",
            "description": "Canary first, then expand after verification.",
            "value": "safe-rollout",
        }
        assert result["selected_option"] == result["options"][1]
        assert result["user_response"] == "safe-rollout"

    def test_structured_choice_transport_token_resolves_exact_id(self):
        """A clicked option id wins even when its value looks like another index."""
        result = json.loads(clarify_tool(
            "Which option?",
            choices=[
                {
                    "id": "first",
                    "label": "Looks numeric",
                    "description": "First option.",
                    "value": "2",
                },
                {
                    "id": "second",
                    "label": "Second",
                    "description": "Second option.",
                    "value": "other",
                },
            ],
            callback=lambda _question, _choices: (
                f"{CLARIFY_OPTION_RESPONSE_PREFIX}first"
            ),
        ))

        assert result["selected_option"]["id"] == "first"
        assert result["selected_option"]["index"] == 1
        assert result["user_response"] == "2"

    def test_unknown_structured_choice_id_returns_error_without_leaking_token(self):
        result = json.loads(clarify_tool(
            "Which option?",
            choices=[{"id": "known", "label": "Known", "description": "Valid."}],
            callback=lambda _question, _choices: (
                f"{CLARIFY_OPTION_RESPONSE_PREFIX}missing"
            ),
        ))

        assert result["error"] == "Selected clarify option is no longer available."
        assert result["selected_option"] is None
        assert result["user_response"] == ""
        assert CLARIFY_OPTION_RESPONSE_PREFIX not in json.dumps(result)

    def test_custom_response_does_not_alias_to_an_existing_option(self):
        result = json.loads(clarify_tool(
            "Which option?",
            choices=[
                {"id": "first", "label": "Safe", "description": "First.", "value": "safe"},
                {"id": "second", "label": "Other option", "description": "Second.", "value": "other"},
            ],
            callback=lambda _question, _choices: (
                f"{CLARIFY_CUSTOM_RESPONSE_PREFIX}Safe"
            ),
        ))

        assert result["selected_option"] is None
        assert result["user_response"] == "Safe"
        assert result["response_kind"] == "custom"
        assert CLARIFY_CUSTOM_RESPONSE_PREFIX not in json.dumps(result)

    def test_agent_runtime_dispatch_forwards_context_and_recommendation(self):
        from agent.agent_runtime_helpers import invoke_tool

        seen = {}

        def callback(question, choices):
            seen["question"] = question
            seen["choices"] = choices
            return "1"

        agent = SimpleNamespace(
            _memory_manager=None,
            clarify_callback=callback,
            session_id="",
            valid_tool_names=set(),
            enabled_toolsets=None,
            disabled_toolsets=None,
        )

        def run_middleware(_name, payload, execute, **_kwargs):
            return execute(payload)

        with patch(
            "hermes_cli.middleware.run_tool_execution_middleware",
            side_effect=run_middleware,
        ):
            result = json.loads(invoke_tool(
                agent,
                "clarify",
                {
                    "question": "Which rollout?",
                    "context": "No canary traffic.",
                    "recommendation": "Choose Safe.",
                    "choices": ["Fast", "Safe"],
                },
                "",
                pre_tool_block_checked=True,
                skip_tool_request_middleware=True,
            ))

        assert seen["question"] == (
            "Which rollout?\n\nContext: No canary traffic.\n\n"
            "Recommendation: Choose Safe."
        )
        assert seen["choices"] == ["Fast", "Safe"]
        assert result["user_response"] == "Fast"


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

    def test_schema_accepts_legacy_strings_and_structured_choices(self):
        choices_spec = CLARIFY_SCHEMA["parameters"]["properties"]["choices"]
        variants = choices_spec["items"]["anyOf"]

        assert {variant.get("type") for variant in variants} == {"string", "object"}
        object_variant = next(variant for variant in variants if variant.get("type") == "object")
        assert object_variant["required"] == ["label", "description"]
        assert "context" in CLARIFY_SCHEMA["parameters"]["properties"]
        assert "recommendation" in CLARIFY_SCHEMA["parameters"]["properties"]

    def test_structured_choice_union_survives_gemini_sanitizing(self):
        sanitized = sanitize_gemini_tool_parameters(CLARIFY_SCHEMA["parameters"])
        variants = sanitized["properties"]["choices"]["items"]["anyOf"]

        assert {variant.get("type") for variant in variants} == {"string", "object"}

    def test_max_choices_is_four(self):
        """MAX_CHOICES constant should be 4."""
        assert MAX_CHOICES == 4
