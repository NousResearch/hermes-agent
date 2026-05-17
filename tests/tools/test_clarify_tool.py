"""Tests for tools/clarify_tool.py - Interactive clarifying questions."""

import json
from typing import List, Optional

import pytest

from tools.clarify_tool import (
    clarify_tool,
    check_clarify_requirements,
    MAX_CHOICES,
    CLARIFY_SCHEMA,
    _has_context,
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

    def test_schema_description_mandates_context(self):
        """Schema must tell the agent context is mandatory and show the chain shape."""
        desc = CLARIFY_SCHEMA["description"]
        # Spell out the mandatory rule
        assert "MANDATORY" in desc or "mandatory" in desc
        # Spell out the shape so the agent can reproduce it on first try
        assert "Theme:" in desc
        assert "Epic:" in desc
        assert "Task:" in desc
        assert "→" in desc
        # Mention the skill so the agent can load it for the full pattern
        assert "asking-the-user-well" in desc

    def test_question_param_description_mentions_context(self):
        """The `question` parameter description must remind about the context block."""
        q_desc = CLARIFY_SCHEMA["parameters"]["properties"]["question"]["description"]
        assert "context" in q_desc.lower()


# =============================================================================
# Context enforcement (kanban t_f2e0e531)
# =============================================================================


class TestContextDetection:
    """Tests for the `_has_context` heuristic."""

    def test_short_bare_question_lacks_context(self):
        ok, reason = _has_context("What color?", None)
        assert ok is False
        assert "char" in reason or "line" in reason

    def test_chain_arrow_alone_in_short_question_lacks_context(self):
        # Has the arrow but is too short to be a real canvas.
        ok, _ = _has_context("A → B?", None)
        assert ok is False

    def test_long_question_with_chain_arrow_has_context(self):
        q = (
            "Theme: Timeline & Timesheets → Epic: dashboard surface → "
            "Task: pick canonical Postgres stack → "
            "Question: which of the three timeline DBs is canonical? "
            "Stack A (timeline-new) has 1443 tasks; Stack C (work) has 356 "
            "records and 1793h of Xsolla signals."
        )
        ok, reason = _has_context(q, None)
        assert ok is True, reason

    def test_long_question_with_table_has_context(self):
        q = (
            "Decide which loader to ship first. Here are the three candidates:\n\n"
            "| name | status | rows |\n"
            "| --- | --- | --- |\n"
            "| a | ready | 142  |\n"
            "| b | wip   | 89   |"
        )
        ok, reason = _has_context(q, None)
        assert ok is True, reason

    def test_long_question_with_code_fence_has_context(self):
        q = (
            "I need to know which constant to pick. The current code reads:\n\n"
            "```python\nMAX_CHOICES = 4\nTIMEOUT = 600\n```\n\n"
            "Should we lift MAX_CHOICES to 6 to fit the new survey shape?"
        )
        ok, reason = _has_context(q, None)
        assert ok is True, reason

    def test_multiline_canvas_without_explicit_marker_has_context(self):
        # Four lines of substance counts as an embedded canvas.
        q = (
            "Looking at the three options below:\n"
            "first option summary line\n"
            "second option summary line\n"
            "third option summary line\n"
            "which one should ship?"
        )
        ok, reason = _has_context(q, None)
        assert ok is True, reason

    def test_long_question_without_any_marker_lacks_context(self):
        # Wall of prose with no chain, no structure, no canvas.
        q = (
            "I have been thinking a lot about this and would like to ask you "
            "which approach you would prefer because there are several ways "
            "we could potentially go forward and I want to make sure we pick "
            "the one that is best for the project overall and matches your "
            "intuition about the right shape of the eventual deliverable."
        )
        ok, reason = _has_context(q, None)
        assert ok is False
        assert "marker" in reason


class TestContextEnforcement:
    """Tests for the agent-path enforcement (require_context=True)."""

    def test_default_does_not_enforce(self):
        """Direct library callers (require_context=False default) are unchanged."""
        def cb(q, c):
            return "blue"
        result = json.loads(clarify_tool("What color?", callback=cb))
        # No error — the short bare call still works for direct callers.
        assert "error" not in result
        assert result["user_response"] == "blue"

    def test_require_context_rejects_bare_short_call(self):
        """With require_context=True a bare short question is rejected."""
        def cb(q, c):
            return "should not be called"
        result = json.loads(clarify_tool(
            "What color?",
            callback=cb,
            require_context=True,
        ))
        assert "error" in result
        # The error message tells the agent exactly what to add
        err = result["error"]
        assert "context" in err.lower()
        assert "Theme:" in err
        assert "Epic:" in err
        assert "Task:" in err
        assert "→" in err
        # And names the skill so the agent can load it
        assert "asking-the-user-well" in err

    def test_require_context_accepts_question_with_chain(self):
        """A well-formed chain question with require_context=True succeeds."""
        captured = {}

        def cb(q, c):
            captured["q"] = q
            return "A"

        good_question = (
            "Theme: Hermes infra → Epic: clarify-with-context → "
            "Task: ship enforcement → Question: should we enforce now or "
            "wait for canvas field? Current state: schema has no `context=` "
            "field yet (t_bcacbcb9 blocked); enforcement on `question` text "
            "is the only available gate. Risk: false-reject rate ~unknown."
        )
        result = json.loads(clarify_tool(
            good_question,
            choices=["enforce now", "wait for t_bcacbcb9"],
            callback=cb,
            require_context=True,
        ))
        assert "error" not in result, result
        assert result["user_response"] == "A"
        assert captured["q"] == good_question

    def test_require_context_accepts_canvas_with_table(self):
        """A multi-line canvas with a markdown table is accepted."""
        captured = {}

        def cb(q, c):
            captured["q"] = q
            return "row 2"

        canvas_question = (
            "Decide which loader to schedule. Candidates:\n\n"
            "| name | status |\n"
            "| --- | --- |\n"
            "| smb-share | ready |\n"
            "| sharepoint | wip |\n\n"
            "Which one ships first?"
        )
        result = json.loads(clarify_tool(
            canvas_question,
            choices=["smb-share", "sharepoint"],
            callback=cb,
            require_context=True,
        ))
        assert "error" not in result, result
        assert result["user_response"] == "row 2"

    def test_rejection_does_not_invoke_callback(self):
        """A rejected call must not block the user with a prompt."""
        called = {"n": 0}

        def cb(q, c):
            called["n"] += 1
            return "should not happen"

        clarify_tool("?", callback=cb, require_context=True)
        assert called["n"] == 0

    def test_registry_handler_enforces_context(self):
        """The handler registered with the tool registry passes require_context=True."""
        # Re-import the registry entry the agent loop sees.
        from tools.registry import registry
        entry = registry.get_entry("clarify")
        assert entry is not None, "clarify tool not registered"

        # Call the handler with a bare short question — should reject.
        result = json.loads(entry.handler(
            {"question": "Pick one"},
            callback=lambda q, c: "should not run",
        ))
        assert "error" in result
        assert "context" in result["error"].lower()
