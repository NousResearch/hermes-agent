"""Tests for tools/clarify_tool.py - Interactive clarifying questions."""

import json
from typing import List, Optional


from tools.clarify_tool import (
    clarify_tool,
    check_clarify_requirements,
    MAX_CHOICES,
    MAX_QUESTIONS,
    CLARIFY_SCHEMA,
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


    def test_choices_converted_to_strings(self):
        """Non-string choices should be converted to strings."""
        choices_received = []

        def mock_callback(question: str, choices: Optional[List[str]]) -> str:
            choices_received.extend(choices or [])
            return "answer"

        clarify_tool("Pick", choices=[1, 2, 3], callback=mock_callback)  # type: ignore
        assert choices_received == ["1 (Recommended)", "2", "3"]


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


    def test_flatten_accepts_value_only_dict(self):
        assert _flatten_choice({"value": "Use OAuth"}) == "Use OAuth"

    def test_flatten_unwraps_serialized_json_value(self):
        assert _flatten_choice('{"value": "Use OAuth"}') == "Use OAuth"

    def test_flatten_unwraps_serialized_json_description(self):
        assert _flatten_choice('{"description": "Use API key"}') == "Use API key"

    def test_flatten_keeps_malformed_json_braces_as_plain_text(self):
        assert _flatten_choice('{not valid json}') == "{not valid json}"

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
            "Tight, covers all 3 points (Recommended)",
            "Loose layout",
            "A plain string choice",
        ]
        # and the resolved answer is clean text, not a dict repr
        assert result["user_response"] == "Tight, covers all 3 points"
        assert "{" not in result["user_response"]
        assert all("{" not in c for c in result["choices_offered"])


class TestClarifySchema:
    """Tests for the OpenAI function-calling schema."""

    def test_schema_name(self):
        """Schema should have correct name."""
        assert CLARIFY_SCHEMA["name"] == "clarify"


    def test_max_choices_is_four(self):
        """MAX_CHOICES constant should be 4."""
        assert MAX_CHOICES == 4


    def test_schema_multi_select_default_false(self):
        """multi_select should default to false (not in required)."""
        # The model should treat it as false when omitted
        assert "multi_select" not in CLARIFY_SCHEMA["parameters"]["required"]


    def test_schema_description_advertises_batching(self):
        """The top-level description must tell the model it can batch.

        The `questions` parameter description alone is not enough — the
        model decides HOW to call from the tool description, so the batch
        capability has to be surfaced there or it keeps asking one
        question per call.
        """
        description = CLARIFY_SCHEMA["description"]
        assert "questions" in description
        assert "one call" in description.lower()


    def test_schema_questions_param_is_required_and_capped(self):
        """`questions` is the single documented way to call (a single question
        is a one-entry array) and carries the batch cap so the model sees the
        limit. The legacy top-level `question` shape stays handler-accepted
        but unadvertised."""
        params = CLARIFY_SCHEMA["parameters"]
        assert params["required"] == ["questions"]
        assert params["properties"]["questions"]["maxItems"] == MAX_QUESTIONS
        assert params["properties"]["questions"].get("minItems") == 1
        # Legacy shape must remain accepted by the handler even though the
        # schema no longer advertises it.
        assert "question" not in params["properties"]


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


class TestClarifyRecommendedLabel:
    """The first choice is the agent's pick and is labelled as such.

    The schema tells the model to order choices best-first, so the tool tags
    element 0 with "(Recommended)" at the one platform-agnostic entry point —
    CLI, TUI, desktop, and messaging adapters all inherit the same label. The
    label is presentation only: it never appears in the answer the agent reads.
    """

    def test_first_choice_is_labelled(self):
        seen = []

        def cb(question, choices):
            seen.extend(choices or [])
            return choices[1]

        clarify_tool("Pick", choices=["Rebase", "Merge"], callback=cb)
        assert seen == ["Rebase (Recommended)", "Merge"]

    def test_answer_strips_the_label(self):
        """Picking the recommended option returns the bare option text."""
        def cb(question, choices):
            return choices[0]

        result = json.loads(clarify_tool("Pick", choices=["Rebase", "Merge"], callback=cb))
        assert result["user_response"] == "Rebase"
        assert result["choices_offered"] == ["Rebase", "Merge"]

    def test_multi_select_answers_strip_the_label(self):
        def cb(question, choices, multi_select=False):
            return ", ".join(choices[:2])

        result = json.loads(clarify_tool(
            "Pick some",
            choices=["Rebase", "Merge", "Squash"],
            multi_select=True,
            callback=cb,
        ))
        assert result["user_response"] == ["Rebase", "Merge"]

    def test_single_choice_is_not_labelled(self):
        """One option isn't a recommendation — there's nothing to prefer it over."""
        seen = []

        def cb(question, choices):
            seen.extend(choices or [])
            return choices[0]

        clarify_tool("Confirm", choices=["Ship it"], callback=cb)
        assert seen == ["Ship it"]

    def test_label_is_not_doubled(self):
        """A model that wrote its own label doesn't get a second one."""
        seen = []

        def cb(question, choices):
            seen.extend(choices or [])
            return choices[0]

        clarify_tool("Pick", choices=["Rebase (recommended)", "Merge"], callback=cb)
        assert seen == ["Rebase (recommended)", "Merge"]

    def test_open_ended_unaffected(self):
        def cb(question, choices):
            assert choices is None
            return "whatever"

        result = json.loads(clarify_tool("Thoughts?", callback=cb))
        assert result["choices_offered"] is None
        assert result["user_response"] == "whatever"


class TestInvokeCallbackDispatch:
    """_invoke_callback uses signature inspection, never a TypeError retry."""

    def test_internal_typeerror_not_swallowed_or_retried(self):
        """A compatible callback that raises TypeError internally must be
        invoked exactly once and its error surfaced — not retried with the
        legacy 2-arg form (which would prompt the user twice)."""
        from tools.clarify_tool import _invoke_callback
        calls = []

        def bad_callback(question, choices, multi_select=False):
            calls.append(1)
            raise TypeError("internal bug")

        import pytest
        with pytest.raises(TypeError, match="internal bug"):
            _invoke_callback(bad_callback, "Q?", ["a"], True)
        assert len(calls) == 1


    def test_var_keyword_callback_receives_flag(self):
        from tools.clarify_tool import _invoke_callback
        seen = {}

        def kw_cb(question, choices, **kwargs):
            seen.update(kwargs)
            return "ok"

        _invoke_callback(kw_cb, "Q?", ["a"], True)
        assert seen.get("multi_select") is True


class TestRegistryMultiSelectPassThrough:
    """The registered tool handler must forward multi_select from tool args."""

    def test_handler_passes_multi_select(self):
        from tools.registry import registry
        entry = registry.get_entry("clarify")
        seen = {}

        def cb(question, choices, multi_select=False):
            seen["multi"] = multi_select
            return "a, b"

        result = json.loads(entry.handler(
            {"question": "Pick", "choices": ["a", "b"], "multi_select": True},
            callback=cb,
        ))
        assert seen["multi"] is True
        assert result["user_response"] == ["a", "b"]

    def test_handler_default_single_select(self):
        from tools.registry import registry
        entry = registry.get_entry("clarify")
        seen = {}

        def cb(question, choices, multi_select=False):
            seen["multi"] = multi_select
            return "a"

        result = json.loads(entry.handler(
            {"question": "Pick", "choices": ["a", "b"]},
            callback=cb,
        ))
        assert seen["multi"] is False
        assert result["user_response"] == "a"


class TestClarifyBatchValidation:
    """Validation of the `questions` batch parameter (issue #18450)."""

    def test_batch_takes_precedence_over_question(self):
        """When both are present, `questions` wins and `question` is ignored."""
        seen = {}

        def cb(question, choices, multi_select=False, questions=None):
            seen["questions"] = questions
            return {"answers": {"q0": "blue"}}

        result = json.loads(clarify_tool(
            "ignored single question",
            questions=[{"question": "What color?"}],
            callback=cb,
        ))
        assert "responses" in result
        assert len(result["responses"]) == 1
        assert result["responses"][0]["question"] == "What color?"
        assert seen["questions"][0]["question"] == "What color?"

    def test_batch_rejects_more_than_five(self):
        result = json.loads(clarify_tool(
            "",
            questions=[{"question": f"Q{i}?"} for i in range(6)],
            callback=lambda *a, **k: "",
        ))
        assert "error" in result

    def test_batch_rejects_blank_question_text(self):
        result = json.loads(clarify_tool(
            "",
            questions=[{"question": "Real?"}, {"question": "   "}],
            callback=lambda *a, **k: "",
        ))
        assert "error" in result

    def test_batch_rejects_non_list(self):
        result = json.loads(clarify_tool(
            "", questions={"question": "Q?"}, callback=lambda *a, **k: "",
        ))
        assert "error" in result

    def test_batch_empty_list_falls_back_to_single_question(self):
        """An empty questions array degrades to the single-question path."""
        def cb(question, choices):
            assert question == "Single?"
            return "yes"

        result = json.loads(clarify_tool("Single?", questions=[], callback=cb))
        assert result["user_response"] == "yes"
        assert "responses" not in result

    def test_batch_choices_flattened_capped_and_labelled_per_question(self):
        """Each question gets the full choice pipeline: flatten, cap, label."""
        seen = {}

        def cb(question, choices, multi_select=False, questions=None):
            seen["questions"] = questions
            return {"answers": {"q0": "a", "q1": "Loose layout"}}

        clarify_tool(
            "",
            questions=[
                {"question": "Pick letter", "choices": ["a", "b", "c", "d", "e", "f"]},
                {"question": "Pick layout", "choices": [
                    {"description": "Loose layout"}, "Tight",
                ]},
            ],
            callback=cb,
        )
        q0, q1 = seen["questions"]
        assert len(q0["choices"]) == MAX_CHOICES
        assert q0["choices"][0] == "a (Recommended)"
        assert q1["choices"] == ["Loose layout (Recommended)", "Tight"]

    def test_batch_internal_ids_are_stable_and_model_id_echoed(self):
        """Wire ids are q0..qN. A model-supplied id only shows in results."""
        seen = {}

        def cb(question, choices, multi_select=False, questions=None):
            seen["questions"] = questions
            return {"answers": {"q0": "A", "q1": "B"}}

        result = json.loads(clarify_tool(
            "",
            questions=[
                {"id": "approach", "question": "Which approach?"},
                {"question": "Timeline?"},
            ],
            callback=cb,
        ))
        assert [q["qid"] for q in seen["questions"]] == ["q0", "q1"]
        assert result["responses"][0]["id"] == "approach"
        assert "id" not in result["responses"][1]

    def test_batch_multi_select_needs_choices(self):
        """multi_select is only honored when the question has choices."""
        seen = {}

        def cb(question, choices, multi_select=False, questions=None):
            seen["questions"] = questions
            return {"answers": {"q0": "free text"}}

        clarify_tool(
            "",
            questions=[{"question": "Thoughts?", "multi_select": True}],
            callback=cb,
        )
        assert seen["questions"][0]["multi_select"] is False


class TestClarifyBatchDispatch:
    """Batch-capable callbacks get the list once. Legacy callbacks loop."""

    def test_batch_callback_receives_list_once(self):
        calls = []

        def cb(question, choices, multi_select=False, questions=None):
            calls.append(questions)
            return {"answers": {"q0": "x", "q1": "y"}}

        result = json.loads(clarify_tool(
            "",
            questions=[{"question": "One?"}, {"question": "Two?"}],
            callback=cb,
        ))
        assert len(calls) == 1
        assert [r["user_response"] for r in result["responses"]] == ["x", "y"]

    def test_batch_callback_json_string_response(self):
        """A _block-style bridge returns the answers as a JSON string."""
        def cb(question, choices, multi_select=False, questions=None):
            return json.dumps({"answers": {"q0": "picked"}})

        result = json.loads(clarify_tool(
            "", questions=[{"question": "One?"}], callback=cb,
        ))
        assert result["responses"][0]["user_response"] == "picked"

    def test_batch_recommended_label_stripped_per_question(self):
        def cb(question, choices, multi_select=False, questions=None):
            return {"answers": {"q0": questions[0]["choices"][0]}}

        result = json.loads(clarify_tool(
            "",
            questions=[{"question": "Pick", "choices": ["Rebase", "Merge"]}],
            callback=cb,
        ))
        assert result["responses"][0]["user_response"] == "Rebase"
        assert result["responses"][0]["choices_offered"] == ["Rebase", "Merge"]

    def test_batch_multi_select_answer_parsed_to_list(self):
        def cb(question, choices, multi_select=False, questions=None):
            return {"answers": {"q0": '["red", "blue"]'}}

        result = json.loads(clarify_tool(
            "",
            questions=[{
                "question": "Colors?",
                "choices": ["red", "blue", "green"],
                "multi_select": True,
            }],
            callback=cb,
        ))
        assert result["responses"][0]["user_response"] == ["red", "blue"]

    def test_batch_timed_out_flag_passthrough_with_partials(self):
        """Timeout keeps the locked answers and sets the top-level flag."""
        def cb(question, choices, multi_select=False, questions=None):
            return {"answers": {"q0": "kept"}, "timed_out": True}

        result = json.loads(clarify_tool(
            "",
            questions=[{"question": "One?"}, {"question": "Two?"}],
            callback=cb,
        ))
        assert result["timed_out"] is True
        assert result["responses"][0]["user_response"] == "kept"
        assert result["responses"][1]["user_response"] == ""

    def test_batch_empty_response_is_skip_not_timeout(self):
        """A cancel-all resolves every answer empty with no timed_out flag."""
        def cb(question, choices, multi_select=False, questions=None):
            return ""

        result = json.loads(clarify_tool(
            "", questions=[{"question": "One?"}], callback=cb,
        ))
        assert result["responses"][0]["user_response"] == ""
        assert "timed_out" not in result

    def test_legacy_callback_gets_sequential_calls_in_order(self):
        """A callback without `questions` support is looped per question."""
        calls = []

        def legacy_cb(question, choices, multi_select=False):
            calls.append((question, tuple(choices or []) or None, multi_select))
            return f"answer to {question}"

        result = json.loads(clarify_tool(
            "",
            questions=[
                {"question": "One?", "choices": ["a", "b"]},
                {"question": "Two?"},
            ],
            callback=legacy_cb,
        ))
        assert [c[0] for c in calls] == ["One?", "Two?"]
        assert calls[0][1] == ("a (Recommended)", "b")
        assert calls[1][1] is None
        assert [r["user_response"] for r in result["responses"]] == [
            "answer to One?", "answer to Two?",
        ]
        assert "timed_out" not in result

    def test_legacy_loop_aborts_on_timeout_and_keeps_partials(self):
        """The loop stops on the first timeout. Collected answers survive."""
        from tools.clarify_tool import TIMEOUT_RESPONSE
        calls = []

        def legacy_cb(question, choices):
            calls.append(question)
            if len(calls) == 2:
                return TIMEOUT_RESPONSE
            return "answered"

        result = json.loads(clarify_tool(
            "",
            questions=[
                {"question": "One?"}, {"question": "Two?"}, {"question": "Three?"},
            ],
            callback=legacy_cb,
        ))
        assert calls == ["One?", "Two?"]
        assert result["timed_out"] is True
        assert [r["user_response"] for r in result["responses"]] == [
            "answered", "", "",
        ]

    def test_legacy_loop_skip_continues(self):
        """An explicit empty answer is a skip. The loop continues."""
        calls = []

        def legacy_cb(question, choices):
            calls.append(question)
            return "" if len(calls) == 1 else "second"

        result = json.loads(clarify_tool(
            "",
            questions=[{"question": "One?"}, {"question": "Two?"}],
            callback=legacy_cb,
        ))
        assert calls == ["One?", "Two?"]
        assert [r["user_response"] for r in result["responses"]] == ["", "second"]
        assert "timed_out" not in result

    def test_single_question_result_shape_unchanged(self):
        """No `questions` arg keeps the historic result keys exactly."""
        def cb(question, choices):
            return "blue"

        result = json.loads(clarify_tool(
            "Color?", choices=["red", "blue"], callback=cb,
        ))
        assert set(result.keys()) == {"question", "choices_offered", "user_response"}


class TestRegistryBatchPassThrough:
    """The registered handler forwards `questions` from tool args."""

    def test_handler_passes_questions(self):
        from tools.registry import registry
        entry = registry.get_entry("clarify")
        seen = {}

        def cb(question, choices, multi_select=False, questions=None):
            seen["questions"] = questions
            return {"answers": {"q0": "yes"}}

        result = json.loads(entry.handler(
            {"questions": [{"question": "Go?"}]},
            callback=cb,
        ))
        assert seen["questions"][0]["question"] == "Go?"
        assert result["responses"][0]["user_response"] == "yes"
