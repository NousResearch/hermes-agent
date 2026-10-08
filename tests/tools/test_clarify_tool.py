"""Tests for tools/clarify_tool.py - Interactive clarifying questions."""

import json

from tools.clarify_tool import (
    clarify_tool,
    MAX_CHOICES,
    MAX_QUESTIONS,
    CLARIFY_SCHEMA,
)


def _ask(callback, questions):
    return json.loads(clarify_tool(questions, callback=callback))


def _reply(answers, outcome="submitted", **extra):
    return {"answers": answers, "outcome": outcome, **extra}


class TestClarifyToolBasics:
    """Basic functionality tests for clarify_tool."""

    def test_simple_question_with_callback(self):
        """Should return user response for simple question."""
        def mock_callback(questions):
            assert questions[0]["question"] == "What color?"
            assert questions[0]["choices"] is None
            return _reply({"q0": "blue"})

        result = _ask(mock_callback, [{"question": "What color?"}])
        response = result["responses"][0]
        assert response["question"] == "What color?"
        assert response["choices_offered"] is None
        assert response["status"] == "answered"
        assert response["user_response"] == "blue"
        assert result["outcome"] == "submitted"

    def test_no_callback_returns_error(self):
        """Should return error when no callback is provided."""
        result = json.loads(clarify_tool([{"question": "What do you want?"}]))
        assert "error" in result
        assert "responses" not in result


class TestClarifyToolChoicesValidation:
    """Tests for choices parameter validation."""

    def test_choices_trimmed_to_max(self):
        """Should trim choices to MAX_CHOICES."""
        choices_passed = []

        def mock_callback(questions):
            choices_passed.extend(questions[0]["choices"] or [])
            return _reply({"q0": "picked"})

        many_choices = ["a", "b", "c", "d", "e", "f", "g"]
        _ask(mock_callback, [{"question": "Pick one", "choices": many_choices}])

        assert len(choices_passed) == MAX_CHOICES


class TestClarifyToolCallbackHandling:
    """Tests for callback error handling."""

    def test_callback_exception_returns_error(self):
        """Should return error if callback raises exception."""
        def failing_callback(questions):
            raise RuntimeError("User cancelled")

        result = _ask(failing_callback, [{"question": "Question?"}])
        assert "error" in result
        assert "User cancelled" in result["error"]

    def test_non_dict_string_reply_does_not_crash(self):
        """A surface returning a bare string must degrade to 'undelivered', not
        raise ``'str' object has no attribute 'get'`` and swallow the wait."""
        result = _ask(lambda questions: "some string reply", [{"question": "Pick", "choices": ["A", "B"]}])
        assert "error" not in result
        assert result["outcome"] == "undelivered"
        assert result["notice"] == "some string reply"

    def test_non_dict_none_reply_does_not_crash(self):
        result = _ask(lambda questions: None, [{"question": "Pick", "choices": ["A", "B"]}])
        assert "error" not in result
        assert result["outcome"] == "undelivered"

    def test_non_dict_empty_string_reply_does_not_crash(self):
        result = _ask(lambda questions: "   ", [{"question": "Pick", "choices": ["A", "B"]}])
        assert "error" not in result
        assert result["outcome"] == "undelivered"
        assert "notice" not in result

    def test_user_response_stripped(self):
        """User response should be stripped of whitespace."""
        def mock_callback(questions):
            return _reply({"q0": "  response with spaces  \n"})

        result = _ask(mock_callback, [{"question": "Q?"}])
        assert result["responses"][0]["user_response"] == "response with spaces"


class TestClarifySchema:
    """Tests for the OpenAI function-calling schema."""

    def test_schema_questions_param_is_required_and_capped(self):
        """`questions` is the single documented way to call (a single question
        is a one-entry array) and carries the batch cap so the model sees the
        limit."""
        params = CLARIFY_SCHEMA["parameters"]
        assert params["required"] == ["questions"]
        assert params["properties"]["questions"]["maxItems"] == MAX_QUESTIONS


class TestClarifyToolMultiSelect:
    """Tests for multi_select (checkbox) support added to clarify_tool."""

    def test_multi_select_true_returns_list(self):
        """When multi_select=True, user_response should be a list of strings."""
        def mock_callback(questions):
            return _reply({"q0": ["red", "blue"]})

        result = _ask(mock_callback, [{
            "question": "Which colors?",
            "choices": ["red", "blue", "green"],
            "multi_select": True,
        }])
        response = result["responses"][0]
        assert response["user_response"] == ["red", "blue"]
        assert isinstance(response["user_response"], list)

    def test_multi_select_single_choice_still_list(self):
        """Even a single selection should be a list when multi_select=True."""
        def mock_callback(questions):
            return _reply({"q0": '["red"]'})

        result = _ask(mock_callback, [{
            "question": "Which color?",
            "choices": ["red", "blue"],
            "multi_select": True,
        }])
        response = result["responses"][0]
        assert response["user_response"] == ["red"]
        assert isinstance(response["user_response"], list)


class TestClarifyRecommendedLabel:
    """The first choice is the agent's pick and is labelled as such.

    The schema tells the model to order choices best-first, so the tool tags
    element 0 with "(Recommended)" at the one platform-agnostic entry point —
    CLI, TUI, desktop, and messaging adapters all inherit the same label. The
    label is presentation only: it never appears in the answer the agent reads.
    """

    def test_first_choice_is_labelled(self):
        seen = []

        def cb(questions):
            seen.extend(questions[0]["choices"] or [])
            return _reply({"q0": seen[1]})

        _ask(cb, [{"question": "Pick", "choices": ["Rebase", "Merge"]}])
        assert seen == ["Rebase (Recommended)", "Merge"]

    def test_answer_strips_the_label(self):
        """Picking the recommended option returns the bare option text."""
        def cb(questions):
            return _reply({"q0": questions[0]["choices"][0]})

        result = _ask(cb, [{"question": "Pick", "choices": ["Rebase", "Merge"]}])
        response = result["responses"][0]
        assert response["user_response"] == "Rebase"
        assert response["choices_offered"] == ["Rebase", "Merge"]

    def test_multi_select_answers_strip_the_label(self):
        def cb(questions):
            return _reply({"q0": questions[0]["choices"][:2]})

        result = _ask(cb, [{
            "question": "Pick some",
            "choices": ["Rebase", "Merge", "Squash"],
            "multi_select": True,
        }])
        assert result["responses"][0]["user_response"] == ["Rebase", "Merge"]

    def test_single_choice_is_not_labelled(self):
        """One option isn't a recommendation — there's nothing to prefer it over."""
        seen = []

        def cb(questions):
            seen.extend(questions[0]["choices"] or [])
            return _reply({"q0": seen[0]})

        _ask(cb, [{"question": "Confirm", "choices": ["Ship it"]}])
        assert seen == ["Ship it"]

    def test_label_is_not_doubled(self):
        """A model that wrote its own label doesn't get a second one."""
        seen = []

        def cb(questions):
            seen.extend(questions[0]["choices"] or [])
            return _reply({"q0": seen[0]})

        _ask(cb, [{"question": "Pick", "choices": ["Rebase (recommended)", "Merge"]}])
        assert seen == ["Rebase (recommended)", "Merge"]

    def test_open_ended_unaffected(self):
        def cb(questions):
            assert questions[0]["choices"] is None
            return _reply({"q0": "whatever"})

        result = _ask(cb, [{"question": "Thoughts?"}])
        response = result["responses"][0]
        assert response["choices_offered"] is None
        assert response["user_response"] == "whatever"


class TestRegistryMultiSelectPassThrough:
    """The registered tool handler must forward multi_select from tool args."""

    def test_handler_passes_multi_select(self):
        from tools.registry import registry
        entry = registry.get_entry("clarify")
        seen = {}

        def cb(questions):
            seen["multi"] = questions[0]["multi_select"]
            return _reply({"q0": ["a", "b"]})

        result = json.loads(entry.handler(
            {"questions": [{"question": "Pick", "choices": ["a", "b"], "multi_select": True}]},
            callback=cb,
        ))
        assert seen["multi"] is True
        assert result["responses"][0]["user_response"] == ["a", "b"]

    def test_handler_default_single_select(self):
        from tools.registry import registry
        entry = registry.get_entry("clarify")
        seen = {}

        def cb(questions):
            seen["multi"] = questions[0]["multi_select"]
            return _reply({"q0": "a"})

        result = json.loads(entry.handler(
            {"questions": [{"question": "Pick", "choices": ["a", "b"]}]},
            callback=cb,
        ))
        assert seen["multi"] is False
        assert result["responses"][0]["user_response"] == "a"


class TestClarifyBatchValidation:
    """Validation of the `questions` batch parameter (issue #18450)."""

    def test_batch_rejects_more_than_five(self):
        result = _ask(lambda questions: _reply({}), [{"question": f"Q{i}?"} for i in range(6)])
        assert "error" in result

    def test_batch_rejects_blank_question_text(self):
        result = _ask(lambda questions: _reply({}), [{"question": "Real?"}, {"question": "   "}])
        assert "error" in result

    def test_all_blank_choices_are_an_error_not_open_ended(self):
        """#73152: a choices list whose entries are all blank must not quietly become a free-text card."""
        asked = []
        result = _ask(lambda questions: asked.append(questions) or _reply({}),
                      [{"question": "Pick one?", "choices": ["", "   "]}])
        assert "error" in result and "blank" in result["error"]
        assert asked == []

    def test_empty_choices_list_stays_open_ended(self):
        seen = {}
        _ask(lambda questions: seen.setdefault("q", questions) and _reply({"q0": "free text"}),
             [{"question": "Anything?", "choices": []}])
        assert seen["q"][0]["choices"] is None

    def test_over_limit_choice_is_rejected_at_the_source(self):
        """#124127: a choice longer than a surface renders is refused, not silently dropped downstream."""
        from tools.clarify_tool import MAX_CHOICE_CHARS
        long_choice = "x" * (MAX_CHOICE_CHARS + 1)
        result = _ask(lambda questions: _reply({}), [{"question": "Pick?", "choices": ["short", long_choice]}])
        assert "error" in result and str(MAX_CHOICE_CHARS) in result["error"]

    def test_long_multi_line_choice_within_limit_is_kept(self):
        seen = {}
        choice = "Option A\n" + "detail " * 60
        _ask(lambda questions: seen.setdefault("q", questions) and _reply({"q0": "Option A"}),
             [{"question": "Pick?", "choices": [choice, "B"]}])
        assert seen["q"][0]["choices_offered"][0] == choice.strip()

    def test_batch_rejects_non_list(self):
        result = _ask(lambda questions: _reply({}), {"question": "Q?"})
        assert "error" in result

    def test_batch_choices_capped_and_labelled_per_question(self):
        """Each question gets the full choice pipeline: cap, label."""
        seen = {}

        def cb(questions):
            seen["questions"] = questions
            return _reply({"q0": "a", "q1": "Loose layout"})

        _ask(cb, [
            {"question": "Pick letter", "choices": ["a", "b", "c", "d", "e", "f"]},
            {"question": "Pick layout", "choices": ["Loose layout", "Tight"]},
        ])
        q0, q1 = seen["questions"]
        assert len(q0["choices"]) == MAX_CHOICES
        assert q0["choices"][0] == "a (Recommended)"
        assert q1["choices"] == ["Loose layout (Recommended)", "Tight"]

    def test_batch_internal_ids_are_stable(self):
        """Wire ids are q0..qN."""
        seen = {}

        def cb(questions):
            seen["questions"] = questions
            return _reply({"q0": "A", "q1": "B"})

        _ask(cb, [{"question": "Which approach?"}, {"question": "Timeline?"}])
        assert [q["qid"] for q in seen["questions"]] == ["q0", "q1"]

    def test_batch_multi_select_needs_choices(self):
        """multi_select is only honored when the question has choices."""
        seen = {}

        def cb(questions):
            seen["questions"] = questions
            return _reply({"q0": "free text"})

        _ask(cb, [{"question": "Thoughts?", "multi_select": True}])
        assert seen["questions"][0]["multi_select"] is False


class TestClarifyBatchDispatch:
    """The callback gets the list once and answers by qid."""

    def test_batch_callback_receives_list_once(self):
        calls = []

        def cb(questions):
            calls.append(questions)
            return _reply({"q0": "x", "q1": "y"})

        result = _ask(cb, [{"question": "One?"}, {"question": "Two?"}])
        assert len(calls) == 1
        assert [r["user_response"] for r in result["responses"]] == ["x", "y"]

    def test_batch_multi_select_answer_parsed_to_list(self):
        def cb(questions):
            return _reply({"q0": '["red", "blue"]'})

        result = _ask(cb, [{
            "question": "Colors?",
            "choices": ["red", "blue", "green"],
            "multi_select": True,
        }])
        assert result["responses"][0]["user_response"] == ["red", "blue"]

    def test_batch_timed_out_keeps_partials(self):
        """Timeout keeps the locked answers and marks the rest unanswered."""
        def cb(questions):
            return _reply({"q0": "kept"}, outcome="timed_out", notice="no reply")

        result = _ask(cb, [{"question": "One?"}, {"question": "Two?"}])
        assert result["outcome"] == "timed_out"
        assert result["notice"] == "no reply"
        assert result["responses"][0]["status"] == "answered"
        assert result["responses"][0]["user_response"] == "kept"
        assert result["responses"][1]["status"] == "unanswered"
        assert result["responses"][1]["user_response"] is None

    def test_batch_skipped_is_not_unanswered(self):
        """A question locked empty is skipped; a missing one is unanswered."""
        def cb(questions):
            return _reply({"q0": None})

        result = _ask(cb, [{"question": "One?"}, {"question": "Two?"}])
        assert [r["status"] for r in result["responses"]] == ["skipped", "unanswered"]
        assert result["outcome"] == "submitted"
        assert "notice" not in result


class TestRegistryBatchPassThrough:
    """The registered handler forwards `questions` from tool args."""

    def test_handler_passes_questions(self):
        from tools.registry import registry
        entry = registry.get_entry("clarify")
        seen = {}

        def cb(questions):
            seen["questions"] = questions
            return _reply({"q0": "yes"})

        result = json.loads(entry.handler(
            {"questions": [{"question": "Go?"}]},
            callback=cb,
        ))
        assert seen["questions"][0]["question"] == "Go?"
        assert result["responses"][0]["user_response"] == "yes"
