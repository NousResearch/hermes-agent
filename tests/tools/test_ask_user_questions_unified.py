"""End-to-end contracts for AskUserQuestions over the shared clarify lane."""

from __future__ import annotations

import json

from tools.ask_user_questions_tool import ask_user_questions_tool


def _questions() -> list[dict]:
    return [
        {
            "header": "Scope",
            "question": "Which areas should be included?",
            "options": [
                {"label": "API", "description": "Backend endpoints"},
                {
                    "label": "UI",
                    "description": "Desktop and terminal surfaces",
                    "recommended": True,
                },
            ],
            "multiSelect": True,
        },
        {
            "header": "Notes",
            "question": "Anything else?",
            "options": [{"label": "Nothing else"}, {"label": "Add detail"}],
        },
    ]


def test_shared_clarify_lane_supports_multiselect_and_inline_other() -> None:
    captured: dict = {}

    def clarify_callback(_title, _choices, *, questions=None, multi_select=False):
        captured["questions"] = questions
        return {
            "answers": {
                "q0": json.dumps(["UI", "Accessibility"]),
                "q1": "Keep the existing shortcuts",
            }
        }

    result = json.loads(
        ask_user_questions_tool(_questions(), clarify_callback=clarify_callback)
    )

    wire = captured["questions"]
    assert wire[0]["choices"] == ["API", "UI (Recommended)"]
    assert wire[0]["multi_select"] is True
    assert wire[0]["header"] == "Scope"
    assert wire[0]["options"][1]["description"] == "Desktop and terminal surfaces"

    assert result["answers"][0] == {
        "index": 0,
        "question": "Which areas should be included?",
        "answer": ["UI", "Accessibility"],
        "status": "answered",
        "needs_text": False,
    }
    assert result["answers"][1]["answer"] == "Keep the existing shortcuts"
    assert result["answers"][1]["status"] == "answered"
    assert result["needs_followup"] == []
    assert result["responses"][0]["user_response"] == ["UI", "Accessibility"]


def test_timeout_and_cancel_are_distinct_and_preserve_partial_answers() -> None:
    def timed_out(_title, _choices, *, questions=None, multi_select=False):
        return {"answers": {"q0": json.dumps(["API"])}, "timed_out": True}

    timeout_result = json.loads(
        ask_user_questions_tool(_questions(), clarify_callback=timed_out)
    )
    assert timeout_result["timed_out"] is True
    assert timeout_result["answers"][0]["status"] == "answered"
    assert timeout_result["answers"][1]["status"] == "timed_out"

    def cancelled(_title, _choices, *, questions=None, multi_select=False):
        return {"answers": {"q0": json.dumps(["UI"])}, "cancelled": True}

    cancel_result = json.loads(
        ask_user_questions_tool(_questions(), clarify_callback=cancelled)
    )
    assert cancel_result["cancelled"] is True
    assert cancel_result["answers"][0]["status"] == "answered"
    assert cancel_result["answers"][1]["status"] == "cancelled"


def test_legacy_callback_remains_compatible() -> None:
    result = json.loads(
        ask_user_questions_tool(
            _questions(),
            callback=lambda questions: {0: "UI", 1: "Nothing else"},
        )
    )

    assert [row["answer"] for row in result["answers"]] == ["UI", "Nothing else"]
    assert all(row["status"] == "answered" for row in result["answers"])
