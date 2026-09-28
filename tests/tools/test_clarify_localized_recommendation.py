"""Localized option badges stay presentation metadata (issue #126059)."""
import json

import pytest

from tools.clarify_tool import clarify_tool, mark_recommended, strip_recommended


@pytest.mark.parametrize("marker", ["（推荐）", "(推荐)", "（推薦）", "(推薦)"])
@pytest.mark.parametrize("mode", ["single", "multi", "batch", "legacy_batch"])
def test_localized_marker_display_and_answer(marker, mode):
    choices = [f"Option A{marker}", "Option B"]
    seen = []

    def legacy(question, displayed, multi_select=False):
        seen.append(displayed)
        return json.dumps([displayed[0], displayed[1]]) if multi_select else displayed[0]

    def batch(question, displayed, *, questions):
        seen.append(questions[0]["choices"])
        return {"answers": {"q0": questions[0]["choices"][0]}}

    if mode in ("single", "multi"):
        result = json.loads(clarify_tool("Pick", choices, multi_select=mode == "multi", callback=legacy))
    else:
        result = json.loads(clarify_tool("", questions=[{"question": "Pick", "choices": choices}],
                                         callback=batch if mode == "batch" else legacy))["responses"][0]
    assert seen == [choices]  # Preserve the supplied locale; don't append English.
    assert result["user_response"] == (["Option A", "Option B"] if mode == "multi" else "Option A")
    assert choices == [f"Option A{marker}", "Option B"]


@pytest.mark.parametrize("text", ["Option A（推荐） (Recommended)", "Option A (recommended)（推薦）"])
def test_preexisting_stacked_markers_are_removed(text):
    assert strip_recommended(text) == "Option A"


@pytest.mark.parametrize("text", ["Recommended reading", "Option A (optional)", "推荐算法", "A（推荐理由）", "A（推荐） details", ""])
def test_ordinary_choice_text_is_preserved(text):
    assert strip_recommended(text) == text


def test_normal_empty_and_single_choices_preserve_existing_contract():
    assert mark_recommended([]) == []
    assert mark_recommended(["A"]) == ["A"]
    marked = mark_recommended(["A", "B"])
    assert marked == ["A (Recommended)", "B"]
    assert mark_recommended(marked) == marked
    assert strip_recommended(marked[0]) == "A"


def test_callback_failure_is_not_retried():
    calls = []

    def fail(question, choices):
        calls.append(choices)
        raise RuntimeError("display unavailable")

    result = json.loads(clarify_tool("Pick", ["A（推荐）", "B"], callback=fail))
    assert "display unavailable" in result["error"]
    assert len(calls) == 1
