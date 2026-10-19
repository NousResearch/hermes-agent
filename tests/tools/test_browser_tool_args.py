"""Argument checks shared by both browser lanes (``tools/browser_tool_args.py``).

A bad argument must become one clear error before either lane sees it: the extension clamps or
drops what it does not understand, and the built-in browser only reads the arguments it declares.
"""

import pytest

from tools.browser_tool_args import BATCH_MAX_STEPS, invalid_arguments


def _batch(*steps, **extra):
    return {"actions": list(steps), **extra}


def test_a_well_formed_batch_passes():
    args = _batch(
        {"action": "click", "ref": "@e1"},
        {"action": "type", "ref": "@e1", "text": "hello", "clear": False},
        {"action": "press", "key": "a", "modifiers": ["Control"]},
        {"action": "scroll", "direction": "right", "amount": 300},
        {"action": "click", "x": 10, "y": 20.5},
        {"action": "navigate", "url": "https://example.test"},
        {"action": "browser_back"},
        stop_on_error=False,
    )
    assert invalid_arguments("browser_batch", args) is None


def test_the_step_limit_is_the_extensions():
    ok = _batch(*[{"action": "back"}] * BATCH_MAX_STEPS)
    assert invalid_arguments("browser_batch", ok) is None
    too_many = _batch(*[{"action": "back"}] * (BATCH_MAX_STEPS + 1))
    error = invalid_arguments("browser_batch", too_many)
    assert error and "at most 20" in error and "split" in error


@pytest.mark.parametrize("actions", [None, [], "click", {"action": "click"}])
def test_a_batch_needs_a_non_empty_list(actions):
    assert "non-empty list" in invalid_arguments("browser_batch", {"actions": actions})


@pytest.mark.parametrize(
    "step, fragment",
    [
        ("click", "must be an object"),
        ({"ref": "@e1"}, "which cannot run inside a batch"),
        ({"action": "snapshot"}, "which cannot run inside a batch"),
        ({"action": "tab_create", "url": "https://example.test"}, "which cannot run inside a batch"),
        ({"action": "click"}, "needs 'ref'"),
        ({"action": "click", "x": 5}, "needs both 'x' and 'y'"),
        ({"action": "click", "ref": "@e1", "x": "5"}, "'x' must be a number"),
        ({"action": "type", "ref": "@e1"}, "needs 'text'"),
        ({"action": "type", "ref": "@e1", "text": "x", "clear": "no"}, "'clear' must be true or false"),
        ({"action": "scroll", "direction": "sideways"}, "'direction' must be one of"),
        ({"action": "scroll", "direction": "down", "amount": 50}, "from 100 to 5000"),
        ({"action": "scroll", "direction": "down", "amount": 5001}, "from 100 to 5000"),
        ({"action": "scroll", "direction": "down", "amount": 250.5}, "'amount' must be a whole number"),
        ({"action": "scroll", "direction": "down", "ref": "@e1"}, "does not take ref"),
        ({"action": "press", "key": "Enter", "modifiers": ["Hyper"]}, "unknown modifier 'Hyper'"),
        ({"action": "press", "key": "Enter", "modifiers": "Shift"}, "'modifiers' must be a list"),
        ({"action": "navigate", "url": ""}, "needs 'url'"),
        ({"action": "drag", "ref": "@e1"}, "needs 'to_ref'"),
        ({"action": "back", "ref": "@e1"}, "takes no arguments"),
    ],
)
def test_a_bad_step_is_refused_with_its_position_and_cause(step, fragment):
    error = invalid_arguments("browser_batch", _batch({"action": "back"}, step))
    assert error and error.startswith("browser_batch: step 2") and fragment in error


def test_typing_an_empty_string_is_allowed_because_it_clears_a_field():
    assert invalid_arguments("browser_batch", _batch({"action": "type", "ref": "@e1", "text": ""})) is None
    assert invalid_arguments("browser_batch", _batch({"action": "fill", "ref": "@e1", "value": ""})) is None


def test_batch_level_arguments_are_checked():
    assert "'stop_on_error' must be true or false" in invalid_arguments(
        "browser_batch", _batch({"action": "back"}, stop_on_error="no"))
    assert "does not take steps" in invalid_arguments("browser_batch", {**_batch({"action": "back"}), "steps": []})


@pytest.mark.parametrize("tool", ("browser_tab_group", "browser_tab_ungroup"))
def test_the_tab_group_tools_take_a_list_of_distinct_tab_ids(tool):
    assert invalid_arguments(tool, {"tab_ids": [4]}) is None
    assert invalid_arguments(tool, {"tab_ids": [4, 9, 12]}) is None
    for bad, fragment in [
        ({}, "needs 'tab_ids'"),
        ({"tab_ids": []}, "non-empty list of tab ids"),
        ({"tab_ids": [4, True]}, "non-empty list of tab ids"),
        ({"tab_ids": [4, 2.5]}, "non-empty list of tab ids"),
        ({"tab_ids": [4, 4]}, "must list each tab once"),
        ({"tab_ids": [4], "tab_id": 4}, "does not take tab_id"),
        ({"tab_ids": [4], "windows": 1}, "does not take windows"),
    ]:
        error = invalid_arguments(tool, bad)
        assert error and error.startswith(f"{tool}: ") and fragment in error, (bad, error)


def test_tools_without_a_check_pass_through():
    assert invalid_arguments("browser_snapshot", {"anything": 1}) is None


def test_a_batch_names_its_tab_once_and_a_step_may_not():
    steps = [{"action": "click", "ref": "@e1"}]
    assert invalid_arguments("browser_batch", {"actions": steps, "tab_id": 3}) is None
    error = invalid_arguments("browser_batch", {"actions": [{"action": "click", "ref": "@e1", "tab_id": 3}]})
    assert error and "tab_id goes on the batch itself" in error


@pytest.mark.parametrize("tab_id", [0, -2, 1.5, "3", True])
def test_any_tool_refuses_a_tab_id_that_is_not_a_positive_whole_number(tab_id):
    error = invalid_arguments("browser_snapshot", {"tab_id": tab_id})
    assert error and "'tab_id' must be a tab id" in error
    assert invalid_arguments("browser_snapshot", {"tab_id": 3}) is None
