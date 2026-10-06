"""Regression tests for #118825: delegate_task must reject undeclared per-task
fields (e.g. a per-task model/provider pin) instead of silently ignoring
them."""

from tools.delegate_tool_tasks import _normalize_task_list

_MAX = 8
_ROLE = "worker"


def test_unknown_task_field_is_rejected():
    task_list, err = _normalize_task_list(
        None,
        None,
        [{"goal": "Do the delegated work end to end", "model": "cheap-model"}],
        None,
        _ROLE,
        _MAX,
    )
    assert task_list is None
    assert err is not None
    assert "model" in err and "Task 0" in err


def test_unknown_provider_field_is_rejected():
    task_list, err = _normalize_task_list(
        None,
        None,
        [{"goal": "Do the delegated work end to end", "provider": "other"}],
        None,
        _ROLE,
        _MAX,
    )
    assert task_list is None
    assert err is not None
    assert "provider" in err


def test_unknown_field_rejected_on_second_task():
    task_list, err = _normalize_task_list(
        None,
        None,
        [
            {"goal": "First delegated task goal", "context": "a"},
            {"goal": "Second delegated task goal", "temperature": 0.2},
        ],
        None,
        _ROLE,
        _MAX,
    )
    assert task_list is None
    assert err is not None
    assert "temperature" in err and "Task 1" in err


def test_declared_fields_are_accepted():
    task_list, err = _normalize_task_list(
        None,
        None,
        [
            {
                "goal": "Do the delegated work end to end",
                "context": "background",
                "role": "worker",
                "output_schema": None,
                "images": None,
                "group": "batch-a",
            }
        ],
        None,
        _ROLE,
        _MAX,
    )
    assert err is None
    assert task_list is not None
    assert task_list[0]["goal"] == "Do the delegated work end to end"


def test_single_goal_form_still_works():
    task_list, err = _normalize_task_list(
        "Do the delegated work end to end", "ctx", None, None, _ROLE, _MAX
    )
    assert err is None
    assert task_list is not None
    assert task_list[0]["goal"] == "Do the delegated work end to end"
