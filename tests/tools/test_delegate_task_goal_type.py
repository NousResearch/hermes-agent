"""Regression: non-string task ``goal`` values must produce a friendly tool error, not crash.

A model can emit ``tasks=[{"goal": 123}]`` (goal as a number/bool/etc). ``_normalize_task_list``
already gives a clean error for non-dict task entries and short-circuit strings, but the
per-task goal presence check called ``task.get("goal", "").strip()`` directly, raising
AttributeError for any non-string goal. The batch gate (line above) deliberately coerces with
``str(task.get("goal", ""))`` — this brings the presence check in line with that contract.
"""

import pytest

from tools.delegate_tool_tasks import _normalize_task_list


@pytest.mark.parametrize("bad_goal", [123, 4.5, True, None, ["do thing"], {"g": 1}])
def test_non_string_task_goal_returns_error_not_crash(bad_goal):
    tasks = [{"goal": bad_goal}, {"goal": "A second, long enough goal for the batch gate."}]
    task_list, err = _normalize_task_list(None, None, tasks, None, "leaf", 3)
    assert task_list is None
    assert err is not None
    assert "Task 0" in err and "goal" in err


def test_single_task_non_string_goal_returns_error_not_crash():
    tasks = [{"goal": 99}]
    task_list, err = _normalize_task_list(None, None, tasks, None, "leaf", 3)
    assert task_list is None
    assert err is not None
    assert "Task 0" in err
