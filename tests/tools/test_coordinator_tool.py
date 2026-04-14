"""Tests for tools.coordinator_tool."""

import json

from agent.task_manager import TaskManager
from tools import coordinator_tool
from tools.registry import registry


def test_coordinate_plan_action():
    payload = json.loads(
        registry.dispatch(
            "coordinate",
            {
                "action": "plan",
                "task_description": "Ship the release",
                "subtasks": [
                    {"title": "Prepare notes"},
                    {"title": "Deploy"},
                ],
            },
        )
    )

    assert payload["source_task"] == "Ship the release"
    assert [step["title"] for step in payload["steps"]] == ["Prepare notes", "Deploy"]
    assert all(step["status"] == "pending" for step in payload["steps"])


def test_coordinate_execute_action(tmp_path, monkeypatch):
    manager = TaskManager(storage_path=tmp_path / "state" / "tasks.json")
    monkeypatch.setattr(coordinator_tool, "TaskManager", lambda *args, **kwargs: manager)

    payload = json.loads(
        registry.dispatch(
            "coordinate",
            {
                "action": "execute",
                "task_description": "Ship the release",
                "subtasks": [
                    {"title": "Prepare notes"},
                    {"title": "Deploy", "needs_external_reasoning": True},
                ],
            },
        )
    )

    assert all(step["status"] == "completed" for step in payload["steps"])
    assert all(step.get("task_id") for step in payload["steps"])
    tasks = manager.list(include_all=True)
    assert len(tasks) == 2
    assert {task["status"] for task in tasks} == {"completed"}


def test_coordinate_summarize_action():
    payload = json.loads(
        registry.dispatch(
            "coordinate",
            {
                "action": "summarize",
                "task_description": "Ship the release",
                "subtasks": [{"title": "Prepare notes"}],
            },
        )
    )

    assert "Ship the release" in payload["summary"]
    assert "Prepare notes" in payload["summary"]
    assert "Plan " in payload["summary"]
