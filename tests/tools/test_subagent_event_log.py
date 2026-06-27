import json
from pathlib import Path

import pytest


class _Parent:
    session_id = "parent-session"
    _current_task_id = "parent-task"
    _current_turn_id = "turn-1"
    _active_children = []
    _active_children_lock = None


class _Child:
    model = "test-model"
    enabled_toolsets = ["terminal", "file"]
    session_id = "child-session"
    _subagent_id = "sa-test-1234"
    _parent_subagent_id = "parent-sa"
    _delegate_depth = 1
    _delegate_role = "leaf"
    tool_progress_callback = None
    session_prompt_tokens = 3
    session_completion_tokens = 5
    session_reasoning_tokens = 0
    session_estimated_cost_usd = 0.01

    def get_activity_summary(self):
        return {"api_call_count": 1, "current_tool": None, "max_iterations": 50}

    def run_conversation(self, user_message, task_id, stream_callback=None):
        self.seen_task_id = task_id
        return {
            "final_response": "done " + user_message,
            "completed": True,
            "api_calls": 2,
            "messages": [],
        }

    def close(self):
        self.closed = True


def _read_events(home: Path):
    path = home / "afk" / "subagent_events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_single_child_writes_spawn_and_terminal_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import delegate_tool

    result = delegate_tool._run_single_child(
        0,
        "implement a very long prompt " + ("x" * 1000),
        child=_Child(),
        parent_agent=_Parent(),
        task_count=1,
    )

    assert result["status"] == "completed"
    events = _read_events(tmp_path)
    assert [event["event_type"] for event in events] == [
        "subagent.spawned",
        "subagent.completed",
    ]

    spawned, completed = events
    assert spawned["schema_version"] == 1
    assert spawned["parent_session_id"] == "parent-session"
    assert spawned["child_session_id"] == "child-session"
    assert spawned["subagent_id"] == "sa-test-1234"
    assert spawned["hermes_subagent_id"] == "sa-test-1234"
    assert spawned["parent_subagent_id"] == "parent-sa"
    assert spawned["task_index"] == 0
    assert spawned["task_count"] == 1
    assert spawned["role"] == "leaf"
    assert spawned["depth"] == 0
    assert spawned["status"] == "running"
    assert spawned["model"] == "test-model"
    assert spawned["toolsets"] == ["terminal", "file"]
    assert len(spawned["goal_preview"]) < 400

    assert completed["status"] == "completed"
    assert completed["api_calls"] == 2
    assert completed["duration_seconds"] >= 0
    assert completed["summary_preview"].startswith("done implement")
    assert completed["files_read"] == []
    assert completed["files_written"] == []
    assert "prompt" not in completed
    assert "result" not in completed


def test_event_log_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import delegate_tool

    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {"subagent_event_log": {"enabled": False}},
    )

    result = delegate_tool._run_single_child(
        0,
        "disabled",
        child=_Child(),
        parent_agent=_Parent(),
        task_count=1,
    )

    assert result["status"] == "completed"
    assert not (tmp_path / "afk" / "subagent_events.jsonl").exists()


def test_event_log_write_failure_does_not_fail_delegation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import delegate_tool

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(delegate_tool, "_append_jsonl_locked", _boom)

    result = delegate_tool._run_single_child(
        0,
        "still succeeds",
        child=_Child(),
        parent_agent=_Parent(),
        task_count=1,
    )

    assert result["status"] == "completed"
