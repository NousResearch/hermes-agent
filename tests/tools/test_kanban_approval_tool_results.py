"""Lossless durable-approval markers at terminal/execute_code boundaries."""

from __future__ import annotations

import json

from tools import code_execution_tool, terminal_tool


_PENDING = {
    "approved": False,
    "status": "kanban_approval_pending",
    "kanban_approval_pending": True,
    "request_id": "apr_123",
    "display_target": "redacted target",
    "description": "destructive command",
}


def test_terminal_preserves_durable_approval_marker(monkeypatch):
    class FakeEnv:
        env = {}

        def execute(self, *_args, **_kwargs):
            raise AssertionError("parked command executed")

    task_id = "kanban-terminal"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": "/tmp",
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_a, **_k: _PENDING)

    result = json.loads(
        terminal_tool.terminal_tool(command="rm -rf /tmp/demo", task_id=task_id)
    )

    assert result["status"] == "kanban_approval_pending"
    assert result["kanban_approval_pending"] is True
    assert result["request_id"] == "apr_123"
    assert result["display_target"] == "redacted target"
    assert result["description"] == "destructive command"


def test_execute_code_preserves_durable_approval_marker(monkeypatch):
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._docker_has_host_access", lambda _config: False
    )
    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard", lambda *_a, **_k: _PENDING
    )

    result = json.loads(code_execution_tool.execute_code("print('hello')"))

    assert result["status"] == "kanban_approval_pending"
    assert result["kanban_approval_pending"] is True
    assert result["request_id"] == "apr_123"
    assert result["display_target"] == "redacted target"
    assert result["description"] == "destructive command"
    assert result["tool_calls_made"] == 0
