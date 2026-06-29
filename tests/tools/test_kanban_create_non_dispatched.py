"""Tool-surface tests for native non-dispatched Kanban creation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    tid = None
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def payload(status="todo", correlation_key="tool-key", no_dispatch=True):
    return {
        "title": "native non-dispatched task",
        "assignee": "bafuxunan",
        "body": "Carries audit_task_contract.v1 for structured verdict metadata.",
        "initial_status": status,
        "no_dispatch": no_dispatch,
        "correlation_key": correlation_key,
        "metadata": {
            "audit_task_contract": {
                "schema_version": "audit_task_contract.v1",
                "verdict_source": "structured_completion_metadata",
            }
        },
    }


def test_tool_create_non_dispatched_todo_returns_read_after_create_proof(worker_env):
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create(payload(status="todo")))
    assert out["ok"] is True
    assert out["status"] == "todo"
    assert out["proof"]["no_dispatch"] is True
    assert out["proof"]["read_after_create_verified"] is True
    assert out["proof"]["worker_execution_started"] is False
    assert out["proof"]["dispatcher_pickup_count"] == 0


def test_tool_create_non_dispatched_ready_cannot_be_claimed(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create(payload(status="ready", correlation_key="ready-key")))
    assert out["ok"] is True
    with kb.connect() as conn:
        assert kb.claim_task(conn, out["task_id"]) is None


def test_tool_todo_ready_requires_no_dispatch(worker_env):
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create(payload(status="todo", no_dispatch=False)))
    assert out.get("ok") is not True
    assert "requires no_dispatch=true" in out["error"] or "no_dispatch must be true" in out["error"]


def test_tool_schema_exposes_non_dispatched_contract_fields():
    from tools.kanban_tools import KANBAN_CREATE_SCHEMA

    props = KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    assert props["initial_status"]["enum"] == ["running", "blocked", "todo", "ready"]
    for key in ["no_dispatch", "correlation_key", "metadata", "capability"]:
        assert key in props
