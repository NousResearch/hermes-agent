"""The existing native tools carry observed task-event guards end to end."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import kanban_tools as tools


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def test_show_block_and_unblock_tools_preserve_newer_stop(board):
    task_id = kb.create_task(board, title="Tool assignment", assignee="worker", initial_status="blocked")
    shown = json.loads(tools._handle_show({"task_id": task_id}))
    activation = shown["latest_event_id"]
    assert activation == shown["events"][-1]["id"]
    stopped = json.loads(tools._handle_block({"task_id": task_id, "kind": "needs_input",
                                              "reason": "Stop", "expected_event_id": activation}))
    assert stopped.get("status") == "blocked"
    before = kb.list_events(board, task_id)
    refused = json.loads(tools._handle_unblock({"task_id": task_id, "expected_event_id": activation}))
    assert "expected event changed" in refused.get("error", "")
    assert kb.list_events(board, task_id) == before
    current = json.loads(tools._handle_show({"task_id": task_id}))["latest_event_id"]
    resumed = json.loads(tools._handle_unblock({"task_id": task_id, "expected_event_id": current}))
    assert resumed.get("status") == "ready"


def test_tools_refuse_malformed_guard_without_mutation(board):
    task_id = kb.create_task(board, title="Malformed guard", assignee="worker", initial_status="blocked")
    before = kb.list_events(board, task_id)
    for handler, args in (
        (tools._handle_unblock, {"task_id": task_id}),
        (tools._handle_block, {"task_id": task_id, "reason": "Stop", "kind": "needs_input"}),
    ):
        refused = json.loads(handler({**args, "expected_event_id": True}))
        assert "expected_event_id" in refused.get("error", "")
    assert kb.list_events(board, task_id) == before
    assert kb.get_task(board, task_id).status == "blocked"
