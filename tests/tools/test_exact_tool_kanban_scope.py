"""Restricted task workers cannot use their permitted lifecycle names cross-task."""
import json
from unittest.mock import patch

import pytest

from tools import kanban_tools as kt


@pytest.mark.parametrize("handler", [kt._handle_show, kt._handle_comment])
def test_exact_grant_worker_is_fenced_to_own_task(tmp_path, monkeypatch, handler):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "assigned-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")
    (tmp_path / "config.yaml").write_text(json.dumps({"agent": {"allowed_tools": [
        "kanban_show", "kanban_comment", "kanban_heartbeat", "kanban_complete", "kanban_block",
    ]}}))
    with patch.object(kt, "_board", side_effect=AssertionError("board should not be reached")) as board:
        result = json.loads(handler({"task_id": "other-task", "body": "not mine"} if handler == kt._handle_comment else {"task_id": "other-task"}))
    assert "worker is scoped to task assigned-task" in result["error"]
    board.assert_not_called()


@pytest.mark.parametrize("payload", [
    {"metadata": {"_staged_artifacts": [{"stored_path": "/outside/file"}]}},
    {"metadata": {"artifacts": ["/outside/file"]}},
    {"artifacts": ["/outside/file"]},
])
def test_restricted_completion_refuses_file_delivery_before_board_access(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "assigned-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")
    (tmp_path / "config.yaml").write_text(json.dumps({"agent": {"allowed_tools": ["kanban_complete"]}}))
    with patch.object(kt, "_board", side_effect=AssertionError("board should not be reached")) as board:
        result = json.loads(kt._handle_complete({"task_id": "assigned-task", "summary": "Fixture", **payload}))
    assert "Restricted worker cannot supply artifact paths or internal metadata" in result["error"]
    board.assert_not_called()
