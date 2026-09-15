"""Tool-created review children retain their explicit contract through dispatch."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from tools import kanban_tools  # noqa: F401 -- register production handlers
from tools.registry import registry


@pytest.mark.parametrize("step", ["review", "release"])
def test_create_show_and_dispatch_review_child(tmp_path, monkeypatch, step):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(home / "workspaces"))
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK", "HERMES_SESSION_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    kb.init_db()
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="upstream", assignee="builder")
        args = {"title": "downstream", "assignee": "independent", "parents": [parent], "review_child_step": step}
        created = json.loads(registry.dispatch("kanban_create", args))
        child = created["task_id"]
        shown = json.loads(registry.dispatch("kanban_show", {"task_id": child}))["task"]
        assert shown["workflow_template_id"] == "hermes:review_child_v1"
        assert shown["current_step_key"] == step
        assert shown["status"] == "todo"
        assert kb.complete_task(conn, parent, summary="ready")
        kb.add_comment(conn, child, author="builder", body="https://github.com/example/repo/pull/123")
        result = kbd.dispatch_once(conn, dry_run=True)
        assert child in {entry[0] for entry in result.spawned}
        invalid = json.loads(registry.dispatch("kanban_create", {**args, "review_child_step": "implementation"}))
        assert "error" in invalid
