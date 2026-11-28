"""Creation schema dispatch and cap/usage readback through the real registry."""
import json
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def test_registry_create_and_show_worker_run_budget(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from tools import kanban_tools  # noqa: F401 -- registers the existing toolset
    from tools.registry import registry

    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    result = json.loads(registry.dispatch("kanban_create", {
        "title": "bounded child", "assignee": "worker", "max_worker_runs": 1,
    }))
    assert result.get("ok"), result
    tid = result["task_id"]
    shown = json.loads(registry.dispatch("kanban_show", {"task_id": tid}))
    assert shown["task"]["max_worker_runs"] == 1
    assert shown["task"]["worker_runs_used"] == 0
    invalid = json.loads(registry.dispatch("kanban_create", {
        "title": "bad", "assignee": "worker", "max_worker_runs": -1,
    }))
    assert "error" in invalid
