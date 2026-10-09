"""The registered create tool validates assignees against disk profiles."""

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools.registry import registry
import tools.kanban_tools  # noqa: F401 - registers the production tool


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    profile = home / "profiles" / "builder"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n")
    kb.init_db()
    return home


@pytest.mark.parametrize("assignee,expected", [
    ("missing", "triage"), ("builder", "todo"),
])
def test_tool_create_assignee_routes_through_registry(board_home, assignee, expected):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="open parent")
    result = json.loads(registry.dispatch("kanban_create", {
        "title": "child", "assignee": assignee, "parents": [parent],
        "initial_status": "blocked" if assignee == "missing" else "running",
    }))
    assert result["ok"] is True
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
        comments = kb.list_comments(conn, task.id)
    assert task.status == expected
    if assignee == "missing":
        assert "missing" in result["warning"]
        assert "builder" in result["valid_profiles"]
        assert any("missing" in comment.body and "triage" in comment.body for comment in comments)
    else:
        assert "warning" not in result
        assert not comments


@pytest.mark.parametrize("unavailable", [False, True])
def test_tool_empty_profile_enumeration_keeps_assignee(board_home, monkeypatch, unavailable):
    def profiles(**kwargs):
        if unavailable:
            raise OSError("profile store unavailable")
        return []

    monkeypatch.setattr(kb, "list_profiles_on_disk", profiles)
    result = json.loads(registry.dispatch("kanban_create", {
        "title": "card", "assignee": "missing",
    }))
    assert result["ok"] is True
    assert result["status"] == "ready"
    assert "warning" not in result


def test_tool_partial_profile_inventory_keeps_assignee(board_home, monkeypatch):
    original = Path.iterdir

    def failing_iterdir(path):
        if path == board_home / "profiles":
            raise OSError("profile directory unavailable")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    result = json.loads(registry.dispatch("kanban_create", {"title": "card", "assignee": "builder"}))
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
        assert not kb.list_comments(conn, task.id)
    assert task.status == "ready"
    assert task.assignee == "builder"
    assert "warning" not in result


@pytest.mark.parametrize("original_assignee,expected_status", [
    ("builder", "ready"), ("missing", "triage"),
])
def test_tool_idempotent_replay_has_no_new_warning(board_home, original_assignee, expected_status):
    key = f"replay-{original_assignee}"
    first = json.loads(registry.dispatch("kanban_create", {
        "title": "original", "assignee": original_assignee, "idempotency_key": key,
    }))
    replay = json.loads(registry.dispatch("kanban_create", {
        "title": "replay", "assignee": "missing", "idempotency_key": key,
    }))
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, first["task_id"])
        comments = kb.list_comments(conn, task.id)
    assert replay["task_id"] == first["task_id"]
    assert task.status == expected_status
    assert task.assignee == original_assignee
    assert len(comments) == (1 if original_assignee == "missing" else 0)
    assert ("warning" in first) == (original_assignee == "missing")
    assert "warning" not in replay
