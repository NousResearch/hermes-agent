"""Kanban agent tools expose task due dates: kanban_create sets one, kanban_list filters and
sorts by it, kanban_show / kanban_list return it as epoch AND configured-zone ISO-8601.

Driven through ``registry.dispatch`` so the schema's argument validation is exercised too.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import hermes_time


@pytest.fixture
def board(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    monkeypatch.setenv("HERMES_TIMEZONE", "UTC")
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_SESSION_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hermes_time.reset_cache()
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    import tools.kanban_tools  # registers the kanban toolset
    from tools.registry import registry
    yield lambda name, args: json.loads(registry.dispatch(name, args))
    hermes_time.reset_cache()


def test_create_list_show_carry_due_date(board):
    made = board("kanban_create", {"title": "pay rent", "assignee": "me", "due": "2026-12-01T09:00:00Z"})
    assert made["ok"] and made["due_at_iso"] == "2026-12-01T09:00:00+00:00"
    late = board("kanban_create", {"title": "reply to landlord", "assignee": "me",
                                   "due": str(int(time.time()) - 3600)})
    plain = board("kanban_create", {"title": "someday", "assignee": "me", "priority": 5})

    listed = board("kanban_list", {"sort": "due"})
    assert [t["id"] for t in listed["tasks"]] == [late["task_id"], made["task_id"], plain["task_id"]]
    assert listed["tasks"][1]["due_at_iso"] == "2026-12-01T09:00:00+00:00"
    assert listed["tasks"][2]["due_at"] is None and listed["tasks"][2]["due_at_iso"] is None
    # Default order is still priority first.
    assert board("kanban_list", {})["tasks"][0]["id"] == plain["task_id"]

    overdue = board("kanban_list", {"due": "overdue"})
    assert [t["id"] for t in overdue["tasks"]] == [late["task_id"]]

    shown = board("kanban_show", {"task_id": made["task_id"]})["task"]
    assert shown["due_at"] == made["due_at"]


def test_bad_due_input_is_a_tool_error_and_creates_nothing(board):
    out = board("kanban_create", {"title": "x", "assignee": "me", "due": "next week"})
    assert "error" in out and "due date" in out["error"]
    assert board("kanban_list", {})["count"] == 0
    rejected = board("kanban_list", {"due": "someday"})
    assert "error" in rejected


@pytest.mark.parametrize("due", [1.5, 1796139000.5])
def test_float_due_is_refused_not_truncated(board, due):
    """A JSON number with a fraction must not be stored as a truncated epoch (1.5 -> 1970)."""
    out = board("kanban_create", {"title": "x", "assignee": "me", "due": due})
    assert "error" in out, out
    assert board("kanban_list", {})["count"] == 0
