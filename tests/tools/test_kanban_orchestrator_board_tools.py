"""kanban_archive / kanban_promote / kanban_unlink: orchestrator-only board tools.

Thin wrappers over ``kanban_db.archive_task`` / ``specify_triage_task`` /
``unlink_tasks``; these pin the gate (hidden from task workers, refused at
runtime for them), the happy paths, and the refusals.
"""
from __future__ import annotations

import json
import os

import pytest

_NEW_TOOLS = ("kanban_archive", "kanban_promote", "kanban_unlink")


@pytest.fixture
def board(monkeypatch, tmp_path):
    """Isolated board; caller is an orchestrator (no HERMES_KANBAN_TASK)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    yield kb, conn
    conn.close()


def _call(name, args):
    from tools import kanban_tools as kt
    handler = {"kanban_archive": kt._handle_archive, "kanban_promote": kt._handle_promote,
               "kanban_unlink": kt._handle_unlink}[name]
    return json.loads(handler(args))


def _events(kb, conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


# --- gate -------------------------------------------------------------------

def test_new_tools_use_the_orchestrator_gate():
    from tools import kanban_tools as kt
    from tools.registry import registry
    for name in _NEW_TOOLS:
        assert name in kt._ORCHESTRATOR_TOOLS
        assert registry._tools[name].check_fn is kt._check_kanban_orchestrator_mode


def test_gate_off_for_task_worker_and_on_for_orchestrator(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import kanban_tools as kt
    monkeypatch.setattr(kt, "_profile_has_kanban_toolset", lambda: True)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert kt._check_kanban_orchestrator_mode() is True
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    assert kt._check_kanban_orchestrator_mode() is False
    # Lifecycle tools stay visible to that worker; only routing tools hide.
    assert kt._check_kanban_mode() is True


@pytest.mark.parametrize("name,args", [
    ("kanban_archive", {"task_id": "X", "reason": "withdrawn"}),
    ("kanban_promote", {"task_id": "X"}),
    ("kanban_unlink", {"parent_id": "P", "child_id": "X"}),
])
def test_worker_is_refused_at_runtime_and_nothing_changes(board, monkeypatch, name, args):
    kb, conn = board
    parent = kb.create_task(conn, title="p", assignee="w")
    tid = kb.create_task(conn, title="x", assignee="w", parents=[parent], triage=True)
    args = {k: {"X": tid, "P": parent}.get(v, v) for k, v in args.items()}
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_some_worker")
    out = _call(name, args)
    assert "orchestrator-only" in out.get("error", "")
    assert kb.get_task(conn, tid).status == "triage"
    assert kb.parent_ids(conn, tid) == [parent]


@pytest.mark.parametrize("name,args", [
    ("kanban_archive", {"task_id": "t_x", "reason": "r"}),
    ("kanban_promote", {"task_id": "t_x"}),
    ("kanban_unlink", {"parent_id": "a", "child_id": "b"}),
])
def test_delegate_child_is_refused(board, monkeypatch, name, args):
    from tools import kanban_tools as kt
    monkeypatch.setattr(kt, "_delegation_ctx",
                        lambda pred, default: pred == "is_delegated_child_process_context" or default)
    out = _call(name, args)
    assert "delegate_task child" in out.get("error", "")


# --- kanban_archive ---------------------------------------------------------

def test_archive_happy_path_records_reason_as_comment(board):
    kb, conn = board
    tid = kb.create_task(conn, title="withdrawn", assignee="w")
    out = _call("kanban_archive", {"task_id": tid, "reason": "Withdrawn by John"})
    assert out == {"ok": True, "task_id": tid, "status": "archived"}
    assert kb.get_task(conn, tid).status == "archived"
    comments = kb.list_comments(conn, tid)
    assert [(c.author, c.body) for c in comments] == [
        ("orchestrator", "Archived: Withdrawn by John")]
    assert _events(kb, conn, tid, "archived")


def test_archive_requires_reason(board):
    kb, conn = board
    tid = kb.create_task(conn, title="t", assignee="w")
    out = _call("kanban_archive", {"task_id": tid, "reason": "  "})
    assert "reason is required" in out["error"]
    assert kb.get_task(conn, tid).status != "archived"


def test_archive_refuses_running_task_with_live_claim(board, monkeypatch):
    kb, conn = board
    tid = kb.create_task(conn, title="busy", assignee="w")
    kb.claim_task(conn, tid)
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (os.getpid(), tid))
    conn.commit()
    monkeypatch.setattr(kb, "_worker_alive", lambda pid, started_at=None: True)
    out = _call("kanban_archive", {"task_id": tid, "reason": "stop"})
    assert "live worker claim" in out["error"]
    assert "Nothing changed" in out["error"]
    assert kb.get_task(conn, tid).status == "running"
    assert kb.list_comments(conn, tid) == []


def test_archive_allows_running_task_whose_worker_is_gone(board, monkeypatch):
    kb, conn = board
    tid = kb.create_task(conn, title="orphan", assignee="w")
    kb.claim_task(conn, tid)
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (999999, tid))
    conn.commit()
    monkeypatch.setattr(kb, "_worker_alive", lambda pid, started_at=None: False)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **k: {"skipped": True})
    out = _call("kanban_archive", {"task_id": tid, "reason": "worker died"})
    assert out["ok"] is True
    assert kb.get_task(conn, tid).status == "archived"


def test_archive_unknown_and_already_archived(board):
    kb, conn = board
    assert "unknown task" in _call("kanban_archive", {"task_id": "t_nope", "reason": "r"})["error"]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.archive_task(conn, tid)
    out = _call("kanban_archive", {"task_id": tid, "reason": "again"})
    assert "already archived" in out["error"]
    assert kb.list_comments(conn, tid) == []


# --- kanban_promote ---------------------------------------------------------

def test_promote_triage_to_todo_lands_ready_when_parent_free(board):
    kb, conn = board
    tid = kb.create_task(conn, title="triaged", assignee="w", triage=True)
    out = _call("kanban_promote", {"task_id": tid})
    # Parent gating decides: no parents -> the re-gate promotes it straight to ready.
    assert out == {"ok": True, "task_id": tid, "status": "ready"}
    assert _events(kb, conn, tid, "specified")


def test_promote_triage_to_todo_stays_todo_under_open_parent(board):
    kb, conn = board
    parent = kb.create_task(conn, title="p", assignee="w")
    tid = kb.create_task(conn, title="c", assignee="w", parents=[parent], triage=True)
    out = _call("kanban_promote", {"task_id": tid, "to": "todo"})
    assert out["ok"] is True and out["status"] == "todo"


def test_promote_to_ready_refuses_open_parent_without_mutation(board):
    kb, conn = board
    parent = kb.create_task(conn, title="p", assignee="w")
    tid = kb.create_task(conn, title="c", assignee="w", parents=[parent], triage=True)
    out = _call("kanban_promote", {"task_id": tid, "to": "ready"})
    assert "unsatisfied parent dependencies" in out["error"] and parent in out["error"]
    assert kb.get_task(conn, tid).status == "triage"


def test_promote_to_ready_happy_path(board):
    kb, conn = board
    parent = kb.create_task(conn, title="p", assignee="w")
    tid = kb.create_task(conn, title="c", assignee="w", parents=[parent], triage=True)
    kb.archive_task(conn, parent)
    out = _call("kanban_promote", {"task_id": tid, "to": "ready"})
    assert out == {"ok": True, "task_id": tid, "status": "ready"}


@pytest.mark.parametrize("status", ["todo", "ready", "blocked", "done"])
def test_promote_refuses_non_triage(board, status):
    kb, conn = board
    tid = kb.create_task(conn, title="t", assignee="w")
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
    conn.commit()
    out = _call("kanban_promote", {"task_id": tid})
    assert "only applies to 'triage'" in out["error"]
    assert kb.get_task(conn, tid).status == status


def test_promote_rejects_bad_target_and_unknown_task(board):
    kb, conn = board
    tid = kb.create_task(conn, title="t", assignee="w", triage=True)
    assert "to must be" in _call("kanban_promote", {"task_id": tid, "to": "running"})["error"]
    assert kb.get_task(conn, tid).status == "triage"
    assert "unknown task" in _call("kanban_promote", {"task_id": "t_nope"})["error"]


# --- kanban_unlink ----------------------------------------------------------

def test_unlink_happy_path_regates_child_and_records_actor(board):
    kb, conn = board
    parent = kb.create_task(conn, title="p", assignee="w")
    child = kb.create_task(conn, title="c", assignee="w", parents=[parent])
    assert kb.get_task(conn, child).status == "todo"
    out = _call("kanban_unlink", {"parent_id": parent, "child_id": child})
    assert out == {"ok": True, "parent_id": parent, "child_id": child, "status": "ready"}
    assert kb.parent_ids(conn, child) == []
    [ev] = _events(kb, conn, child, "unlinked")
    assert ev.payload == {"parent": parent, "child": child, "actor": "orchestrator"}


def test_unlink_missing_link_and_args(board):
    kb, conn = board
    a = kb.create_task(conn, title="a", assignee="w")
    b = kb.create_task(conn, title="b", assignee="w")
    assert "no link" in _call("kanban_unlink", {"parent_id": a, "child_id": b})["error"]
    assert "both parent_id and child_id" in _call("kanban_unlink", {"parent_id": a})["error"]


def test_unlink_db_actor_is_optional_for_existing_callers(board):
    kb, conn = board
    parent = kb.create_task(conn, title="p", assignee="w")
    child = kb.create_task(conn, title="c", assignee="w", parents=[parent])
    assert kb.unlink_tasks(conn, parent, child) is True
    [ev] = _events(kb, conn, child, "unlinked")
    assert ev.payload == {"parent": parent, "child": child}
