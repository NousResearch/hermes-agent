"""kanban_archive — first-class orchestrator tool for finishing tasks.

The tool is the model-facing counterpart of ``hermes kanban archive``: a thin
wrapper over the DB primitive ``kanban_db.archive_task``, which keeps the task
row, comments, attachments, events, runs, and dependency links and only moves
the status to ``archived`` (closing the active run so history survives). These
tests pin the tool-surface contract: orchestrator-only visibility, delegated
children refused, deterministic refusal for unknown / already-archived tasks,
active-run closure, child release through the post-archive recompute, board
parity with the internal primitive — all through the real registry with a temp
HERMES_HOME.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Simulate being a dispatcher-spawned worker: HERMES_HOME isolated and
    HERMES_KANBAN_TASK pinned (same pattern as tests/tools/test_kanban_tools.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


@pytest.fixture
def orchestrator_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with an empty board; no HERMES_KANBAN_TASK —
    i.e. an orchestrator profile with the kanban toolset."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _connect():
    from hermes_cli import kanban_db_connect as kbc
    return kbc.connect()


def _create(conn, title="finished card", parents=None):
    from hermes_cli import kanban_db as kb
    return kb.create_task(
        conn, title=title, assignee="worker", parents=parents or ())


def _handle(args):
    from tools import kanban_tools as kt
    return json.loads(kt._handle_archive(args))


def _handle_raw(args):
    from tools import kanban_tools as kt
    return kt._handle_archive(args)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_archive_moves_task_to_archived_and_keeps_id(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create(conn)
    out = _handle({"task_id": tid})
    assert out["ok"] is True, out
    assert out["task_id"] == tid
    assert out["status"] == "archived"
    with _connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.id == tid
    assert task.status == "archived"


def test_archive_requires_task_id(orchestrator_env):
    err = _handle_raw({})
    assert "task_id is required" in err


# ---------------------------------------------------------------------------
# Preservation: row, comments, attachments, events, links
# ---------------------------------------------------------------------------

def test_archive_preserves_row_comments_attachments_events_links(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create(conn)
        _create(conn, title="child", parents=[tid])
        kb.add_comment(conn, tid, "tester", "keep me")
        kb.add_attachment(
            conn, tid, filename="evidence.txt", stored_path="unused/evidence.txt",
            content_type="text/plain", size=8, uploaded_by="tester")
        events_before = len(kb.list_events(conn, tid))
    out = _handle({"task_id": tid})
    assert out["ok"] is True, out
    with _connect() as conn:
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
        attachments = kb.list_attachments(conn, tid)
        events_after = kb.list_events(conn, tid)
    assert task is not None  # the row survives (soft archive, no delete)
    assert task.status == "archived"
    assert [c.body for c in comments] == ["keep me"]
    assert [a.filename for a in attachments] == ["evidence.txt"]
    assert len(events_after) > events_before  # archived event appended, nothing purged
    assert any(e.kind == "archived" for e in events_after)
    # The dependency edge survives too: the child is still linked (asserted
    # behaviourally via the release recompute in
    # test_archive_releases_child_via_recompute).


# ---------------------------------------------------------------------------
# Active run closure / history
# ---------------------------------------------------------------------------

def test_archive_closes_active_run(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create(conn)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    out = _handle({"task_id": tid})
    assert out["ok"] is True, out
    with _connect() as conn:
        run = kb.latest_run(conn, tid)
        task = kb.get_task(conn, tid)
    assert run is not None
    assert run.ended_at is not None
    # The primitive (kanban_db.archive_task) closes a run that was still
    # active with outcome ``reclaimed`` and a fixed summary; parity means
    # the tool preserves that exact history.
    assert run.outcome == "reclaimed"
    assert run.summary == "task archived with run still active"
    assert task.current_run_id is None


# ---------------------------------------------------------------------------
# Child release through the post-archive recompute
# ---------------------------------------------------------------------------

def test_archive_releases_child_via_recompute(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create(conn)
        child = _create(conn, title="child", parents=[tid])
        assert kb.get_task(conn, child).status == "todo"
    out = _handle({"task_id": tid})
    assert out["ok"] is True, out
    with _connect() as conn:
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Deterministic refusal: unknown / already archived
# ---------------------------------------------------------------------------

def test_archive_unknown_task_rejected_without_mutation(orchestrator_env):
    out = _handle_raw({"task_id": "t_does_not_exist"})
    assert "not found" in out


def test_archive_already_archived_rejected(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create(conn)
    assert _handle({"task_id": tid})["ok"] is True
    err = _handle_raw({"task_id": tid})
    assert "already archived" in err
    with _connect() as conn:
        assert kb.get_task(conn, tid).status == "archived"


def test_archive_missing_task_id_reported(orchestrator_env):
    err = _handle_raw({"task_id": "t_gone"})
    assert "not found" in err


# ---------------------------------------------------------------------------
# Parity with the internal primitive
# ---------------------------------------------------------------------------

def test_parity_with_internal_archive_primitive(orchestrator_env):
    """Same board, same end state whichever entry point archived the task."""
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        via_tool = _create(conn, title="via tool")
        via_primitive = _create(conn, title="via primitive")
    assert _handle({"task_id": via_tool})["ok"] is True
    with _connect() as conn:
        assert kb.archive_task(conn, via_primitive) is True
    with _connect() as conn:
        a, b = kb.get_task(conn, via_tool), kb.get_task(conn, via_primitive)
    assert a.status == b.status == "archived"


# ---------------------------------------------------------------------------
# Visibility + refusal: orchestrator-only
# ---------------------------------------------------------------------------

def test_archive_hidden_from_dispatcher_worker_schema(monkeypatch, tmp_path, worker_env):
    """HERMES_KANBAN_TASK pinned → kanban_archive is stripped from the schema
    alongside kanban_list / kanban_unblock / kanban_specify."""
    import tools.kanban_tools  # noqa: F401 — ensure registered
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    schema = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "kanban_archive" not in names
    assert "kanban_unblock" not in names  # sanity: the established gate


def test_archive_visible_to_orchestrator_profile(monkeypatch, tmp_path, orchestrator_env):
    import tools.kanban_tools as kt
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    monkeypatch.setattr(kt, "_profile_has_kanban_toolset", lambda: True)
    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "kanban_archive" in names


def test_worker_call_rejected_even_on_own_board(worker_env):
    """A dispatcher-spawned worker must never archive tasks — the
    orchestrator-only refusal fires before any task matching."""
    from hermes_cli import kanban_db as kb
    tid = worker_env  # the pinned, claimed task itself
    err = _handle_raw({"task_id": tid})
    assert "orchestrator-only" in err
    with _connect() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_delegated_child_mutation_denied(worker_env, monkeypatch):
    """A delegate_task child shares the parent's env, so it must not archive
    even though the inherited HERMES_KANBAN_TASK looks like ownership."""
    import tools.kanban_tools as kt
    from hermes_cli import kanban_db as kb
    monkeypatch.setattr(kt, "_delegation_ctx", lambda predicate, default=False: True)
    tid = worker_env
    err = _handle_raw({"task_id": tid})
    assert "delegate_task child" in err
    with _connect() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_registry_dispatch_routes_to_handler(orchestrator_env):
    """End-to-end through the registry dispatch, not just the bare handler."""
    from hermes_cli import kanban_db as kb
    from model_tools import handle_function_call
    with _connect() as conn:
        tid = _create(conn)
    out = handle_function_call(
        "kanban_archive",
        {"task_id": tid},
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True)
    d = json.loads(out)
    assert d["ok"] is True, out
    with _connect() as conn:
        assert kb.get_task(conn, tid).status == "archived"
