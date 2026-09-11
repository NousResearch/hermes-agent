"""kanban_specify — first-class orchestrator tool for the triage column.

The tool is the LLM-free sibling of ``hermes kanban specify``: the calling
orchestrator writes the spec itself and the DB layer
(``kanban_db.specify_triage_task``) keeps the update + promotion atomic and
triage-only. These tests pin the tool-surface contract: orchestrator-only
visibility, triage-only CAS failing closed, byte-exact preservation of
omitted fields (including the ``task_type`` marker on body line 1),
parent gating through the post-write recompute, board isolation, and the
actor/reason audit trail.
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


def _create_triage(conn, title="rough idea", body=None, assignee="worker", parents=None):
    from hermes_cli import kanban_db as kb
    return kb.create_task(
        conn, title=title, body=body, assignee=assignee,
        parents=parents or (), triage=True)


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Dispatcher-spawned worker context: HERMES_KANBAN_TASK pinned to a
    claimed running task (mirrors tests/tools/test_kanban_tools.py)."""
    home = tmp_path / ".hermes-worker"
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


def _handle(args):
    from tools import kanban_tools as kt
    return json.loads(kt._handle_specify(args))


def _handle_raw(args):
    from tools import kanban_tools as kt
    return kt._handle_specify(args)


# ---------------------------------------------------------------------------
# Happy path + audit trail
# ---------------------------------------------------------------------------

def test_specify_updates_fields_and_promotes(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create_triage(conn, body="task_type: research\n\nRough notes.")
    out = _handle({
        "task_id": tid,
        "title": "Refined: rough idea",
        "body": "task_type: research\n\n**Goal**\nDo the thing.",
        "assignee": "worker-b",
        "reason": "dispatcher flagged this card as too thin to dispatch",
        # Unknown author-ish args must be ignored (anti-forgery, #19713).
        "author": "hermes-system",
    })
    assert out["ok"] is True, out
    assert out["status"] == "ready"  # no parents → recompute promotes past todo
    assert out["title"] == "Refined: rough idea"
    assert out["assignee"] == "worker-b"

    with _connect() as conn:
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert task.status == "ready"
    assert task.body == "task_type: research\n\n**Goal**\nDo the thing."
    assert len(comments) == 1
    assert comments[0].author == "test-orchestrator"
    assert "Specified" in comments[0].body
    assert "Reason: dispatcher flagged this card as too thin" in comments[0].body
    assert len(events) == 1
    assert events[0].payload["changed_fields"] == ["title", "body", "assignee"]
    assert events[0].payload["reason"] == "dispatcher flagged this card as too thin to dispatch"


def test_specify_preserves_omitted_fields_exactly(orchestrator_env):
    """Omitted title/body/assignee stay byte-exact — the task_type marker on
    body line 1 is what keeps the dispatcher's task_type gate working."""
    from hermes_cli import kanban_db as kb
    original_body = "task_type: code\n\nline A\nline B — строка."
    with _connect() as conn:
        tid = _create_triage(conn, title="keep-title", body=original_body,
                             assignee="worker-a")
    out = _handle({"task_id": tid, "title": "Only retitled"})
    assert out["ok"] is True, out

    with _connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
        comments = kb.list_comments(conn, tid)
    assert task.body == original_body  # byte-exact, no normalization
    assert task.assignee == "worker-a"
    assert events[0].payload["changed_fields"] == ["title"]
    assert comments[0].body == "Specified — updated title and promoted to todo."


# ---------------------------------------------------------------------------
# Validation: fail closed
# ---------------------------------------------------------------------------

def test_specify_requires_task_id(orchestrator_env):
    assert "task_id is required" in _handle_raw({})


def test_specify_requires_at_least_one_field(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create_triage(conn)
    err = _handle_raw({"task_id": tid})
    assert "at least one of title, body, or assignee" in err
    with _connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_specify_rejects_blank_title(orchestrator_env):
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create_triage(conn)
    err = _handle_raw({"task_id": tid, "title": "   ", "body": "ok"})
    assert "title cannot be blank" in err
    with _connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_specify_unknown_task_rejected(orchestrator_env):
    assert "not found" in _handle_raw({"task_id": "t_doesnotexist", "title": "x"})


def test_specify_non_triage_task_fails_closed(orchestrator_env):
    """A todo/ready/... task must not be touched: specify is a triage-column
    promotion, not a general-purpose edit."""
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create_triage(conn, title="already fine")
    out = _handle({"task_id": tid, "title": "Nope"})  # promotes it first
    assert out["ok"] is True
    err = _handle_raw({"task_id": tid, "title": "Nope again"})
    assert "not in triage" in err
    with _connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.title == "Nope"  # second call changed nothing


def test_specify_stale_race_fails_closed(orchestrator_env, monkeypatch):
    """Task leaves triage between the existence check and the write → the
    tool reports failure instead of silently re-writing fields."""
    from hermes_cli import kanban_db as kb

    def fake_specify(conn, task_id, **kw):
        return False  # rowcount 0: someone promoted/archived it first

    monkeypatch.setattr(kb, "specify_triage_task", fake_specify)
    with _connect() as conn:
        tid = _create_triage(conn)
    err = _handle_raw({"task_id": tid, "title": "racy"})
    assert "not in triage" in err
    with _connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert task.title == "rough idea"


# ---------------------------------------------------------------------------
# Parent gating + board isolation
# ---------------------------------------------------------------------------

def test_specify_with_open_parents_lands_todo_then_releases(orchestrator_env):
    """Open parents → todo (dispatcher-ready, but gated); after the last
    parent is done, the normal promote path moves the child to ready."""
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _create_triage(conn, parents=[parent])
    out = _handle({"task_id": child, "title": "child spec"})
    assert out["ok"] is True, out
    assert out["status"] == "todo"

    with _connect() as conn:
        assert kb.complete_task(conn, parent, summary="parent done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_specify_board_isolation(orchestrator_env):
    """board=<slug> routes the whole call; other boards are untouched."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    with _connect() as conn:
        default_tid = _create_triage(conn, title="default-card")
    with kbc.connect(board="alt") as conn:
        alt_tid = _create_triage(conn, title="alt-card")

    out = _handle({"task_id": alt_tid, "title": "alt spec", "board": "alt"})
    assert out["ok"] is True, out

    with kbc.connect(board="alt") as conn:
        assert kb.get_task(conn, alt_tid).title == "alt spec"
        assert kb.get_task(conn, alt_tid).status == "ready"
    with _connect() as conn:
        default_task = kb.get_task(conn, default_tid)
    assert default_task.status == "triage"
    assert default_task.title == "default-card"


# ---------------------------------------------------------------------------
# Visibility: orchestrator-only
# ---------------------------------------------------------------------------

def test_specify_hidden_from_dispatcher_worker_schema(monkeypatch, tmp_path, worker_env):
    """HERMES_KANBAN_TASK pinned → kanban_specify is stripped from the schema
    alongside kanban_list / kanban_unblock."""
    import tools.kanban_tools  # noqa: F401 — ensure registered
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    schema = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "kanban_specify" not in names
    assert "kanban_unblock" not in names  # sanity: the established gate


def test_specify_visible_to_orchestrator_profile(monkeypatch, tmp_path, orchestrator_env):
    import tools.kanban_tools as kt
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    monkeypatch.setattr(kt, "_profile_has_kanban_toolset", lambda: True)
    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "kanban_specify" in names


def test_worker_call_rejected_even_on_own_board(worker_env):
    """A dispatcher-spawned worker must never mutate the board via specify —
    the orchestrator-only refusal fires before any task matching."""
    from hermes_cli import kanban_db as kb
    with _connect() as conn:
        tid = _create_triage(conn)
    err = _handle_raw({"task_id": tid, "title": "worker was here"})
    assert "orchestrator-only" in err
    with _connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert task.title == "rough idea"


def test_registry_dispatch_routes_to_handler(orchestrator_env):
    """End-to-end through the registry dispatch, not just the bare handler."""
    from hermes_cli import kanban_db as kb
    from model_tools import handle_function_call
    with _connect() as conn:
        tid = _create_triage(conn)
    out = handle_function_call(
        "kanban_specify",
        {"task_id": tid, "title": "registry routed", "reason": "e2e"},
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True)
    d = json.loads(out)
    assert d["ok"] is True, out
    with _connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.title == "registry routed"
    assert task.status == "ready"
