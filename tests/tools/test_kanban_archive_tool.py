"""kanban_archive: the model-callable wrapper over the CLI's archive path (#143206).

Behaviour contract: the tool routes through the exact ``kanban_db.archive_task()``
the CLI ``hermes kanban archive`` uses — no parallel DB logic. These tests pin the
sibling-consistent surface (error shapes, env-default task id, worker task-scope,
board-pinned call) and the orchestrator-only fences staying where they are.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with an initialised board; NOT a dispatcher worker, so
    handlers must be called with an explicit task_id like an orchestrator would."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-archiver")
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kbc


def _status(kbc, tid):
    from hermes_cli import kanban_db as kb
    conn = kbc.connect()
    try:
        task = kb.get_task(conn, tid)
        return task.status if task else None
    finally:
        conn.close()


def _make(kbc, title, **kw):
    from hermes_cli import kanban_db as kb
    conn = kbc.connect()
    try:
        return kb.create_task(conn, title=title, assignee="archiver", **kw)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Handler happy paths — the same transitions the CLI archive accepts
# ---------------------------------------------------------------------------

def test_archive_todo_card(board_env):
    from tools import kanban_tools as kt
    tid = _make(board_env, "todo-card")
    out = json.loads(kt._handle_archive({"task_id": tid}))
    assert out.get("ok") is True, out
    assert out["task_id"] == tid
    assert out["status"] == "archived"
    assert _status(board_env, tid) == "archived"


def test_archive_blocked_card(board_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    tid = _make(board_env, "blocked-card")
    conn = board_env.connect()
    try:
        assert kb.block_task(conn, tid, reason="stuck")
    finally:
        conn.close()
    out = json.loads(kt._handle_archive({"task_id": tid}))
    assert out.get("ok") is True, out
    assert _status(board_env, tid) == "archived"


def test_archive_done_card(board_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    tid = _make(board_env, "done-card")
    conn = board_env.connect()
    try:
        assert kb.complete_task(conn, tid, summary="finished", force=True)
    finally:
        conn.close()
    out = json.loads(kt._handle_archive({"task_id": tid}))
    assert out.get("ok") is True, out
    assert _status(board_env, tid) == "archived"


def test_archive_running_claimed_matches_cli(board_env):
    """CLI-parity contract: ``hermes kanban archive`` on a running card with a live
    claim archives it — ``archive_task()`` ends the run as ``reclaimed`` and (only
    host-local, pid-bearing claims) signals the worker. The tool must not be
    stricter than the CLI; divergence is worse than the gap."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    tid = _make(board_env, "running-card")
    conn = board_env.connect()
    try:
        assert kb.claim_task(conn, tid) is not None
    finally:
        conn.close()
    out = json.loads(kt._handle_archive({"task_id": tid}))
    assert out.get("ok") is True, out
    conn = board_env.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        # Claim cleared exactly like the CLI path's UPDATE does.
        assert task.claim_lock is None and task.worker_pid is None
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "archived" in kinds
    finally:
        conn.close()


def test_archive_parent_frees_child_like_cli(board_env):
    """``recompute_ready`` on archive must promote parent-gated children — the
    same post-archive effect the CLI path has."""
    from tools import kanban_tools as kt
    parent = _make(board_env, "parent")
    child = _make(board_env, "child", parents=[parent])
    assert _status(board_env, child) == "todo"
    out = json.loads(kt._handle_archive({"task_id": parent}))
    assert out.get("ok") is True, out
    assert _status(board_env, child) == "ready"


# ---------------------------------------------------------------------------
# Error shapes — identical to sibling tools
# ---------------------------------------------------------------------------

def test_archive_unknown_id_error_shape(board_env):
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_archive({"task_id": "t_nope"}))
    assert out.get("error"), out
    # Same shape as the siblings' _existing_task rejection: names the id.
    assert "t_nope" in out["error"] and "not found" in out["error"]


def test_archive_twice_is_not_found_error(board_env):
    """CLI semantics: an already-archived id fails the transition CAS; the tool
    surfaces it with the same unknown-id error shape (``archive_task`` returns
    False for both), matching ``hermes kanban archive`` exit behaviour."""
    from tools import kanban_tools as kt
    tid = _make(board_env, "once-only")
    assert json.loads(kt._handle_archive({"task_id": tid})).get("ok") is True
    out = json.loads(kt._handle_archive({"task_id": tid}))
    assert out.get("error") and tid in out["error"]
    assert _status(board_env, tid) == "archived"


def test_archive_bare_call_without_worker_names_kanban_list(board_env):
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_archive({}))
    assert out.get("error"), out
    assert "kanban_list" in out["error"]


# ---------------------------------------------------------------------------
# Worker scoping and routing
# ---------------------------------------------------------------------------

def test_archive_env_default_for_own_task(monkeypatch, board_env):
    """A dispatcher worker may archive its own card bare (HERMES_KANBAN_TASK
    default), like every sibling task-scoped tool."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt
    tid = _make(board_env, "own-card")
    conn = board_env.connect()
    try:
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setattr(kt, "_is_dispatcher_owned_worker", lambda: True)
    out = json.loads(kt._handle_archive({}))
    assert out.get("ok") is True, out
    assert out["status"] == "archived"


def test_worker_cannot_archive_sibling_card(monkeypatch, board_env):
    """Same ownership fence kanban_unblock enforces: a worker scoped to one card
    must not mutate another."""
    from tools import kanban_tools as kt
    mine = _make(board_env, "mine")
    theirs = _make(board_env, "theirs")
    monkeypatch.setenv("HERMES_KANBAN_TASK", mine)
    out = json.loads(kt._handle_archive({"task_id": theirs}))
    assert out.get("error"), out
    assert "refusing" in out["error"] or "scoped" in out["error"]
    assert _status(board_env, theirs) == "ready"


def test_archive_with_explicit_board_arg(tmp_path, monkeypatch):
    """Board pinning rides the shared ``_board`` helper: an explicit slug targets
    that board's DB, not the env-resolved one."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_boards as kbb
    from tools import kanban_tools as kt
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kbb.create_board("scratch-archive", name="Scratch Archive")
    from hermes_cli import kanban_db_connect as kbc2
    conn = kbc2.connect(board="scratch-archive")
    try:
        tid = kb.create_task(conn, title="on-scratch", assignee="archiver")
    finally:
        conn.close()

    out = json.loads(kt._handle_archive({"task_id": tid, "board": "scratch-archive"}))
    assert out.get("ok") is True, out
    conn = kbc2.connect(board="scratch-archive")
    try:
        assert kb.get_task(conn, tid).status == "archived"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Registration and access-model fences
# ---------------------------------------------------------------------------

def test_archive_registered_lifecycle_gate(board_env):
    """Registered in the kanban toolset under the lifecycle gate (visible to any
    profile with the toolset and to workers), NOT the orchestrator-only gate."""
    from tools import kanban_tools as kt
    from tools.registry import registry
    names = [name for name, *_ in kt._TOOLS]
    assert "kanban_archive" in names
    schema = registry.get_schema("kanban_archive")
    props = schema["parameters"]["properties"]
    assert set(props) == {"task_id", "board"}
    assert schema["parameters"]["required"] == ["task_id"]


def test_orchestrator_only_fences_untouched(board_env):
    """kanban_list / kanban_unblock stay hidden from dispatcher workers; archive
    shares the worker-visible gate of kanban_block."""
    from tools import kanban_tools as kt
    assert kt._ORCHESTRATOR_TOOLS == frozenset({"kanban_list", "kanban_unblock"})
    gates = {name: gate for name, gate in (
        ("kanban_archive", kt._check_kanban_mode),
        ("kanban_list", kt._check_kanban_orchestrator_mode),
        ("kanban_unblock", kt._check_kanban_orchestrator_mode))}
    assert gates["kanban_archive"] is kt._check_kanban_mode


def test_archive_visible_to_worker_toolset_schema(monkeypatch, board_env):
    """A dispatcher-spawned worker's resolved kanban toolset includes
    kanban_archive; the orchestrator-only tools remain in the bundle for
    non-worker profiles."""
    from toolsets import resolve_toolset
    names = set(resolve_toolset("kanban", include_registry=False))
    assert {"kanban_archive", "kanban_block", "kanban_complete"} <= names


def test_archive_rejects_unknown_arguments(board_env):
    """The shared unknown-arg pre-check covers the new tool before any write."""
    from tools.registry import registry
    out = json.loads(registry.dispatch("kanban_archive", {"task_id": "t_x", "force": True}))
    assert out.get("error") and "force" in out["error"]


def test_archive_exposed_to_codex_runtime_workers():
    """Codex app-server workers only reach Hermes tools named in EXPOSED_TOOLS;
    archive is a stateless lifecycle tool (reads HERMES_KANBAN_TASK, writes
    kanban.db via archive_task()), so it belongs there like its siblings."""
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

    assert "kanban_archive" in EXPOSED_TOOLS
