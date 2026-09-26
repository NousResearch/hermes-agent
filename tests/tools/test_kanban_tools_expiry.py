"""Wake-Guard: kanban tool calls must not act on (or for) an expired scoping
(t_bdd69e28, port of PR #91's tests/tools/test_kanban_tools_expiry.py, adapted
to the upstream line: no remediation-target reconcile allowance and no
auto-progress bridge exist upstream, so those two groups are absent here; the
ownership veto raises ``_Reject`` instead of returning an error string).
"""

import json

import pytest

from tools import kanban_tools as kt
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _fresh_board(tmp_path, monkeypatch, name="tools-expiry.db"):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / name))
    kb.init_db()
    return kbc.connect()


def _env_task(monkeypatch, tid, run_id=None):
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    if run_id is not None:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _latest_run_id(conn, tid):
    row = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    return row["id"] if row is not None else None


# --- T3: task resolution / attribution -----------------------------------------


def test_default_task_id_none_when_scoped_task_terminal(tmp_path, monkeypatch):
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="dead card", assignee="worker")
        kb.complete_task(conn, tid, summary="finished")
    finally:
        conn.close()
    _env_task(monkeypatch, tid)

    assert kt._default_task_id(None) is None


def test_default_task_id_none_when_scoping_run_ended(tmp_path, monkeypatch):
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="ended run", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = _latest_run_id(conn, tid)
        kb.complete_task(conn, tid, summary="run closed", expected_run_id=run_id)
    finally:
        conn.close()
    _env_task(monkeypatch, tid, run_id)

    assert kt._default_task_id(None) is None


def test_default_task_id_survives_unknown_freshness(tmp_path, monkeypatch):
    """A missing task row is UNKNOWN freshness — fail open to the legacy
    resolution instead of guessing."""
    _fresh_board(tmp_path, monkeypatch)
    _env_task(monkeypatch, "t_does_not_exist")

    assert kt._default_task_id(None) == "t_does_not_exist"


def test_create_without_args_not_attributed_to_dead_card(tmp_path, monkeypatch):
    """kanban_create under an expired scoping must not stamp the dead card as
    creator or project provenance."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="dead creator", assignee="worker")
        kb.complete_task(conn, tid, summary="finished")
    finally:
        conn.close()
    _env_task(monkeypatch, tid)

    result = kt._handle_create({"title": "child card", "assignee": "worker"}, **{})

    data = json.loads(result)
    assert data["ok"] is True
    new_tid = data["task_id"]
    conn = kbc.connect()
    try:
        created_event = next(
            (e for e in kb.list_events(conn, new_tid) if e.kind == "created"), None)
        payload = dict(created_event.payload or {}) if created_event is not None else {}
        assert created_event is not None
        assert payload.get("creator_task_id") != tid
        assert kb.parent_ids(conn, new_tid) == []
    finally:
        conn.close()


# --- T4a: ownership veto messages ----------------------------------------------


def test_foreign_mutation_denied_with_expired_scoping_message(tmp_path, monkeypatch):
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="dead card", assignee="worker")
        kb.complete_task(conn, tid, summary="finished")
    finally:
        conn.close()
    _env_task(monkeypatch, tid)

    with pytest.raises(kt._Reject) as excinfo:
        kt._enforce_worker_task_ownership("t_other_task")
    message = str(excinfo.value)
    assert "EXPIRED" in message
    assert tid in message
    assert "t_other_task" in message


def test_foreign_mutation_denied_with_active_scoping_message(tmp_path, monkeypatch):
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="live card", assignee="worker")
    finally:
        conn.close()
    _env_task(monkeypatch, tid)

    with pytest.raises(kt._Reject) as excinfo:
        kt._enforce_worker_task_ownership("t_other_task")
    message = str(excinfo.value)
    assert "worker is scoped to task" in message
    assert "EXPIRED" not in message


def test_foreign_mutation_denied_when_freshness_unknown(tmp_path, monkeypatch):
    """Unknown freshness keeps the STANDARD veto (fail-open on freshness, the
    foreign-task veto itself never loosens)."""
    _fresh_board(tmp_path, monkeypatch)
    _env_task(monkeypatch, "t_does_not_exist")

    with pytest.raises(kt._Reject) as excinfo:
        kt._enforce_worker_task_ownership("t_other_task")
    message = str(excinfo.value)
    assert "worker is scoped to task" in message
    assert "EXPIRED" not in message


# --- T5: heartbeat terminal refusal --------------------------------------------


def test_heartbeat_refuses_terminal_task(tmp_path, monkeypatch):
    """The kanban_heartbeat tool surface refuses a dead card. The @_kanban_handler
    wrapper converts ``_Reject`` into its plain message string, so the refusal is
    asserted on the returned tool error text."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="dead card", assignee="worker")
        kb.complete_task(conn, tid, summary="finished")
    finally:
        conn.close()
    _env_task(monkeypatch, tid)

    result = kt._handle_heartbeat({"task_id": tid, "note": "still alive?"}, **{})

    assert "terminal" in result
    assert "kanban_heartbeat" in result


# --- T7: auto-heartbeat bridge --------------------------------------------------


def test_auto_heartbeat_bridge_skips_expired_scoping(tmp_path, monkeypatch):
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="dead card", assignee="worker")
        kb.complete_task(conn, tid, summary="finished")
        events_before = len(kb.list_events(conn, tid))
    finally:
        conn.close()
    _env_task(monkeypatch, tid)
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)

    assert kt.heartbeat_current_worker_from_env() is False

    conn = kbc.connect()
    try:
        assert len(kb.list_events(conn, tid)) == events_before
    finally:
        conn.close()


def test_auto_bridges_still_fire_for_active_scoping(tmp_path, monkeypatch):
    """Negative control: a live scoped worker still gets its claim extension."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="live card", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    _env_task(monkeypatch, tid)
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)

    assert kt.heartbeat_current_worker_from_env() is True
