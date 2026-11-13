"""Mark's auto-resume model: a promised stop is silent (no completion, shown in ``list``), and a child that runs out
of attempts or time reports exactly ONE failure."""
import json
import time
from unittest.mock import patch

from tests.tools.test_delegate_auto_resume import _AlwaysLimited, _enable, _repo, _scheduler
from tests.tools.test_delegate_continue import _parent
from tests.tools.test_delegate_resumable import _db_rows
from tests.waiting import wait_until
from tools.process_registry import process_registry


def _events(delegation_id):
    got = []
    while not process_registry.completion_queue.empty():
        evt = process_registry.completion_queue.get_nowait()
        if evt.get("type") == "async_delegation" and evt.get("delegation_id") == delegation_id:
            got.append(evt)
    return got


def _ledger(parent, delegation_id):
    import sqlite3
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test reads a temp-home store
    try:
        return conn.execute("SELECT state, delivery_state FROM async_delegations WHERE delegation_id=?",
                            (delegation_id,)).fetchone()
    finally:
        conn.close()


def _silent_stop(tmp_path, parent):
    from tools.delegate_tool import delegate_task
    from tools.terminal_tool import record_session_cwd
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(_repo(tmp_path, "repo")))
    with patch("run_agent.AIAgent", side_effect=_AlwaysLimited()):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        wait_until(lambda: (_ledger(parent, out["delegation_id"]) or ("running",))[0] != "running",
                   what="the unit to finish")
    return out["subagent_ids"][0], out["delegation_id"]


def test_a_promised_stop_delivers_nothing_and_list_shows_it_waiting(tmp_path):
    from tools.delegate_tool import delegate_task
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    assert _events(unit) == []  # no completion queued
    assert _ledger(parent, unit)[1] == "delivered"  # and none restart recovery could resurrect
    listed = json.loads(delegate_task(action="list", parent_agent=parent))
    row = next(s for s in listed["subagents"] if s["subagent_id"] == sid)
    assert row["status"] == "waiting_to_resume" and "attempt 1 of 2" in row["note"], row


def test_past_the_deadline_the_parent_gets_exactly_one_failure(tmp_path):
    from hermes_constants import get_hermes_home
    from tools.delegate_tool_auto_resume import tick
    _enable()
    parent, dispatched = _parent(), []
    sid, unit = _silent_stop(tmp_path, parent)
    (get_hermes_home() / "config.yaml").write_text(json.dumps(
        {"delegation": {"auto_resume": True, "worktree_isolation": True, "auto_resume_deadline_minutes": 1}}))
    later = time.time() + 120
    for _ in range(3):
        assert tick(_scheduler(parent, dispatched), capacity=3, now=later) == []
    events = _events(unit)
    assert dispatched == [] and len(events) == 1, events
    result = events[0]["results"][0]
    assert events[0]["status"] == "failed" and result["status"] == "failed"
    assert "deadline" in result["error"] and result["subagent_id"] == sid
    state, _pinned, record = _db_rows(parent)[sid]
    assert state == "resumable" and "deadline" in record["auto_gave_up"]  # still continuable by hand
