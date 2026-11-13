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


def _deadline(minutes):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(json.dumps(
        {"delegation": {"auto_resume": True, "worktree_isolation": True, "auto_resume_deadline_minutes": minutes}}))


def test_a_waiting_child_is_not_listed_as_failed_until_it_gives_up(tmp_path):
    from tools.async_delegation import failed_delegations_for_session
    from tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    _sid, unit = _silent_stop(tmp_path, parent)
    assert failed_delegations_for_session(parent_session_id="parent-sess") == []
    _deadline(1)
    tick(_scheduler(parent, []), capacity=3, now=time.time() + 120)
    failed = failed_delegations_for_session(parent_session_id="parent-sess")
    assert [(f["delegation_id"], f["goal"]) for f in failed] == [(unit, "write the parser")], failed


def test_a_crash_during_the_last_attempt_reports_the_give_up_once_not_outcome_unknown(tmp_path):
    import sqlite3
    from tools.async_delegation import recover_abandoned_delegations
    from tools.delegate_tool_auto_resume import MAX_AUTO_ATTEMPTS, tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    dead = 2 ** 22 + 7  # the last automatic run died with its process: both rows still read as running
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test seeds a temp-home store
    with conn:
        conn.execute("UPDATE resumable_subagents SET state='running', owner_pid=?, record_json=json_set(record_json, "
                     "'$.auto_attempts', ?, '$.auto_promised', json('false'), '$.owner_pid', ?) WHERE subagent_id=?",
                     (dead, MAX_AUTO_ATTEMPTS, dead, sid))
        conn.execute("UPDATE async_delegations SET state='running', owner_pid=?, delivery_state='pending' "
                     "WHERE delegation_id=?", (dead, unit))
    conn.close()
    recover_abandoned_delegations()
    assert _ledger(parent, unit) == ("unknown", "delivered")  # no generic "outcome unknown" wake
    for _ in range(3):
        tick(_scheduler(parent, []), capacity=3)
    events = _events(unit)
    assert len(events) == 1 and events[0]["results"][0]["auto_gave_up"], events
    assert f"gave up after {MAX_AUTO_ATTEMPTS} automatic attempts" in events[0]["results"][0]["error"]


def test_a_spawn_with_a_waiting_childs_exact_goal_is_refused_until_it_is_stopped(tmp_path):
    from tests.tools.test_delegate_continue import _ScriptedModel
    from tools.delegate_tool import delegate_task
    _enable()
    parent = _parent()
    sid, _unit = _silent_stop(tmp_path, parent)
    refused = json.loads(delegate_task(goal="  write the parser\n", background=True, parent_agent=parent))
    assert sid in refused["error"] and f"delegate_task(action='stop', subagent_id='{sid}')" in refused["error"]
    assert "waiting to resume" in refused["error"], refused
    assert json.loads(delegate_task(action="stop", subagent_id=sid, parent_agent=parent))["status"] == "dismissed"
    with patch("run_agent.AIAgent", side_effect=_ScriptedModel()):
        again = json.loads(delegate_task(goal="write the parser", background=True, parent_agent=parent))
    assert again.get("status") == "dispatched" and sid not in again["subagent_ids"], again


def test_the_duplicate_guard_is_off_without_auto_resume(tmp_path):
    from hermes_constants import get_hermes_home
    from tests.tools.test_delegate_continue import _ScriptedModel
    from tools.delegate_tool import delegate_task
    _enable()
    parent = _parent()
    _sid, _unit = _silent_stop(tmp_path, parent)
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": {"auto_resume": False}}))
    with patch("run_agent.AIAgent", side_effect=_ScriptedModel()):
        out = json.loads(delegate_task(goal="write the parser", background=True, parent_agent=parent))
    assert out.get("status") == "dispatched", out
