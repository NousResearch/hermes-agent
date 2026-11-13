"""Mark's auto-resume model: a promised stop is silent (no completion, shown in ``list``), and a child that runs out
of attempts or time reports exactly ONE failure."""
import contextlib
import json
import os
import sqlite3
import time

import pytest
from unittest.mock import patch

from tests.tools.test_delegate_auto_resume import _AlwaysLimited, _enable, _repo, _scheduler
from tests.tools.test_delegate_continue import _parent, _ScriptedModel
from tests.tools.test_delegate_resumable import _db_rows
from tests.waiting import wait_until
from hermes_agent.tools.process_registry import process_registry


def _events(delegation_id):
    """The queued completions of unit *delegation_id*: its own and each child's give-up failure (own row)."""
    got = []
    while not process_registry.completion_queue.empty():
        evt = process_registry.completion_queue.get_nowait()
        if evt.get("type") == "async_delegation" and str(evt.get("delegation_id") or "").split(":auto:")[0] == delegation_id:
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
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.terminal_tool import record_session_cwd
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(_repo(tmp_path, "repo")))
    with patch("run_agent.AIAgent", side_effect=_AlwaysLimited()):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        wait_until(lambda: (_ledger(parent, out["delegation_id"]) or ("running",))[0] != "running",
                   what="the unit to finish")
    return out["subagent_ids"][0], out["delegation_id"]


def test_a_promised_stop_delivers_nothing_and_list_shows_it_waiting(tmp_path):
    from hermes_agent.tools.delegate_tool import delegate_task
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
    from hermes_agent.tools.delegate_tool_auto_resume import tick
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
    from hermes_agent.tools.async_delegation import failed_delegations_for_session
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    _sid, unit = _silent_stop(tmp_path, parent)
    assert failed_delegations_for_session(parent_session_id="parent-sess") == []
    _deadline(1)
    tick(_scheduler(parent, []), capacity=3, now=time.time() + 120)
    failed = failed_delegations_for_session(parent_session_id="parent-sess")
    assert [(f["delegation_id"], f["goal"]) for f in failed] == [(f"{unit}:auto:{_sid}", "write the parser")], failed


def test_a_crash_during_the_last_attempt_reports_the_give_up_once_not_outcome_unknown(tmp_path):
    import sqlite3
    from hermes_agent.tools.async_delegation import recover_abandoned_delegations
    from hermes_agent.tools.delegate_tool_auto_resume import MAX_AUTO_ATTEMPTS, tick
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
    from hermes_agent.tools.delegate_tool import delegate_task
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
    from hermes_agent.tools.delegate_tool import delegate_task
    _enable()
    parent = _parent()
    _sid, _unit = _silent_stop(tmp_path, parent)
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": {"auto_resume": False}}))
    with patch("run_agent.AIAgent", side_effect=_ScriptedModel()):
        out = json.loads(delegate_task(goal="write the parser", background=True, parent_agent=parent))
    assert out.get("status") == "dispatched", out


# ── delivery: each child reports once, on its own row, whatever the pool or a crash does ──────────────────────────
def _deliver_all(events):
    from hermes_agent.tools.async_delegation import claim_event_delivery, complete_completion_delivery
    delivered = []
    for event in events:
        token = claim_event_delivery(event, "test")
        if token:
            delivered += [r["subagent_id"] for r in event["results"] if r.get("status") != "waiting_to_resume"]
            assert complete_completion_delivery(event["delegation_id"], token)
    return delivered


class _LimitedWhenSlow(_ScriptedModel):
    """Only a child whose task says "slow" hits the limit at once (reset already passed); the other completes."""

    def __call__(self, **kw):
        child = super().__call__(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, conversation_history=None, **rest):
            result = ok(user_message, conversation_history=conversation_history, **rest)
            if "slow" not in user_message:
                # Finishes only after its limited sibling was recorded, so that one's early-notice check runs while
                # this child is still running (dispatch records a finished child right before that check).
                from hermes_agent.tools.async_delegation import _records, _records_lock

                def sibling_recorded():
                    with _records_lock:
                        return any(r.get("_finished_children") for r in _records.values())
                wait_until(sibling_recorded, what="the limited sibling to be recorded")
                return result
            return {**result, "completed": False, "failed": True, "error": "limit", "failure_reason": "rate_limit",
                    "failure_resets_at": time.time() - 1}
        child.run_conversation.side_effect = run_conversation
        return child


def test_a_mixed_batch_reports_each_child_once_and_no_early_failure_for_the_waiting_one(tmp_path):
    """One child completes, one is limit-stopped and promised a resume: no early per-task failure notice for the
    waiting one; the batch's own completion carries the finished child, and the give-up its own row and ack."""
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable(independent_completions=False, worktree_isolation=False, max_concurrent_children=3)
    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=_LimitedWhenSlow()):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}, {"goal": "slow: write the tests"}],
                                       background=True, parent_agent=parent))
        wait_until(lambda: (_ledger(parent, out["delegation_id"]) or ("running",))[0] != "running", what="the unit")
    first = _events(out["delegation_id"])
    assert [e.get("task_failure_notice") for e in first] == [None], first  # the batch result only, no early notice
    _deadline(1)
    for _ in range(3):
        tick(lambda *args: None, capacity=3, now=time.time() + 120)
    later = _events(out["delegation_id"])
    assert len(later) == 1 and later[0]["results"][0]["auto_gave_up"], later
    assert sorted(_deliver_all(first + later)) == sorted(out["subagent_ids"])


def test_a_full_pool_still_settles_a_deadline(tmp_path):
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    _deadline(1)
    tick(lambda *args: None, capacity=0, now=time.time() + 120)
    events = _events(unit)
    assert len(events) == 1 and events[0]["results"][0]["subagent_id"] == sid, events


def test_a_give_up_interrupted_before_its_commit_reports_once_on_a_later_tick(tmp_path):
    """The gave-up mark and the replayable failure commit together: a failure between them loses neither, and the
    retry neither repeats nor drops the one report (its row stays pending for restart replay until acked)."""
    import contextlib
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    _deadline(1)
    with patch("hermes_agent.tools.async_delegation_recovery_hints.transcript_tails", side_effect=RuntimeError("disk full")), \
            contextlib.suppress(RuntimeError):
        tick(lambda *args: None, capacity=3, now=time.time() + 120)
    assert _events(unit) == []
    for _ in range(3):
        tick(lambda *args: None, capacity=3, now=time.time() + 120)
    assert len(_events(unit)) == 1
    assert _ledger(parent, f"{unit}:auto:{sid}") == ("failed", "pending")


# ── a promised child is owed its one report whatever config, TTL or retention caps do ────────────────────────────
def _sql(parent, sql, params=()):
    import sqlite3
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test seeds a temp-home store
    try:
        with conn:
            return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_turning_auto_resume_off_still_reports_a_promised_child_once_and_the_guard_follows_its_deadline(tmp_path):
    from hermes_constants import get_hermes_home
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    _sql(parent, "UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_first_stopped_at', ?) "
                 "WHERE subagent_id=?", (time.time() - 7 * 3600, sid))
    with patch("run_agent.AIAgent", side_effect=_ScriptedModel()):  # past its deadline, before any tick settled it
        again = json.loads(delegate_task(goal="write the parser", background=True, parent_agent=parent))
    assert again.get("status") == "dispatched", again
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": {"auto_resume": False}}))
    dispatched = []
    for _ in range(3):
        assert tick(_scheduler(parent, dispatched), capacity=3) == []
    events = _events(unit)
    assert dispatched == [] and len(events) == 1 and events[0]["results"][0]["subagent_id"] == sid, events
    assert "turned off" in events[0]["results"][0]["error"]


def test_a_stop_while_auto_resume_is_off_still_cancels_a_promised_child(tmp_path):
    from hermes_constants import get_hermes_home
    from hermes_agent.tools.delegate_tool_auto_resume import supersede_for_session, tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    cfg = get_hermes_home() / "config.yaml"
    on = cfg.read_text()
    cfg.write_text(json.dumps({"delegation": {"auto_resume": False}}))
    assert supersede_for_session(parent_session_id=parent.session_id, reason="/stop") == 1
    cfg.write_text(on)
    dispatched = []
    with patch("run_agent.AIAgent", side_effect=_AlwaysLimited()):
        assert tick(_scheduler(parent, dispatched), capacity=3) == []
    assert dispatched == [] and _db_rows(parent)[sid][0] == "superseded" and _events(unit) == []


def test_retention_limits_never_eat_a_waiting_childs_report(tmp_path):
    from hermes_agent.tools.async_delegation import _MAX_RETAINED_COMPLETED, _prune_durable_records, _transaction
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    from pathlib import Path
    from hermes_agent.tools.delegate_tool_resumable import _apply_limits, _write
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    now = time.time()
    with _transaction() as conn:  # rows the count cap may not delete must not count against it either
        for i in range(_MAX_RETAINED_COMPLETED):
            conn.execute("INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, "
                         "updated_at, delivery_state) VALUES (?, '', 'error', ?, ?, 'delivered')", (f"kept-{i}", now, now))
            conn.execute("INSERT INTO resumable_subagents (subagent_id, state, record_json, created_at, updated_at) "
                         "VALUES (?, 'resumable', json(?), ?, ?)",
                         (f"sa-{i}", json.dumps({"unit_id": f"kept-{i}", "auto_promised": True}), now, now))
        conn.execute("INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, updated_at, "
                     "delivery_state, event_json) VALUES ('fresh-pending', '', 'completed', ?, ?, 'pending', '{}')",
                     (now, now))
    _prune_durable_records()
    assert _ledger(parent, "fresh-pending") is not None and _ledger(parent, unit) is not None
    _sql(parent, "UPDATE resumable_subagents SET expires_at=? WHERE subagent_id=?", (time.time() - 1, sid))
    _write(Path(parent._session_db.db_path), _apply_limits)  # TTL passed: the record moves to expired
    assert _db_rows(parent)[sid][0] == "expired"
    for _ in range(3):
        tick(lambda *args: None, capacity=3, now=time.time() + 60)
    events = _events(unit)
    assert len(events) == 1 and "expired" in events[0]["results"][0]["error"], events


def _expire(parent, sid):
    from pathlib import Path
    from hermes_agent.tools.delegate_tool_resumable import _apply_limits, _write
    _sql(parent, "UPDATE resumable_subagents SET expires_at=? WHERE subagent_id=?", (time.time() - 1, sid))
    _write(Path(parent._session_db.db_path), _apply_limits)


def _cancelled_then_expired(tmp_path, cancel, *, expire_first):
    from hermes_constants import get_hermes_home
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    if expire_first:
        _expire(parent, sid)
    cancel(parent, sid)
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": {"auto_resume": False}}))
    if not expire_first:
        _expire(parent, sid)
    for _ in range(3):
        tick(_scheduler(parent, []), capacity=3)
    return _events(unit)


def _slash_stop(parent, _sid):
    from hermes_agent.tools.delegate_tool_auto_resume import supersede_for_session
    assert supersede_for_session(parent_session_id=parent.session_id, reason="/stop") == 1


def _stop(parent, sid):
    from hermes_agent.tools.delegate_tool import delegate_task
    out = json.loads(delegate_task(action="stop", subagent_id=sid, parent_agent=parent))
    assert out.get("status") == "dismissed", out


def _stop_claimed(parent, sid):  # the parent's own continue holds it: not a dismiss, but the stop is kept
    from hermes_agent.tools.delegate_tool import delegate_task
    out = json.loads(delegate_task(action="stop", subagent_id=sid, parent_agent=parent))
    assert out.get("status") == "stop_recorded", out


@pytest.mark.parametrize("expire_first", [False, True], ids=["cancel-then-ttl", "ttl-then-cancel"])
@pytest.mark.parametrize("cancel", [_slash_stop, _stop], ids=["slash-stop", "action-stop"])
def test_a_child_its_parent_cancelled_never_reports_even_when_its_record_expires(tmp_path, cancel, expire_first):
    assert _cancelled_then_expired(tmp_path, cancel, expire_first=expire_first) == []


def _home():
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _turn_off():
    (_home() / "config.yaml").write_text(json.dumps({"delegation": {"auto_resume": False}}))


@contextlib.contextmanager
def _no_commits(path):
    conn = sqlite3.connect(path)  # health: allow HX015 -- test observer of other connections' commits
    try:
        version = conn.execute("PRAGMA data_version").fetchone()[0]
        stamps = {p: os.stat(p).st_mtime_ns for p in (path, f"{path}-wal") if os.path.exists(p)}
        yield
        assert conn.execute("PRAGMA data_version").fetchone()[0] == version, "a tick committed to state.db"
        assert {p: os.stat(p).st_mtime_ns for p in stamps} == stamps
    finally:
        conn.close()


def test_a_disabled_tick_with_nothing_owed_never_writes_the_store(tmp_path):
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    _stop(parent, sid)  # its debt is cleared: nothing owed
    _turn_off()
    with _no_commits(parent._session_db.db_path):
        for _ in range(5):
            assert tick(_scheduler(parent, []), capacity=3) == []
    assert _events(unit) == []


def _manual_claim_then(tmp_path, cancel):
    """The parent's own continue claims a promised child and *cancel* (or nothing) lands while it holds the claim;
    a disabled tick runs inside that window, then the continue is refused before anything ran and ticks follow
    (enabled ones only after a cancel: a cancelled child must not be resumed)."""
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    from hermes_agent.tools.delegate_tool_resumable import _db_path_for, _write, claim, release
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    record, err = claim(parent, sid)
    assert record and not err
    if cancel:
        cancel(parent, sid)
    else:
        _turn_off()
    dispatched = []
    tick(_scheduler(parent, dispatched), capacity=3)
    release(parent, sid, record)
    _write(_db_path_for(parent), lambda c: c.execute("UPDATE resumable_subagents SET record_json=json_set("
                                                     "record_json, '$.next_attempt_at', 0) WHERE subagent_id=?", (sid,)))
    with patch("run_agent.AIAgent", side_effect=_AlwaysLimited()):
        tick(_scheduler(parent, dispatched), capacity=3)
    _turn_off()
    for _ in range(3):
        tick(_scheduler(parent, dispatched), capacity=3)
    return dispatched, _db_rows(parent)[sid], _events(unit)


def test_a_refused_manual_continue_hands_the_promise_back_and_it_reports_once(tmp_path):
    dispatched, _row, events = _manual_claim_then(tmp_path, None)
    assert dispatched == [] and len(events) == 1, events


@pytest.mark.parametrize("cancel", [_slash_stop, _stop_claimed], ids=["slash-stop", "action-stop"])
def test_a_stop_during_the_parents_own_continue_claim_is_never_lost(tmp_path, cancel):
    dispatched, (state, _pinned, rec), events = _manual_claim_then(tmp_path, cancel)
    assert dispatched == [] and events == [], (dispatched, events)
    assert state in ("superseded", "stopped") and rec["auto_promised"] is False and "auto_cancel" not in rec


def test_a_stop_racing_the_refused_continues_hand_back_is_never_lost(tmp_path):
    """The refused continue hands the row back while ``action='stop'`` is deciding: the read and the cancel are one
    transaction, so the stop lands on whichever row it sees and the child is never resumed."""
    import hermes_agent.tools.delegate_tool_resumable as res
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    record, err = res.claim(parent, sid)
    assert record and not err
    real, calls = res._write, []

    def hand_back_first(path, fn):  # the refused continue's hand-back reaches the store just before the stop does
        calls.append(1)
        if len(calls) == 1:
            res.release(parent, sid, record)
        return real(path, fn)
    with patch.object(res, "_write", side_effect=hand_back_first):
        out = json.loads(delegate_task(action="stop", subagent_id=sid, parent_agent=parent))
    assert out.get("status") == "dismissed", out
    res._write(res._db_path_for(parent), lambda c: c.execute("UPDATE resumable_subagents SET record_json=json_set("
                                                             "record_json, '$.next_attempt_at', 0) WHERE subagent_id=?",
                                                             (sid,)))
    dispatched = []
    with patch("run_agent.AIAgent", side_effect=_AlwaysLimited()):
        tick(_scheduler(parent, dispatched), capacity=3)
    assert dispatched == [] and _events(unit) == []


# ── an automatic attempt that never started is not an outcome; a cancel the tick did not see still wins ──────────
def _cancel_after_confirm(parent, sid):
    import hermes_agent.tools.delegate_tool_auto_resume as ar
    real = ar.confirm_start

    def confirm_then_stop(path, subagent_id):
        out = real(path, subagent_id)
        _stop(parent, sid)
        return out
    return patch.object(ar, "confirm_start", side_effect=confirm_then_stop)


def _first_run_record_busy(_parent, _sid):
    import hermes_agent.tools.delegate_tool_resumable as dr
    from tests.tools.test_delegate_auto_resume_parent_wins import _busy_run_record
    return patch.object(dr, "_write", _busy_run_record(dr))


@pytest.mark.parametrize("unstarted", [_cancel_after_confirm, _first_run_record_busy])
def test_an_automatic_attempt_that_never_started_reports_nothing(tmp_path, unstarted):
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    from tests.tools.test_delegate_auto_resume import _settled
    _enable()
    parent, dispatched = _parent(), []
    sid, _unit = _silent_stop(tmp_path, parent)
    with unstarted(parent, sid), patch("run_agent.AIAgent", side_effect=_ScriptedModel()):
        tick(_scheduler(parent, dispatched), capacity=3)
        _settled(dispatched[0])
    assert _events(dispatched[0]) == [] and _ledger(parent, dispatched[0])[1] == "delivered"
    with patch("run_agent.AIAgent", side_effect=_ScriptedModel()):
        tick(_scheduler(parent, dispatched), capacity=3)
        later = [_settled(d) for d in dispatched[1:]]
    if unstarted is _cancel_after_confirm:  # cancelled: nothing, ever
        assert later == [] and _db_rows(parent)[sid][0] == "stopped"
    else:  # still owed: the next attempt is its one report
        assert len(later) == 1 and later[0]["results"][0]["status"] == "completed", later


def test_a_cancel_after_the_tick_read_an_expired_child_still_wins(tmp_path):
    import hermes_agent.tools.delegate_tool_auto_resume as ar
    _enable()
    parent = _parent()
    sid, unit = _silent_stop(tmp_path, parent)
    _expire(parent, sid)
    real = ar._give_up

    def stop_then_give_up(record, reason):
        _stop(parent, sid)
        return real(record, reason)
    with patch.object(ar, "_give_up", side_effect=stop_then_give_up):
        ar.tick(lambda *a: None, capacity=3)
    assert _events(unit) == [] and _ledger(parent, f"{unit}:auto:{sid}") is None


def test_an_expired_record_without_auto_resume_fields_is_still_deleted():
    from pathlib import Path
    from hermes_agent.tools.delegate_tool_resumable import _apply_limits, _write
    parent = _parent()
    _sql(parent, "INSERT INTO resumable_subagents (subagent_id,state,pinned,record_json,created_at,updated_at,"
                 "expires_at) VALUES ('legacy','expired',0,'{}',1,1,1)")
    _write(Path(parent._session_db.db_path), _apply_limits)
    assert _sql(parent, "SELECT subagent_id FROM resumable_subagents WHERE subagent_id='legacy'") == []


def test_a_failed_start_whose_hand_back_also_failed_still_reports_once_after_the_store_recovers(tmp_path):
    import hermes_agent.tools.delegate_tool_resumable as dr
    from tests.tools.test_delegate_auto_resume import _settled
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    _enable()
    parent, dispatched = _parent(), []
    sid, unit = _silent_stop(tmp_path, parent)
    real = dr._write

    def busy(path, fn):  # the run record and the claim's hand-back both hit the busy timeout
        if fn.__qualname__ == "record_start.<locals>._do" or fn.__qualname__.startswith(
                ("release_unstarted.", "_hand_back_pending.")):
            return None
        return real(path, fn)
    with patch.object(dr, "_write", busy), patch("run_agent.AIAgent", side_effect=_ScriptedModel()):
        tick(_scheduler(parent, dispatched), capacity=3)
        _settled(dispatched[0])
    assert _events(dispatched[0]) == []  # never ran: silent, still owed
    with patch("run_agent.AIAgent", side_effect=_ScriptedModel()):  # the store is healthy again, deadline passed
        for _ in range(3):
            tick(_scheduler(parent, dispatched), capacity=3, now=time.time() + 30 * 86400)
    events = _events(unit) + [e for d in dispatched for e in _events(d)]
    assert len(events) == 1 and events[0]["results"][0]["status"] == "failed", (events, _db_rows(parent)[sid])
