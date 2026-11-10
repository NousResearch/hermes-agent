"""A standalone ``hermes serve`` forwards a prompt to the gateway when that gateway runs a background
subagent the conversation dispatched.

The child and the only registry that can list, steer or stop it live in the gateway's memory. With
the gateway's turn over, the canonical ledger (admissions, worker executions) held nothing, so the
legacy backend ran the next turn itself: ``delegate_task list`` answered "No live subagents right now"
while the child worked on, and the agent re-dispatched it. The durable ``async_delegations`` row now
routes the turn to the owner through the RT-37 forward (``owner_forward``).

Real: the SessionDB and its ``async_delegations`` row, a live owner process (liveness and start time
checked for real), the legacy ``prompt.submit`` and its ledger gate, and the owner's
``SessionAuthority`` + ``AuthorityConnection``. Replaced: the gateway identity lookup
(``live_gateway_pid_for_home``) and the owner's discovery and socket dial, as in
``test_legacy_submit_forwards_to_owner``.
"""
import subprocess
import sys
import time

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions
from tests.tui_gateway.test_legacy_submit_forwards_to_owner import _legacy, _owner
from tui_gateway import server


@pytest.fixture
def owner_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield proc
    proc.kill()
    proc.wait()


def _start_time(pid):
    from gateway.status import get_process_start_time
    return get_process_start_time(pid)


def _running_delegation(db_path, parent, pid, started, *, state="running"):
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO async_delegations (delegation_id, origin_session, parent_session_id, state,
                   dispatched_at, updated_at, owner_pid, owner_started_at, task_json)
               VALUES ('deleg_gw', 'agent:main:local:dm:x', ?, ?, ?, ?, ?, ?, '{"goal": "long build"}')""",
            (parent, state, time.time() - 60, time.time(), pid, started))
        conn.commit()
    finally:
        conn.close()


def _gateway_is(monkeypatch, pid):
    import gateway.status
    monkeypatch.setattr(gateway.status, "live_gateway_pid_for_home", lambda _home: pid)


def _submit(sid, text="is the build done?"):
    return server.handle_request({"id": "p", "method": "prompt.submit", "params": {"session_id": sid, "text": text}})


def test_Given_the_gateway_runs_this_chats_background_subagent_When_serve_gets_a_prompt_Then_it_runs_in_the_gateway(
        monkeypatch, tmp_path, owner_process):
    # Given
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, owner_process.pid, _start_time(owner_process.pid))
        _gateway_is(monkeypatch, owner_process.pid)
        _owner(monkeypatch, db, key, begin_runtime_epoch(db, instance_id="gateway"))

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert reply["result"]["status"] == "queued" and reply["result"]["forwarded"] is True
        assert turns == []  # nothing ran in serve
        [mine] = list_session_admissions(db, session_id=key)
        assert (mine["status"], mine["payload"]["text"]) == ("queued", "is the build done?")
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_the_subagent_was_dispatched_from_an_earlier_segment_When_serve_gets_a_prompt_Then_it_runs_in_the_gateway(
        monkeypatch, tmp_path, owner_process):
    # Given: a local conversation the gateway reset into a new segment after dispatching from the first
    import json
    import sqlite3
    from hermes_state_local import POLICY_PREFIX
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session("local-first", "desktop")
        db.create_session(key, "test")
        conn = sqlite3.connect(tmp_path / "state.db")
        conn.execute("UPDATE sessions SET chat_id='local-first' WHERE id=?", (key,))
        conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                     (POLICY_PREFIX + "local-first", json.dumps({"lineage": ["local-first", key]})))
        conn.commit()
        conn.close()
        _running_delegation(tmp_path / "state.db", "local-first", owner_process.pid, _start_time(owner_process.pid))
        _gateway_is(monkeypatch, owner_process.pid)
        _owner(monkeypatch, db, key, begin_runtime_epoch(db, instance_id="gateway"))

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert reply["result"]["forwarded"] is True and turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_a_named_profile_served_by_the_default_gateway_When_serve_gets_a_prompt_Then_it_runs_in_that_gateway(
        monkeypatch, tmp_path, owner_process):
    # Given: the profile home has no gateway of its own; the default home's multiplexer serves it
    from pathlib import Path
    import gateway.status
    import hermes_constants
    default_root = tmp_path / "default-root"
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: default_root)
    monkeypatch.setattr(gateway.status, "live_gateway_pid_for_home",
                        lambda home: owner_process.pid if Path(home) == default_root else None)
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, owner_process.pid, _start_time(owner_process.pid))
        _owner(monkeypatch, db, key, begin_runtime_epoch(db, instance_id="gateway"))

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert reply["result"]["forwarded"] is True and turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_a_standalone_profile_gateway_owns_the_subagent_When_serve_gets_a_prompt_Then_it_runs_in_that_gateway(
        monkeypatch, tmp_path, owner_process):
    # Given: the profile runs its own gateway (gateway.standalone); the default home's has none
    from pathlib import Path
    import gateway.status
    import hermes_constants
    profile_home = Path(hermes_constants.get_hermes_home())
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path / "default-root")
    monkeypatch.setattr(gateway.status, "live_gateway_pid_for_home",
                        lambda home: owner_process.pid if Path(home) == profile_home else None)
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, owner_process.pid, _start_time(owner_process.pid))
        _owner(monkeypatch, db, key, begin_runtime_epoch(db, instance_id="gateway"))

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert reply["result"]["forwarded"] is True and turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_the_owner_is_not_the_gateway_When_serve_gets_a_prompt_Then_it_runs_here(
        monkeypatch, tmp_path, owner_process):
    # Given: a CLI or another standalone backend runs the child; no FIFO there to forward to
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, owner_process.pid, _start_time(owner_process.pid))
        _gateway_is(monkeypatch, None)

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert turns == [sid] and list_session_admissions(db, session_id=key) == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_the_gateway_that_owned_the_subagent_exited_When_serve_gets_a_prompt_Then_it_runs_here(
        monkeypatch, tmp_path):
    # Given
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    started = _start_time(gone.pid)
    gone.wait()
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, gone.pid, started)
        _gateway_is(monkeypatch, gone.pid)

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert turns == [sid]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_the_subagent_finished_When_serve_gets_a_prompt_Then_it_runs_here(
        monkeypatch, tmp_path, owner_process):
    # Given
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, owner_process.pid, _start_time(owner_process.pid),
                            state="completed")
        _gateway_is(monkeypatch, owner_process.pid)

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert turns == [sid]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_another_chats_subagent_runs_in_the_gateway_When_serve_gets_a_prompt_Then_it_runs_here(
        monkeypatch, tmp_path, owner_process):
    # Given
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", "some-other-chat", owner_process.pid,
                            _start_time(owner_process.pid))
        _gateway_is(monkeypatch, owner_process.pid)

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert turns == [sid]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_the_owner_pid_was_recycled_When_serve_gets_a_prompt_Then_it_runs_here(
        monkeypatch, tmp_path, owner_process):
    # Given: the live gateway has the row's pid, but it is not the process that dispatched (start time differs)
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, owner_process.pid, 1)
        _gateway_is(monkeypatch, owner_process.pid)

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert turns == [sid]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_this_process_is_the_gateway_that_owns_the_subagent_When_a_prompt_arrives_Then_it_runs_here(
        monkeypatch, tmp_path):
    # Given: the legacy dispatcher runs inside the owning gateway itself; forwarding would loop to itself
    import os
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, os.getpid(), _start_time(os.getpid()))
        _gateway_is(monkeypatch, os.getpid())

        # When
        reply = _submit(sid)

        # Then
        assert "error" not in reply, reply
        assert turns == [sid]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_Given_the_gateway_is_gone_by_the_forward_When_serve_gets_a_prompt_Then_it_is_refused_by_name(
        monkeypatch, tmp_path, owner_process):
    # Given: identity checked live, then the gateway stops before the dial
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, turns, _emitted = _legacy(monkeypatch, db)
    try:
        db.create_session(key, "test")
        _running_delegation(tmp_path / "state.db", key, owner_process.pid, _start_time(owner_process.pid))
        _gateway_is(monkeypatch, owner_process.pid)
        from tui_gateway import owner_forward

        def stopped(_home):
            raise owner_forward.OwnerNotRunning("the gateway that owns it is stopped")
        monkeypatch.setattr(owner_forward, "discover_owner", stopped)

        # When
        reply = _submit(sid)

        # Then
        assert reply["error"]["code"] == 4090 and reply["error"]["data"] == {"reason": "SESSION_NOT_OWNED"}
        assert "background subagent" in reply["error"]["message"]
        assert turns == []
    finally:
        server._sessions.pop(sid, None)
        db.close()
