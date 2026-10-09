"""Dispatch settlement is correlated and independent of completion callbacks."""

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor, TurnSettlement


_LEASE_WRITER = """
import json, os, sys
from pathlib import Path
from hermes_state import SessionDB
print(json.dumps({'type': 'hello', 'host_pid': os.getpid()}), flush=True)
db = None
for line in sys.stdin:
    frame = json.loads(line)
    if frame['type'] == 'turn.start':
        turn = frame
        db = SessionDB(Path(frame['db_path']))
        holder = f'pid={os.getpid()}:turn=controlled'
        assert db.try_acquire_session_turn_lease(frame['session_key'], holder, ttl_seconds=120)
        print(json.dumps({'type': 'rpc', 'message': {'ready': True}}), flush=True)
    elif frame['type'] == 'finish':
        db.append_message(turn['session_key'], 'assistant', content='original-child-finished')
        db.release_session_turn_lease(turn['session_key'], holder)
        db.close()
        print(json.dumps({'type': 'turn.end', 'request_id': turn['request_id']}), flush=True)
    elif frame['type'] == 'shutdown':
        break
"""


def _supervisor(monkeypatch, proc=None):
    supervisor = HostSupervisor(autostart=False)
    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_send_frame", lambda _frame: None)
    supervisor._proc = proc
    return supervisor


@pytest.mark.parametrize("terminal_type", ["turn.end", "turn.error"])
def test_terminal_receipt_precedes_failing_callback(monkeypatch, terminal_type):
    supervisor = _supervisor(monkeypatch)
    settlement = TurnSettlement()

    def complete(_frame):
        assert settlement.is_set()
        raise RuntimeError("metadata adoption failed")

    supervisor.submit_turn({"request_id": "owned", "sid": "s"},
                           on_complete=complete, settlement=settlement)
    supervisor._handle_host_frame({"type": "turn.end", "request_id": "foreign"})
    assert not settlement.is_set()
    supervisor._handle_host_frame({"type": "hb", "progress_counter": 10})
    assert not settlement.is_set()
    supervisor._handle_host_frame({"type": terminal_type, "request_id": "owned"})
    assert settlement.is_set()


def test_send_failure_is_not_settlement(monkeypatch):
    supervisor = _supervisor(monkeypatch)
    settlement = TurnSettlement()
    frames = []

    def fail_send(_frame):
        raise BrokenPipeError("partial send")

    monkeypatch.setattr(supervisor, "_send_frame", fail_send)
    with pytest.raises(BrokenPipeError):
        supervisor.submit_turn({"request_id": "r"}, on_complete=frames.append,
                               settlement=settlement)
    assert frames[0]["reason"] == "send_failed"
    assert not settlement.is_set()


def test_exact_child_exit_settles_even_after_replacement(monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                             stdin=subprocess.PIPE)
    try:
        supervisor = _supervisor(monkeypatch, child)
        settlement = TurnSettlement()
        supervisor.submit_turn({"request_id": "old"}, settlement=settlement)
        assert not settlement.is_set()
        # A different live host cannot erase the original child's exit evidence.
        supervisor._proc = SimpleNamespace(poll=lambda: None)
        child.communicate(timeout=10)
        assert settlement.is_set()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def test_uncertain_child_liveness_keeps_fence():
    def inaccessible():
        raise PermissionError("cannot observe child")

    settlement = TurnSettlement()
    settlement.process = SimpleNamespace(poll=inaccessible)
    assert not settlement.is_set()


def test_exit_notification_cannot_settle_replacement_dispatch(monkeypatch):
    old = SimpleNamespace(wait=lambda: 1, poll=lambda: 1)
    replacement = SimpleNamespace(poll=lambda: None)
    supervisor = _supervisor(monkeypatch, old)
    before, after = TurnSettlement(), TurnSettlement()
    completed = []
    supervisor.submit_turn({"request_id": "old"}, settlement=before)

    def admit_replacement():
        supervisor._proc = replacement
        supervisor.submit_turn({"request_id": "new"}, settlement=after,
                               on_complete=completed.append)

    # This boundary used to clear the old proc before detaching all pending work.
    monkeypatch.setattr(supervisor, "_remove_registry", admit_replacement)
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda: None)
    supervisor._wait_for_exit(old)
    assert before.is_set()
    assert not after.is_set()
    assert completed == []
    supervisor._handle_host_frame({"type": "turn.end", "request_id": "new"})
    assert after.is_set()
    assert len(completed) == 1


def test_exit_settles_all_receipts_before_fallible_notification(monkeypatch):
    child = SimpleNamespace(wait=lambda: 1, poll=lambda: 1)
    supervisor = _supervisor(monkeypatch, child)
    first, second = TurnSettlement(), TurnSettlement()
    supervisor.submit_turn({"request_id": "a"}, settlement=first)
    supervisor.submit_turn({"request_id": "b"}, settlement=second)

    def fail_sink(_frame):
        assert first.completed.is_set() and second.completed.is_set()
        raise RuntimeError("transport is gone")

    monkeypatch.setattr(supervisor, "rpc_sink", fail_sink)
    monkeypatch.setattr(supervisor, "_remove_registry", lambda: None)
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda: None)
    with pytest.raises(RuntimeError, match="transport is gone"):
        supervisor._wait_for_exit(child)
    assert first.is_set() and second.is_set()


def test_fast_completion_does_not_replace_next_dispatch_receipt(monkeypatch):
    session = {"history_lock": threading.Lock(), "session_key": "stored",
               "history": [], "attached_images": [], "running": True}
    supervisor = _supervisor(monkeypatch)
    sent = []
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    monkeypatch.setattr(server, "_compute_host_turn_frame", lambda *a, **k: {"sid": "s"})
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})

    def done(*_args):
        server._submit_prompt_to_compute_host("second", "s", session, "next")

    def send(frame):
        sent.append((frame, session["_compute_host_turn_settlement"]))
        if len(sent) == 1:
            supervisor._handle_host_frame({"type": "turn.end", "request_id": frame["request_id"]})

    monkeypatch.setattr(supervisor, "_send_frame", send)
    monkeypatch.setattr(server, "_on_compute_host_turn_done", done)
    server._submit_prompt_to_compute_host("first", "s", session, "first")
    assert len(sent) == 2
    assert sent[0][1].is_set()
    assert not sent[1][1].is_set()
    assert session["_compute_host_turn_settlement"] is sent[1][1]
    assert session["_compute_host_turn_id"] == sent[1][0]["request_id"]


def test_old_completion_cannot_release_newer_deferred_turn(monkeypatch, tmp_path):
    from hermes_cli.active_sessions import try_acquire_active_session

    lease, err = try_acquire_active_session(
        session_id="stored", surface="desktop", config={}, registry_home=tmp_path,
        track_liveness=True, metadata={"live_session_id": "s"})
    assert lease is not None and err is None
    monkeypatch.setattr(server, "_deferred_active_session_leases", {})
    monkeypatch.setattr(server, "_deferred_active_session_lease_ages", {})
    monkeypatch.setattr(server, "_deferred_active_session_settlements", {})
    monkeypatch.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _s: True)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    supervisor = _supervisor(monkeypatch, SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    monkeypatch.setattr(server, "_compute_host_turn_frame", lambda *a, **k: {"sid": "s"})
    monkeypatch.setattr(server, "_compute_host_adopt_frame_meta", lambda *a: None)
    monkeypatch.setattr(server, "_clear_inflight_turn", lambda *a: None)
    monkeypatch.setattr(server, "_compute_host_session_info", lambda *a: {})
    monkeypatch.setattr(server, "_emit", lambda *a: None)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *a: None)
    session = {"_sid": "s", "session_key": "stored", "active_session_lease": lease,
               "history_lock": threading.Lock(), "running": True, "_compute_host_active": True,
               "attached_images": []}
    newer = []

    def next_turn_admitted_and_closed(*_args):
        # The prior callback exposed idle and released history_lock before this boundary.
        assert not session["running"]
        session["running"] = True
        server._submit_prompt_to_compute_host("new", "s", session, "new")
        newer.append(session["_compute_host_turn_settlement"])
        server._settle_isolated_turn_before_close(session)

    monkeypatch.setattr(server, "_apply_compute_host_metadata_mirror", next_turn_admitted_and_closed)
    try:
        server._submit_prompt_to_compute_host("old", "s", session, "old")
        supervisor._handle_host_frame({"type": "turn.end", "request_id": session["_compute_host_turn_id"]})
        assert newer and not newer[0].is_set()
        assert not lease.released
        assert lease.lease_id in server._own_live_lease_ids()
    finally:
        lease.release()


def test_real_child_keeps_submit_fence_until_correlated_completion(monkeypatch, tmp_path):
    """Real supervisor, lease and DB; the controlled host blocks instead of calling a provider."""
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_deferred_active_session_leases", {})
    monkeypatch.setattr(server, "_deferred_active_session_lease_ages", {})
    monkeypatch.setattr(server, "_deferred_active_session_settlements", {})
    monkeypatch.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    key = "stored"
    db = SessionDB(home / "state.db")
    db.create_session(key, source="desktop")
    db.append_message(key, "user", content="original-turn")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    parent, err = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, registry_home=home, track_liveness=True,
        metadata={"live_session_id": "original"})
    assert parent is not None and err is None
    ready = threading.Event()
    supervisor = HostSupervisor(
        argv=[sys.executable, "-c", _LEASE_WRITER], cwd=Path(__file__).resolve().parents[2],
        registry_path=home / "host.json", expected_build_sha="unknown", autostart=False,
        rpc_sink=lambda msg: ready.set() if msg.get("ready") else None)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    monkeypatch.setattr(server, "_compute_host_turn_frame", lambda *a, **k: {
        "sid": "original", "session_key": key, "db_path": str(home / "state.db")})

    def lost_callback(*_args):
        raise RuntimeError("completion metadata failed")

    monkeypatch.setattr(server, "_on_compute_host_turn_done", lost_callback)
    session = {"_sid": "original", "session_key": key, "history_lock": threading.Lock(),
               "active_session_lease": parent, "running": True, "_compute_host_active": True,
               "attached_images": []}
    contender = None
    try:
        response = server._submit_prompt_to_compute_host("r", "original", session, "original-turn")
        assert "error" not in response, response
        assert ready.wait(10)
        receipt = session["_compute_host_turn_settlement"]
        server._settle_isolated_turn_before_close(session)
        assert session["_deferred_active_session_lease"] is parent
        age = server._deferred_active_session_lease_ages[parent.lease_id]
        sweep_time = age + server._DEFERRED_ACTIVE_SESSION_LEASE_TTL_SECONDS + 1
        assert server._reap_stale_deferred_leases(now=sweep_time) == 0
        assert receipt.process.poll() is None
        contender, refusal = try_acquire_active_session(
            session_id=key, surface="desktop", config={}, registry_home=home, track_liveness=True,
            metadata={"live_session_id": "second"})
        assert contender is None and getattr(refusal, "reason", "") == "SESSION_NOT_OWNED"
        assert [m["content"] for m in db.get_messages(key)] == ["original-turn"]
        supervisor._send_frame({"type": "finish"})
        assert receipt.completed.wait(10)
        assert receipt.process.poll() is None  # The shared host survives this turn.
        assert server._reap_stale_deferred_leases(now=sweep_time) == 1
        contender, refusal = try_acquire_active_session(
            session_id=key, surface="desktop", config={}, registry_home=home, track_liveness=True,
            metadata={"live_session_id": "second"})
        assert contender is not None and refusal is None
        server._write_submit_user_row({"session_key": key, "profile_home": None}, "next-turn", None)
        assert [m["content"] for m in db.get_messages(key)] == [
            "original-turn", "original-child-finished", "next-turn"]
    finally:
        supervisor.shutdown()
        server._release_deferred_active_session_lease(session)
        if contender is not None:
            contender.release()
        parent.release()
        db.close()
