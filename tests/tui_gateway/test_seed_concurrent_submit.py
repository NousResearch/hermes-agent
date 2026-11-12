"""Two transport dispatchers must not accept rows before a deferred seed settles."""

import errno
import threading

import pytest

from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session

from tests.tui_gateway.test_seeded_branch_retry import _create_branch, _prepare_branch
from tests.tui_gateway import test_stranded_session_adoption as adoption_cases

gateway = adoption_cases.gateway


class _Transport:
    def write(self, frame):
        return True

    def close(self):
        pass


class _ObservedLock:
    """Report a real contender's acquisition without changing lock ownership."""

    def __init__(self, lock, observed, state):
        self.lock = lock
        self.observed = observed
        self.state = state

    def __enter__(self):
        if threading.current_thread().name == "client-b" and self.state["contended"] is None:
            acquired = self.lock.acquire(blocking=False)
            self.state["contended"] = not acquired
            self.observed.set()
            if acquired:
                return self
        self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()


@pytest.mark.parametrize("fail_seed, compute_fallback, competing_owner", [
    (False, False, False), (True, False, False), (True, True, False), (True, False, True),
])
def test_concurrent_dispatch_waits_for_seed_settlement_before_accepting(
        gateway, monkeypatch, request, fail_seed, compute_fallback, competing_owner):
    mod, db, _ = gateway
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"seed {i}"}
                for i in range(501)]
    _prepare_branch(mod, db, monkeypatch, messages)
    parent_before = db.get_messages("seed-retry-parent", include_inactive=True)

    def defer_seed(*args, **kwargs):
        raise OSError(errno.ENOSPC, "defer initial seed")

    with monkeypatch.context() as patch:
        patch.setattr(db, "append_messages_batch", defer_seed)
        created = _create_branch(mod)
    assert "result" in created, created
    sid = created["result"]["session_id"]
    record = mod._sessions[sid]
    paused, observed, b_done = (threading.Event() for _ in range(3))
    state = {"contended": None}
    lock = _ObservedLock(record["history_lock"], observed, state)
    record["history_lock"] = lock
    record["_submit_admission_lock"] = _ObservedLock(threading.Lock(), observed, state)
    ensure = mod._ensure_session_db_row
    append_transaction = db._append_messages_in_transaction
    failed = False
    replies, errors = {}, []
    leases = {}
    release_slot = mod._release_active_session_slot
    request.addfinalizer(lambda: release_slot(record))

    def release_and_compete(session):
        released = release_slot(session)
        if competing_owner and threading.current_thread().name == "client-a":
            leases["released_rows"] = db.get_session("seed-retry-child"), db.get_messages(
                "seed-retry-child", include_inactive=True)
            lease, refusal = try_acquire_active_session(
                session_id="seed-retry-child", surface="desktop", config={},
                metadata={"live_session_id": "competing-runtime"}, track_liveness=True)
            assert refusal is None
            leases["competitor"] = lease
            request.addfinalizer(lease.release)
        return released

    monkeypatch.setattr(mod, "_release_active_session_slot", release_and_compete)


    def pause_before_seed(session):
        if threading.current_thread().name == "client-a":
            leases["first"] = session["active_session_lease"]
            assert leases["first"].enabled
            assert any(entry["lease_id"] == leases["first"].lease_id
                       for entry in active_session_registry_snapshot(strict=True))
            paused.set()
            assert observed.wait(10), "second public dispatcher did not reach the session lock"
            if not state["contended"]:
                assert b_done.wait(10), "unblocked second dispatcher did not finish"
        return ensure(session)

    def fail_second_chunk(conn, key, rows, **kwargs):
        nonlocal failed
        if fail_seed and key == "seed-retry-child" and len(rows) == 1 and rows[0]["content"] == "seed 500" and not failed:
            failed = True
            raise RuntimeError("second seed chunk failed")
        return append_transaction(conn, key, rows, **kwargs)

    monkeypatch.setattr(mod, "_ensure_session_db_row", pause_before_seed)
    monkeypatch.setattr(db, "_append_messages_in_transaction", fail_second_chunk)
    if compute_fallback:
        class BrokenSupervisor:
            def submit_turn(self, frame, *, on_complete=None):
                raise BrokenPipeError("compute dispatch failed before acceptance")

        monkeypatch.setattr(mod, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
        monkeypatch.setattr(mod, "_get_compute_host_supervisor", lambda *a: BrokenSupervisor())
    else:
        monkeypatch.setattr(mod, "_session_uses_compute_host", lambda *a: False)
    monkeypatch.setattr(mod, "_start_agent_build", lambda *a: None)
    monkeypatch.setattr(mod, "_run_after_agent_ready", lambda *a: None)
    monkeypatch.setattr(mod, "_restart_completed_failed_agent_build", lambda *a: False)

    def dispatch(name, text):
        try:
            replies[name] = mod.dispatch({
                "id": name, "method": "prompt.submit",
                "params": {"session_id": sid, "text": text, "queued": True},
            }, _Transport())
        except BaseException as exc:
            errors.append(exc)
        finally:
            if name == "b":
                b_done.set()

    first = threading.Thread(target=dispatch, args=("a", "rejected followup"), name="client-a")
    second = threading.Thread(target=dispatch, args=("b", "accepted followup"), name="client-b")
    first.start()
    try:
        assert paused.wait(10), (replies, errors)
        second.start()
    finally:
        first.join(15)
        if second.ident is not None:
            second.join(15)
    assert not first.is_alive() and not second.is_alive()
    assert not errors, errors
    if fail_seed:
        assert replies["a"]["error"]["code"] == 5071, replies
    else:
        assert replies["a"]["result"]["status"] == "streaming", replies
    registry = active_session_registry_snapshot(strict=True)
    if fail_seed:
        assert leases["first"].released
    if competing_owner:
        assert replies["b"]["error"]["code"] == 4090, replies
        assert (db.get_session("seed-retry-child"),
                db.get_messages("seed-retry-child", include_inactive=True)) == leases["released_rows"]
        assert [entry["lease_id"] for entry in registry] == [leases["competitor"].lease_id]
        assert not record.get("running")
        return
    current_lease = record.get("active_session_lease")
    assert current_lease is not None, "accepted submit lost its active-session lease after refusal"
    assert [entry["lease_id"] for entry in registry] == [current_lease.lease_id]
    if fail_seed:
        assert current_lease.lease_id != leases["first"].lease_id
    assert [row["content"] for row in db.get_messages("seed-retry-child", include_inactive=True)] == [
        *(f"seed {i}" for i in range(501)), *([] if fail_seed else ["rejected followup"]), "accepted followup",
    ]
    assert replies["b"]["result"]["status"] == ("streaming" if fail_seed else "queued"), replies
    assert bool(record.get("queued_prompt")) is not fail_seed
    assert "_branch_seed_pending" not in record
    assert db.get_messages("seed-retry-parent", include_inactive=True) == parent_before
