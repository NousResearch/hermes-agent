"""A room turn keeps its driver lease while it waits for the profile lock and while it runs.

``turn_lock(profile)`` can block for up to ``bot_mode.turn_wait_seconds`` (120 s) behind another
turn on the same profile, past the 30 s room lease. Scaled down here: a lock wait longer than the
lease ttl must still settle the turn, and a lease genuinely taken over during that wait must still
fence the turn before it is submitted.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, PROFILE, ROOM_ID, _identity, _wait_for
from hermes_agent.tui_gateway.hosted_room_driver import HostedRoomRuntime
from hermes_agent.tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC

TTL = 0.6


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    hosted_rooms.create_room(
        path, room_id=ROOM_ID, name="Release room", members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=BINDING.gateway_id, now=time.time())
    return path


def _server(submits: list, *, finish_after: float):
    """Installed server surface whose turn finishes ``finish_after`` seconds after submit."""
    def ok(result):
        return lambda rid, params: {"id": rid, "result": result}

    def prompt_submit(rid, params):
        submits.append(params)
        done = params["_hosted_terminal_callback"]
        threading.Timer(finish_after, done, args=({"status": "settled", "text": "Done."},)).start()
        return {"id": rid, "result": {"status": "streaming"}}

    return SimpleNamespace(
        _methods={
            "session.list": ok({"sessions": []}), "session.create": ok({"session_id": "runtime"}),
            "session.resume": ok({"session_id": "runtime"}), "session.history": ok({"messages": []}),
            "prompt.submit": prompt_submit},
        _sessions={}, _sessions_lock=threading.Lock(),
        _pending_approval_request_payload=lambda _session_key: None)


def _run(db: Path, turn_lock, submits: list, clock=time.time) -> dict:
    identity = _identity()
    state.admit_task(db, identity, clock=time.time, payload={
        "target_profile": PROFILE, "target_member_id": "member-ops", "prompt": "Ship it.",
        "source_event_seq": 1})
    runtime = HostedRoomRuntime(
        db_path=db, rooms=[BINDING], rpc=HostedRoomServerRPC(_server(submits, finish_after=TTL)),
        turn_lock=turn_lock, clock=clock, lease_ttl_seconds=TTL, poll_interval_seconds=0.01,
        active_poll_interval_seconds=0.02)
    runtime.start()
    try:
        _wait_for(lambda: (
            state.get_task(db, identity)["status"] in state.TERMINAL_STATUSES
            or runtime.status()["last_error"] is not None), timeout=10.0)
    finally:
        assert runtime.stop(timeout=5.0)
    return {"task": state.get_task(db, identity), "last_error": runtime.status()["last_error"]}


def test_turn_settles_when_the_profile_lock_wait_and_the_run_outlast_the_lease(db: Path):
    @contextlib.contextmanager
    def busy_lock(_profile):
        # health: allow HX013 -- the lock wait outlasting the lease ttl IS the behaviour under test
        time.sleep(TTL * 2)  # another turn on this profile holds the lock past the lease ttl
        yield

    submits: list = []
    outcome = _run(db, busy_lock, submits)

    assert len(submits) == 1
    assert outcome["last_error"] is None
    assert outcome["task"]["status"] == "settled"


def test_turn_is_fenced_before_submit_when_the_lease_is_taken_over_during_the_wait(db: Path):
    offset = [0.0]

    def clock():
        return time.time() + offset[0]

    @contextlib.contextmanager
    def lock_while_another_process_reclaims(_profile):
        offset[0] += TTL * 2  # this process stalls past its lease; another process reclaims it
        state.acquire_lease(
            db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
            process_generation="process-b", ttl_seconds=3600, clock=clock)
        yield

    submits: list = []
    outcome = _run(db, lock_while_another_process_reclaims, submits, clock=clock)

    assert submits == []
    assert outcome["task"]["status"] != "settled"
    assert "fenced" in (outcome["last_error"] or "")


def test_ambiguous_submit_recovers_the_room_when_the_keeper_renewal_lands_late(db: Path, monkeypatch):
    """A keeper renewal committed before the turn's own renewal but cached after it must not
    roll the cached expiry back: the ambiguous deferral would end before the row expires, the
    next scan would renew the same generation in place, and the attempt would stay running."""
    keeper_committed, submitted = threading.Event(), threading.Event()
    real_renew = state.renew_lease

    def renew(*args, **kwargs):
        renewed = real_renew(*args, **kwargs)
        if threading.current_thread().name.startswith("hosted-room-lease-") and not keeper_committed.is_set():
            keeper_committed.set()
            submitted.wait(5.0)  # descheduled between its commit and its cache write
        return renewed

    monkeypatch.setattr(state, "renew_lease", renew)

    @contextlib.contextmanager
    def lock_released_after_the_keeper_tick(_profile):
        keeper_committed.wait(5.0)
        # health: allow HX013 -- the gap between the two committed expiries IS the race window under test
        time.sleep(0.1)  # the turn's renewal commits a later expiry than the keeper's
        yield

    class AmbiguousSubmit:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def submit(self, **_kwargs):
            submitted.set()
            raise RuntimeError("connection reset after send")

    identity = _identity()
    state.admit_task(db, identity, clock=time.time, payload={
        "target_profile": PROFILE, "target_member_id": "member-ops", "prompt": "Ship it.",
        "source_event_seq": 1})
    runtime = HostedRoomRuntime(
        db_path=db, rooms=[BINDING], rpc=HostedRoomServerRPC(_server([], finish_after=TTL)),
        turn_lock=lock_released_after_the_keeper_tick, clock=time.time, lease_ttl_seconds=TTL,
        poll_interval_seconds=0.01, active_poll_interval_seconds=0.02)
    resolve = runtime._transport_for
    runtime._transport_for = lambda binding, task: AmbiguousSubmit(resolve(binding, task))
    runtime.start()
    try:
        _wait_for(lambda: state.get_task(db, identity)["status"] != "running" and submitted.is_set(),
                  timeout=TTL * 8)
    finally:
        assert runtime.stop(timeout=5.0)
    assert state.get_task(db, identity)["status"] == "indeterminate"
