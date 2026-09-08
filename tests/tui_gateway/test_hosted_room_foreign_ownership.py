"""Ownership of an attempt another process admitted, seen through a separate session registry.

A hosted room's durable store is shared, but a session's *running* state is process-local: a
successor gateway resolves the same session id and sees it idle simply because it never ran it.
These tests give the owner and the successor separate registries -- the one thing a shared fake
cannot express -- and pin that no recovery path treats that emptiness as proof the original
execution ended.

Real SQLite and the real Linux profile lock, rooted in the test's own directory. Nothing here can
create a session or submit a prompt: both raise.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime, room_session_title


ROOM_ID = "foreign-room"
PROFILE = "ops"
BINDING = HostedRoomBinding(room_id=ROOM_ID, gateway_id="gateway-a", authority_epoch=1)
OWNER_PROCESS = "owner-process"


class SessionRegistry:
    """One process's view of the shared session store.

    Session ids are durable and identical across processes; ``active`` is not, because only the
    process that submitted a turn is running it. ``interrupt_keeps_running`` models an adapter that
    accepts the interrupt request while its run thread is still alive.
    """

    def __init__(self, *, interrupt_keeps_running: bool = False) -> None:
        self.sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.interrupt_keeps_running = interrupt_keeps_running

    def add_session(
        self, session_id: str, *, active: bool = False, task_id: str | None = None,
        execution_generation: int | None = None, history: list[dict[str, Any]] | None = None,
        status: str | None = None, profile: str = PROFILE,
        title: str = room_session_title(ROOM_ID)) -> str:
        self.sessions[(profile, title)] = {"session_id": session_id, "title": title}
        self.states[session_id] = {
            "active": active, "task_id": task_id, "execution_generation": execution_generation,
            "history": list(history or []), "status": status}
        return session_id

    def resolve_exact(self, *, profile: str, title: str, source: str):
        self.calls.append(("resolve_exact", {"profile": profile, "title": title}))
        session = self.sessions.get((profile, title))
        return dict(session) if session is not None else None

    def create(self, **kwargs):
        raise AssertionError("recovery must never create a room session")

    def submit(self, **kwargs):
        raise AssertionError("recovery must never submit a prompt")

    def resume(self, *, profile: str, session_id: str, source: str):
        self.calls.append(("resume", {"session_id": session_id}))
        return {"session_id": session_id}

    def history(self, *, profile: str, session_id: str, source: str):
        self.calls.append(("history", {"session_id": session_id}))
        return [dict(message) for message in self.states[session_id]["history"]]

    def info(self, *, profile: str, session_id: str, source: str):
        self.calls.append(("info", {"session_id": session_id}))
        session = self.states[session_id]
        info = {"active": session["active"], "task_id": session["task_id"]}
        if session["status"] is not None:
            info["status"] = session["status"]
        return info

    def interrupt(self, *, profile: str, session_id: str, source: str, expected_task_id: str,
                  expected_execution_generation: int | None = None):
        self.calls.append(("interrupt", {"session_id": session_id, "task_id": expected_task_id}))
        session = self.states[session_id]
        if session["task_id"] != expected_task_id or not session["active"]:
            return {"interrupted": False}
        if not self.interrupt_keeps_running:
            session["active"] = False
        return {"interrupted": True}

    def count(self, method: str) -> int:
        return sum(name == method for name, _params in self.calls)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    hosted_rooms.create_room(
        path, room_id=ROOM_ID, name="Foreign ownership", authority_gateway_id=BINDING.gateway_id,
        members=[{"profile": PROFILE, "handle": PROFILE}], now=time.time())
    return path


def _runtime(db: Path, tmp_path: Path, rpc: SessionRegistry, clock, **kwargs) -> HostedRoomRuntime:
    from tools.bot_relay import acquire_turn_lock

    return HostedRoomRuntime(
        db_path=db, rooms=[BINDING], rpc=rpc, clock=clock,
        turn_lock=lambda profile: acquire_turn_lock(tmp_path / "locks", profile, timeout_seconds=2.0),
        lease_ttl_seconds=kwargs.pop("lease_ttl_seconds", 100.0), **kwargs)


def _identity(task_id: str = "task-1") -> state.TaskIdentity:
    return state.TaskIdentity(ROOM_ID, task_id, "thread-1", f"turn-{task_id}")


def _foreign_attempt(db: Path, clock, identity: state.TaskIdentity, *, admitted: bool):
    """A prior process's attempt, its lease already expired, optionally past its admission fence."""
    state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "Fixture only", "source_event_seq": 1},
        clock=clock)
    owner_lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation=OWNER_PROCESS, ttl_seconds=1.0, clock=clock)
    attempt = state.start_task(db, identity, owner_lease, expected_cancel_generation=0, clock=clock)
    if admitted:
        state.fence_task_admission(db, attempt, clock=clock)
    return owner_lease, attempt


def _successor(db: Path, tmp_path: Path, rpc: SessionRegistry, clock, **kwargs):
    """The next process: takes the room lease and fences the abandoned attempt to indeterminate."""
    runtime = _runtime(db, tmp_path, rpc, clock, **kwargs)
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation=runtime.process_generation, ttl_seconds=100.0, clock=clock)
    runtime._leases[ROOM_ID] = lease
    state.recover_room(db, lease, clock=clock)
    return runtime, lease


def _receipt(identity: state.TaskIdentity, execution_generation: int) -> dict[str, Any]:
    return {
        "role": "assistant", "task_id": identity.task_id,
        "execution_generation": execution_generation, "status": "settled",
        "message_id": f"reply:{identity.task_id}", "content": "The owner's turn finished."}


@pytest.mark.parametrize("defer_first", [False, True], ids=["indeterminate", "deferred"])
def test_retry_refuses_to_requeue_an_attempt_another_process_admitted(
    db: Path, tmp_path: Path, defer_first: bool
):
    """An empty session registry in this process is not evidence about the original execution."""
    clock = [time.time()]
    identity = _identity()
    owner_lease, attempt = _foreign_attempt(db, lambda: clock[0], identity, admitted=True)
    clock[0] = owner_lease.expires_at + 1.0
    # Cold resume: the same durable session id, idle here because this process never ran it.
    observer_rpc = SessionRegistry()
    observer_rpc.add_session("session-1", active=False)
    successor, lease = _successor(
        db, tmp_path, observer_rpc, lambda: clock[0], indeterminate_defer_seconds=5.0)
    if defer_first:
        successor._reconcile_indeterminate(BINDING, lease)
        clock[0] += 6.0
        successor._reconcile_indeterminate(BINDING, lease)
        assert state.get_task(db, identity)["status"] == "deferred"
    before = state.get_task(db, identity)
    assert before["admitted_at"] is not None

    with pytest.raises(state.InvalidTaskTransitionError, match="another process admitted"):
        successor.retry_indeterminate(identity)

    after = state.get_task(db, identity)
    assert (after["status"], after["admitted_at"], after["run_process_generation"]) == (
        before["status"], before["admitted_at"], OWNER_PROCESS)
    assert observer_rpc.count("submit") == 0


@pytest.mark.parametrize("defer_first", [False, True], ids=["indeterminate", "deferred"])
def test_refused_retry_then_stop_reports_uncertainty_not_cancellation(
    db: Path, tmp_path: Path, defer_first: bool
):
    """The composed regression: Retry, then Stop, while the owner's execution is still running."""
    clock = [time.time()]
    identity = _identity()
    owner_lease, attempt = _foreign_attempt(db, lambda: clock[0], identity, admitted=True)
    clock[0] = owner_lease.expires_at + 1.0
    # Two registries: the owner's, where the turn is still running, and the successor's, where the
    # same durable session id resolves to an idle record because this process never ran it.
    owner_rpc = SessionRegistry()
    owner_rpc.add_session(
        "session-1", active=True, task_id=identity.task_id,
        execution_generation=attempt.execution_generation)
    observer_rpc = SessionRegistry()
    observer_rpc.add_session("session-1", active=False)
    assert owner_rpc.states is not observer_rpc.states
    successor, lease = _successor(
        db, tmp_path, observer_rpc, lambda: clock[0], indeterminate_defer_seconds=5.0)
    if defer_first:
        successor._reconcile_indeterminate(BINDING, lease)
        clock[0] += 6.0
        successor._reconcile_indeterminate(BINDING, lease)
        assert state.get_task(db, identity)["status"] == "deferred"

    with pytest.raises(state.InvalidTaskTransitionError, match="another process admitted"):
        successor.retry_indeterminate(identity)
    result = successor.cancel(identity, cancel_id="stop-a-foreign-attempt")

    assert result["status"] == "indeterminate"
    assert state.get_task(db, identity)["admitted_at"] is not None
    assert observer_rpc.count("submit") == 0
    # Repeating the same Stop stays idempotent and still refuses to claim the run ended.
    assert successor.cancel(identity, cancel_id="stop-a-foreign-attempt")["status"] != "cancelled"
    # The owner's execution was never touched from here.
    assert owner_rpc.states["session-1"]["active"] is True
    assert owner_rpc.calls == []


def test_a_never_admitted_foreign_attempt_still_recovers_by_retry(db: Path, tmp_path: Path):
    """The attempt reached no transport, so requeueing it cannot duplicate an execution."""
    clock = [time.time()]
    identity = _identity()
    owner_lease, _attempt = _foreign_attempt(db, lambda: clock[0], identity, admitted=False)
    clock[0] = owner_lease.expires_at + 1.0
    observer_rpc = SessionRegistry()
    observer_rpc.add_session("session-1", active=False)
    successor, _lease = _successor(db, tmp_path, observer_rpc, lambda: clock[0])

    retried = successor.retry_indeterminate(identity)

    assert retried["status"] == "queued"
    assert retried["admitted_at"] is None
    # And a queued task is still cancelled outright, with no transport round trip.
    cancelled = successor.cancel(identity, cancel_id="stop-a-queued-task")
    assert cancelled["status"] == "cancelled"
    assert observer_rpc.count("interrupt") == 0


def _uncertain_attempt(
    db: Path, tmp_path: Path, clock, identity: state.TaskIdentity, rpc: SessionRegistry, *,
    foreign: bool, deferred: bool):
    """An admitted attempt parked in one of the two uncertain states, owned here or elsewhere."""
    state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "Fixture only", "source_event_seq": 1},
        clock=clock)
    runtime = _runtime(db, tmp_path, rpc, clock)
    owner_lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation=OWNER_PROCESS if foreign else runtime.process_generation,
        ttl_seconds=1.0, clock=clock)
    attempt = state.start_task(db, identity, owner_lease, expected_cancel_generation=0, clock=clock)
    state.fence_task_admission(db, attempt, clock=clock)
    clock_value = owner_lease.expires_at + 1.0
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation=runtime.process_generation, ttl_seconds=100.0, clock=lambda: clock_value)
    runtime._leases[ROOM_ID] = lease
    state.recover_room(db, lease, clock=lambda: clock_value)
    if deferred:
        state.defer_indeterminate_task(
            db, identity, lease, expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=attempt.cancel_generation, reason="member_unavailable",
            clock=lambda: clock_value)
    assert state.get_task(db, identity)["status"] == ("deferred" if deferred else "indeterminate")
    return runtime, attempt, clock_value


@pytest.mark.parametrize("deferred", [False, True], ids=["indeterminate", "deferred"])
@pytest.mark.parametrize("foreign", [False, True], ids=["same-owner", "foreign-owner"])
def test_an_exact_receipt_settles_uncertain_work_in_either_state(
    db: Path, tmp_path: Path, deferred: bool, foreign: bool
):
    """Completion is harvested before any replay: same fenced path, both uncertain states."""
    clock = [time.time()]
    identity = _identity()
    rpc = SessionRegistry()
    runtime, attempt, now = _uncertain_attempt(
        db, tmp_path, lambda: clock[0], identity, rpc, foreign=foreign, deferred=deferred)
    clock[0] = now
    rpc.add_session(
        "session-1", active=False, history=[_receipt(identity, attempt.execution_generation)])

    resolved = runtime.retry_indeterminate(identity)

    assert resolved["status"] == "settled"
    assert resolved["result"]["text"] == "The owner's turn finished."
    assert resolved["execution_generation"] == attempt.execution_generation
    assert resolved["admitted_at"] is not None, "settlement keeps the admission evidence"
    assert rpc.count("submit") == 0
    # A settled attempt is terminal: the composed second Retry refuses instead of replaying.
    with pytest.raises(state.InvalidTaskTransitionError, match="cannot retry task in state"):
        runtime.retry_indeterminate(identity)


@pytest.mark.parametrize("deferred", [False, True], ids=["indeterminate", "deferred"])
@pytest.mark.parametrize(
    "mismatch",
    [
        {"task_id": "other-task"},
        {"execution_generation": 99},
        {"role": "user"},
        {"status": "running"},
    ],
    ids=["wrong-task", "wrong-generation", "not-an-assistant-turn", "not-terminal"])
def test_a_mismatched_receipt_never_settles_another_attempt(
    db: Path, tmp_path: Path, deferred: bool, mismatch: dict[str, Any]
):
    """Only this attempt's own exact receipt is proof; anything else leaves the state untouched."""
    clock = [time.time()]
    identity = _identity()
    rpc = SessionRegistry()
    runtime, attempt, now = _uncertain_attempt(
        db, tmp_path, lambda: clock[0], identity, rpc, foreign=True, deferred=deferred)
    clock[0] = now
    rpc.add_session(
        "session-1", active=False,
        history=[{**_receipt(identity, attempt.execution_generation), **mismatch}])

    # No usable receipt, so the foreign-ownership guard is what answers: refusal, not replay.
    with pytest.raises(state.InvalidTaskTransitionError, match="another process admitted"):
        runtime.retry_indeterminate(identity)

    unchanged = state.get_task(db, identity)
    assert unchanged["status"] == ("deferred" if deferred else "indeterminate")
    assert unchanged["settlement_id"] is None
    assert rpc.count("submit") == 0


class PeerClient:
    """A stand-in for the peer client behind the real `PeerHostedRoomTransport`.

    The transport's request carries only room id and grant (`_scoped`); it is the client that
    binds an answer to a run, because `PeerRunsHTTPClient.status` looks up its stored observation
    receipt and returns that receipt's task id and execution generation
    (`tui_gateway/hosted_room_peer_http.py:507-517`). This fake returns those fields directly, so
    these tests cover the transport's routing and the driver's reply-identity check -- **not** the
    real HTTP receipt lookup, which `tests/tui_gateway/test_hosted_room_peer_http.py` covers.
    """

    def __init__(self, *, task_id: str | None, execution_generation: int | None, status: str) -> None:
        self.task_id, self.execution_generation, self.status_value = (
            task_id, execution_generation, status)
        self.scopes: list[dict[str, Any]] = []

    def prepare(self, *, room_id: str, grant: str, profile: str, source: str, create: bool, **extra):
        assert create is False, "recovery must never create a peer session"
        return {"session_id": "peer-session"}

    def history(self, *, room_id: str, grant: str, profile: str, session_id: str):
        return []

    def status(self, *, room_id: str, grant: str, profile: str, session_id: str):
        self.scopes.append({"room_id": room_id, "grant": grant, "profile": profile})
        return {
            "active": False, "status": self.status_value, "task_id": self.task_id,
            "execution_generation": self.execution_generation}

    def dispatch(self, **kwargs):
        raise AssertionError("recovery must never dispatch to the peer")


def _peer_transport(runtime, task_id: str | None, execution_generation: int | None, status: str):
    from tui_gateway.hosted_room_peer_transport import PeerHostedRoomTransport, PeerMemberRoute

    client = PeerClient(
        task_id=task_id, execution_generation=execution_generation, status=status)
    route = PeerMemberRoute(
        home_install_id="home-install", member_id=PROFILE, target_install_id="target-install",
        target_profile=PROFILE, capability_digest="f" * 64, cancellation_scope_id="scope-1",
        trace_id="trace-1", grant="grant-1")
    transport = PeerHostedRoomTransport(binding=BINDING, route=route, client=client)
    runtime.transport_resolver = lambda binding, task: transport
    return client


@pytest.mark.parametrize("deferred", [False, True], ids=["indeterminate", "deferred"])
def test_a_peer_reported_cancellation_of_this_exact_attempt_resolves_it(
    db: Path, tmp_path: Path, deferred: bool
):
    """A peer answer that names this exact task and generation terminalizes it without replay."""
    clock = [time.time()]
    identity = _identity()
    rpc = SessionRegistry()
    runtime, attempt, now = _uncertain_attempt(
        db, tmp_path, lambda: clock[0], identity, rpc, foreign=True, deferred=deferred)
    clock[0] = now
    client = _peer_transport(runtime, identity.task_id, attempt.execution_generation, "cancelled")

    resolved = runtime.retry_indeterminate(identity)

    assert resolved["status"] == "cancelled"
    # The transport's own request carries room + grant; the run identity comes back in the answer.
    assert client.scopes and client.scopes[0]["room_id"] == ROOM_ID
    assert client.scopes[0]["grant"] == "grant-1"
    assert rpc.count("submit") == 0


@pytest.mark.parametrize("deferred", [False, True], ids=["indeterminate", "deferred"])
@pytest.mark.parametrize(
    ("task_id", "execution_generation"),
    [(None, None), ("other-task", 1), ("task-1", 99)],
    ids=["unbound", "another-task", "another-generation"])
def test_a_cancellation_that_names_another_run_is_not_proof(
    db: Path, tmp_path: Path, deferred: bool, task_id: str | None, execution_generation: int | None
):
    """An unbound or mismatched terminal status says nothing about this attempt."""
    clock = [time.time()]
    identity = _identity()
    rpc = SessionRegistry()
    runtime, _attempt, now = _uncertain_attempt(
        db, tmp_path, lambda: clock[0], identity, rpc, foreign=True, deferred=deferred)
    clock[0] = now
    _peer_transport(runtime, task_id, execution_generation, "cancelled")

    with pytest.raises(state.InvalidTaskTransitionError, match="another process admitted"):
        runtime.retry_indeterminate(identity)

    unchanged = state.get_task(db, identity)
    assert unchanged["status"] == ("deferred" if deferred else "indeterminate")
    assert unchanged["admitted_at"] is not None


def test_retry_resolves_a_foreign_attempt_from_its_exact_terminal_receipt(db: Path, tmp_path: Path):
    """Verified cessation, the one positive path: the owner's result is in the durable history."""
    clock = [time.time()]
    identity = _identity()
    owner_lease, attempt = _foreign_attempt(db, lambda: clock[0], identity, admitted=True)
    clock[0] = owner_lease.expires_at + 1.0
    observer_rpc = SessionRegistry()
    observer_rpc.add_session(
        "session-1", active=False, history=[_receipt(identity, attempt.execution_generation)])
    successor, _lease = _successor(db, tmp_path, observer_rpc, lambda: clock[0])

    resolved = successor.retry_indeterminate(identity)

    assert resolved["status"] == "settled"
    assert resolved["result"]["text"] == "The owner's turn finished."
    assert observer_rpc.count("submit") == 0


def test_a_receipt_that_lands_during_a_stop_settles_instead_of_cancelling(db: Path, tmp_path: Path):
    """Completion before stop acknowledgement keeps the member's answer."""
    clock = [time.time()]
    identity = _identity()
    owner_lease, attempt = _foreign_attempt(db, lambda: clock[0], identity, admitted=True)
    clock[0] = owner_lease.expires_at + 1.0
    observer_rpc = SessionRegistry()
    observer_rpc.add_session(
        "session-1", active=False, history=[_receipt(identity, attempt.execution_generation)])
    successor, _lease = _successor(db, tmp_path, observer_rpc, lambda: clock[0])

    result = successor.cancel(identity, cancel_id="stop-that-lost-the-race")

    assert result["status"] == "settled"
    assert observer_rpc.count("interrupt") == 0


def test_an_accepted_interrupt_does_not_acknowledge_a_still_running_turn(db: Path, tmp_path: Path):
    """`interrupted: true` reports the request was accepted, not that the run thread died."""
    clock = [time.time()]
    identity = _identity()
    owner_lease, attempt = _foreign_attempt(db, lambda: clock[0], identity, admitted=True)
    clock[0] = owner_lease.expires_at + 1.0
    observer_rpc = SessionRegistry(interrupt_keeps_running=True)
    observer_rpc.add_session(
        "session-1", active=True, task_id=identity.task_id,
        execution_generation=attempt.execution_generation)
    successor, _lease = _successor(db, tmp_path, observer_rpc, lambda: clock[0])

    result = successor.cancel(identity, cancel_id="stop-a-running-turn")

    assert observer_rpc.count("interrupt") == 1
    assert result["status"] != "cancelled"
    assert state.get_task(db, identity)["status"] != "cancelled"

    # The turn ends leaving no receipt. A successor still has nothing that proves this attempt
    # stopped, so it keeps reporting uncertainty rather than inventing an acknowledgement; only a
    # terminal receipt (or the owning process itself) can close it.
    observer_rpc.states["session-1"]["active"] = False
    assert successor.cancel(identity, cancel_id="stop-a-running-turn")["status"] == "indeterminate"


def test_the_store_refuses_a_direct_cancellation_of_admitted_deferred_work(db: Path, tmp_path: Path):
    """The invariant lives in the transaction, not in a caller's routing read."""
    clock = [time.time()]
    identity = _identity()
    owner_lease, attempt = _foreign_attempt(db, lambda: clock[0], identity, admitted=True)
    clock[0] = owner_lease.expires_at + 1.0
    observer_rpc = SessionRegistry()
    observer_rpc.add_session("session-1", active=False)
    _successor(db, tmp_path, observer_rpc, lambda: clock[0])
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=BINDING.gateway_id, authority_epoch=BINDING.authority_epoch,
        process_generation="another-process", ttl_seconds=100.0, clock=lambda: clock[0] + 200)
    state.defer_indeterminate_task(
        db, identity, lease, expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation, reason="member_unavailable",
        clock=lambda: clock[0] + 200)
    deferred = state.get_task(db, identity)
    assert (deferred["status"], deferred["admitted_at"] is None) == ("deferred", False)

    with pytest.raises(state.InvalidTaskTransitionError, match="two-phase"):
        state.cancel_task(
            db, identity, cancel_id="direct-store-stop",
            expected_cancel_generation=deferred["cancel_generation"], clock=lambda: clock[0])

    assert state.get_task(db, identity)["status"] == "deferred"


def test_direct_cancellation_of_unadmitted_work_stays_available_and_idempotent(db: Path):
    """Queued and never-dispatched deferred work keeps the cheap route, replay included."""
    clock = [time.time()]
    identity = _identity()
    state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "Fixture only", "source_event_seq": 1},
        clock=lambda: clock[0])

    first = state.cancel_task(
        db, identity, cancel_id="stop-queued", expected_cancel_generation=0,
        clock=lambda: clock[0])
    repeated = state.cancel_task(
        db, identity, cancel_id="stop-queued", expected_cancel_generation=0,
        clock=lambda: clock[0])

    assert first["status"] == "cancelled"
    assert repeated["status"] == "cancelled"
    assert repeated["cancel_generation"] == first["cancel_generation"]
