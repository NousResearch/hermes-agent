"""Native Phase 2: exact-owner-death Retry and proof-guarded Stop.

Two coverage kinds, kept explicitly separate:

* **Separate-registry simulation** — an owner and an observer ``HostedRoomRuntime``, each with its
  OWN ``FakeSessionRPC`` registry, over one real temporary task store and the real profile turn
  lock. This is the pattern the parent's foreign-retry regression established. It proves store and
  runtime decisions; it does NOT exercise two native backends, and every death verdict in it is
  *simulated process-death evidence* injected at the probe seam, never a real process ending.
* **Installed-native capture** — the accepted Phase 1 fixture, unchanged, one real backend, for
  what only real handlers can show: that a complete descriptor is stamped with ``admitted_at``
  before any prompt is dispatched, and that an invalid target refuses instead of submitting.

Real disposable-subprocess owner death and real foreign native observation remain separate
acceptance gates.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_room_owner_probe as probe
from gateway import hosted_rooms
from tools.bot_relay import acquire_turn_lock
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime, room_session_title


REPO = Path(__file__).resolve().parents[2]
ROOM_ID = "phase2-room"
PROFILE = "ops"
WAIT = 10.0


def _runtime_helpers():
    """The existing runtime fakes, loaded the way the parent regressions load them."""
    spec = importlib.util.spec_from_file_location(
        "_phase2_runtime_fakes", REPO / "tests/tui_gateway/test_hosted_room_driver_runtime.py")
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def _descriptor(**overrides: Any) -> dict[str, Any]:
    """A complete, syntactically valid Linux owner descriptor for the local domain."""
    domain = probe.capture_local_domain()
    assert domain is not None, "these tests require the Linux descriptor domain"
    return {
        **domain, "home": "/tmp/phase2-home", "profile": PROFILE,
        "runtime_session_id": "runtime-1", "stored_session_key": "stored-1", **overrides}


class OwnerAwareRPC:
    """A `FakeSessionRPC` that also implements the optional native capture capability."""

    def __init__(self, helpers, **kwargs: Any) -> None:
        self._inner = helpers.FakeSessionRPC(**kwargs)
        self.identity: dict[str, Any] | None = _descriptor()
        self.identity_error: Exception | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def execution_identity(self, *, profile: str, session_id: str):
        if self.identity_error is not None:
            raise self.identity_error
        if self.identity is None:
            return None
        return {**self.identity, "profile": profile, "runtime_session_id": session_id}


@pytest.fixture
def separate_registries(tmp_path, monkeypatch):
    """Owner and observer runtimes with distinct RPC registries over one real store."""
    helpers = _runtime_helpers()
    owner_rpc = OwnerAwareRPC(helpers, auto_complete=False)
    observer_rpc = OwnerAwareRPC(helpers, auto_complete=False)
    db = tmp_path / "state.db"
    binding = HostedRoomBinding(ROOM_ID, "phase2-gateway", 1)
    identity = state.TaskIdentity(ROOM_ID, "phase2-task", "thread-1", "turn-1")
    clock_value = [time.time()]

    def clock() -> float:
        return clock_value[0]

    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Phase 2", authority_gateway_id=binding.gateway_id,
        members=[{"profile": PROFILE, "handle": PROFILE}], now=clock())
    task = state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "offline fixture", "source_event_seq": 1},
        clock=clock)

    def turn_lock(member: str):
        return acquire_turn_lock(tmp_path / "locks", member, timeout_seconds=WAIT)

    owner = HostedRoomRuntime(
        db_path=db, rooms=[binding], rpc=owner_rpc, turn_lock=turn_lock, clock=clock,
        lease_ttl_seconds=1.0, active_poll_interval_seconds=0.01)
    observer = HostedRoomRuntime(
        db_path=db, rooms=[binding], rpc=observer_rpc, turn_lock=turn_lock, clock=clock,
        lease_ttl_seconds=100.0, indeterminate_defer_seconds=5.0)
    lease = owner._ensure_lease(binding)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=clock)
    worker = threading.Thread(
        target=owner._execute_attempt, args=(binding, task, attempt), daemon=True)
    worker.start()
    try:
        assert owner_rpc.submitted.wait(WAIT), "the owner runtime never reached admission"
        clock_value[0] = lease.expires_at + 1.0
        owner._wake.set()
        worker.join(WAIT)
        assert not worker.is_alive(), "the owner runtime never left its attempt"
        # A cold successor: same durable session and title, its OWN running registry.
        observer_sid = observer_rpc.add_session(
            profile=PROFILE, title=room_session_title(ROOM_ID))
        assert observer_sid == next(iter(owner_rpc.states))
        assert owner_rpc.states is not observer_rpc.states
        successor_lease = observer._ensure_lease(binding)
        state.recover_room(db, successor_lease, clock=clock)
        assert state.get_task(db, identity)["status"] == "indeterminate"
        yield {
            "db": db, "identity": identity, "binding": binding, "observer": observer,
            "owner_rpc": owner_rpc, "observer_rpc": observer_rpc, "clock": clock_value,
            "monkeypatch": monkeypatch}
    finally:
        owner._stop.set()
        owner._wake.set()
        worker.join(WAIT)


def _verdict(env, verdict: str) -> None:
    """Inject the incarnation verdict at the probe seam (simulated process-death evidence)."""
    import tui_gateway.hosted_room_driver as runtime_module

    env["monkeypatch"].setattr(
        runtime_module, "probe_owner_incarnation", lambda descriptor: verdict)


def test_a_descriptor_is_stamped_with_the_admission_it_belongs_to(separate_registries):
    """Capture is atomic with the admission fence and carries the attempt's own coordinates."""
    task = state.get_task(separate_registries["db"], separate_registries["identity"])

    descriptor = task["owner_descriptor"]

    assert task["admitted_at"] is not None
    assert isinstance(descriptor, dict), "an admitted attempt kept no owner descriptor"
    assert descriptor["execution_generation"] == task["execution_generation"]
    assert descriptor["cancel_generation_at_admission"] == 0
    assert (descriptor["run_gateway_id"], descriptor["run_process_generation"]) == (
        task["run_gateway_id"], task["run_process_generation"])
    assert descriptor["run_lease_generation"] == task["run_lease_generation"]
    assert descriptor["task_id"] == separate_registries["identity"].task_id


def test_retry_refuses_while_the_recorded_owner_incarnation_is_alive(separate_registries):
    """Process-generation difference is not death, and a live owner keeps its attempt."""
    _verdict(separate_registries, "alive")

    with pytest.raises(state.DriverStateError):
        separate_registries["observer"].retry_indeterminate(separate_registries["identity"])

    task = state.get_task(separate_registries["db"], separate_registries["identity"])
    assert task["status"] == "indeterminate"
    assert task["owner_descriptor"] is not None


def test_retry_requeues_exactly_once_when_the_owner_incarnation_is_proven_ended(
        separate_registries):
    """The positive recovery: proven cessation requeues the attempt and clears its descriptor."""
    _verdict(separate_registries, "ended")

    retried = separate_registries["observer"].retry_indeterminate(
        separate_registries["identity"])

    assert retried["status"] == "queued"
    task = state.get_task(separate_registries["db"], separate_registries["identity"])
    assert task["status"] == "queued"
    assert task["admitted_at"] is None
    assert task["owner_descriptor"] is None
    assert task["run_process_generation"] is None
    # The proof was consumed with the row it named: a second Retry has nothing left to recover.
    with pytest.raises(state.DriverStateError):
        separate_registries["observer"].retry_indeterminate(
            separate_registries["identity"])


def test_the_positive_retry_regression_rejects_an_unavailable_death_proof(separate_registries):
    """Mutation / NEGATIVE CONTROL for the positive case above — not a historical red.

    There was no behavioural red for this contract: its first run was a missing-module collection
    error. So the positive regression is instead shown to be discriminating: with the verdict still
    ``ended``, the proof-construction seam is mutated to yield nothing, and the identical Retry that
    requeues above must now refuse and leave the attempt exactly where it was. This exercises the
    storage guard's own refusal, not the probe's.
    """
    import tui_gateway.hosted_room_driver as runtime_module

    _verdict(separate_registries, "ended")
    separate_registries["monkeypatch"].setattr(
        runtime_module.HostedRoomRuntime, "_owner_death_proof", lambda self, task: None)

    with pytest.raises(state.DriverStateError):
        separate_registries["observer"].retry_indeterminate(separate_registries["identity"])

    task = state.get_task(separate_registries["db"], separate_registries["identity"])
    assert task["status"] == "indeterminate", "an unproven attempt was recovered anyway"
    assert task["admitted_at"] is not None
    assert task["owner_descriptor"] is not None


def test_a_corrupt_descriptor_never_regains_the_same_generation_bypass(separate_registries):
    """A descriptor-bearing row whose stored value is unreadable is refused, not treated as legacy.

    The legacy branch is decided on the raw NULL column: deciding it on the decoded value would
    hand exactly these rows back the equal-process-generation requeue bypass.
    """
    db, identity = separate_registries["db"], separate_registries["identity"]
    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_driver_tasks SET owner_descriptor='not valid json' "
            "WHERE room_id=? AND task_id=?", (identity.room_id, identity.task_id))
    _verdict(separate_registries, "ended")

    with pytest.raises(state.DriverStateError):
        separate_registries["observer"].retry_indeterminate(identity)

    assert state.get_task(db, identity)["status"] == "indeterminate"


def test_an_unknown_verdict_never_promotes_to_recovery(separate_registries):
    """Fail closed: an unreadable incarnation is not evidence that the attempt stopped."""
    _verdict(separate_registries, "unknown")

    with pytest.raises(state.DriverStateError):
        separate_registries["observer"].retry_indeterminate(separate_registries["identity"])

    assert state.get_task(
        separate_registries["db"], separate_registries["identity"])["status"] == "indeterminate"


# --------------------------------------------------------------- Stop outcomes

def _begin_stop(env, cancel_id: str) -> dict:
    """Move the uncertain attempt to `stopping` through the real transition."""
    task = state.get_task(env["db"], env["identity"])
    state.begin_task_cancel(
        env["db"], env["identity"], cancel_id=cancel_id,
        expected_cancel_generation=int(task["cancel_generation"]), clock=lambda: env["clock"][0])
    return state.get_task(env["db"], env["identity"])


def test_a_live_owner_keeps_its_stop_provisional_and_is_never_cancelled(separate_registries):
    """POSITIVE control for the ordinary route: death is an extra authority, not a prerequisite.

    With the owner incarnation alive and no native acknowledgement available, the Stop must resolve
    as uncertain exactly as it does today -- never as a cancellation.
    """
    env = separate_registries
    _verdict(env, "alive")
    task = _begin_stop(env, "user-stop")
    lease = env["observer"]._ensure_lease(env["binding"])

    assert env["observer"]._finish_stop(env["binding"], task, lease) is True

    settled = state.get_task(env["db"], env["identity"])
    assert settled["status"] == "indeterminate", "a live owner's Stop was claimed as cancelled"
    assert settled["owner_descriptor"] is not None


def test_a_user_stop_with_a_dead_owner_cancels_under_its_recorded_identity(separate_registries):
    env = separate_registries
    _verdict(env, "ended")
    task = _begin_stop(env, "user-stop")
    lease = env["observer"]._ensure_lease(env["binding"])

    assert env["observer"]._finish_stop(env["binding"], task, lease) is True

    settled = state.get_task(env["db"], env["identity"])
    assert settled["status"] == "cancelled"
    assert settled["cancel_id"] == "user-stop", "the recorded stop identity was replaced"


def test_a_deadline_stop_with_a_dead_owner_stays_failed_with_its_own_reason(separate_registries):
    """Cessation evidence must not relabel a timeout as a user cancellation."""
    env = separate_registries
    _verdict(env, "ended")
    generation = state.get_task(env["db"], env["identity"])["execution_generation"]
    task = _begin_stop(env, f"deadline:{generation}")
    lease = env["observer"]._ensure_lease(env["binding"])

    assert env["observer"]._finish_stop(env["binding"], task, lease) is True

    settled = state.get_task(env["db"], env["identity"])
    assert settled["status"] == "failed", "a deadline stop was relabelled as a cancellation"
    assert settled["result"]["reason_code"] == "turn_deadline_exceeded"


def test_a_stopped_dead_attempt_is_resolved_by_retry_rather_than_replayed(separate_registries):
    """Retry of work the user stopped resolves the stop it carries; it does not resubmit it."""
    env = separate_registries
    _verdict(env, "ended")
    stopping = _begin_stop(env, "user-stop")
    state.mark_stop_uncertain(
        env["db"], env["identity"], env["observer"]._ensure_lease(env["binding"]),
        expected_execution_generation=int(stopping["execution_generation"]),
        expected_cancel_generation=int(stopping["cancel_generation"]),
        clock=lambda: env["clock"][0])

    resolved = env["observer"].retry_indeterminate(env["identity"])

    assert resolved["status"] == "cancelled"
    assert resolved["cancel_id"] == "user-stop"


# -------------------------------------------------- installed-native capture
# One real backend, real handlers, real temporary SQLite. Only these cases can show what the
# actual adapter returns; the simulations above never touch a native session record.

_PHASE1 = importlib.util.spec_from_file_location(
    "_phase2_native_fixture", REPO / "tests/tui_gateway/test_hosted_room_native_phase1.py")
_phase1 = importlib.util.module_from_spec(_PHASE1)
_PHASE1.loader.exec_module(_phase1)
native = _phase1.native  # the accepted Phase 1 fixture, unchanged


def test_the_native_adapter_captures_this_process_for_an_exact_local_session(native):
    """The real capture: this process's incarnation, this record's home, both identifiers."""
    from tui_gateway.hosted_room_driver import room_session_title

    sid = native.create_session(room_session_title(ROOM_ID))
    rpc = native.rpc()

    captured = rpc.execution_identity(profile="default", session_id=sid)

    if captured is None:
        pytest.skip("this host cannot supply the Linux descriptor domain")
    assert captured["pid"] == __import__("os").getpid()
    assert captured["runtime_session_id"] == sid
    assert captured["stored_session_key"] == native.record(sid)["session_key"]
    assert captured["stored_session_key"] != sid, "the durable key is not the runtime handle"
    assert captured["profile"] == "default"
    assert probe.probe_owner_incarnation(captured) == "alive"


def test_the_native_adapter_refuses_an_invalid_target_instead_of_aliasing_it(native):
    """`_profile_home` answering None is not validation: a wrong profile must fail, not alias."""
    from tui_gateway.hosted_room_driver import room_session_title
    from tui_gateway.hosted_room_server_rpc import HostedRoomSessionError

    sid = native.create_session(room_session_title(ROOM_ID))
    rpc = native.rpc()

    with pytest.raises(HostedRoomSessionError):
        rpc.execution_identity(profile="default", session_id="no-such-session")
    with pytest.raises(HostedRoomSessionError):
        rpc.execution_identity(profile="a-profile-that-does-not-exist", session_id=sid)


def test_an_unsupported_transport_still_submits_with_no_descriptor(separate_registries):
    """Optionality is real: the capability is discovered, never required of the Protocol.

    `FakeSessionRPC` implements `InternalSessionRPC` structurally and has no
    `execution_identity` -- exactly like every peer transport -- and must keep working.
    """
    helpers = _runtime_helpers()
    plain = helpers.FakeSessionRPC(auto_complete=False)

    assert not hasattr(plain, "execution_identity")
    runtime = separate_registries["observer"]
    assert runtime._capture_owner_descriptor(plain, PROFILE, "session-1") is None


def test_an_exact_receipt_still_wins_over_a_dead_owner(separate_registries):
    """Receipt-first is unchanged: death is consulted only when nothing else can answer."""
    env = separate_registries
    _verdict(env, "ended")
    task = state.get_task(env["db"], env["identity"])
    session_id = next(iter(env["observer_rpc"].states))
    env["observer_rpc"].states[session_id]["history"].append({
        "role": "assistant", "task_id": env["identity"].task_id,
        "execution_generation": task["execution_generation"], "status": "settled",
        "message_id": "reply:recovered", "content": "the owner finished before it died"})

    resolved = env["observer"].retry_indeterminate(env["identity"])

    assert resolved["status"] == "settled", "a real receipt lost to the death route"
    assert resolved["result"]["text"] == "the owner finished before it died"
