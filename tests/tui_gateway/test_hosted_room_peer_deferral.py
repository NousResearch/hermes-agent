"""A member turn its gateway never received: deferred with proof, never unknown work.

Manual driver cycles over real SQLite state; only the member transport is inert.
"""
from contextlib import nullcontext
from dataclasses import asdict, replace
import time

import pytest

from gateway import hosted_room_driver as state, hosted_rooms
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING, FakeSessionRPC, _admit, _identity, _runtime, db)
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


class RPC:
    def __init__(self, mode):
        self.mode, self.calls, self.callback = mode, [], None

    def resolve_exact(self, **kwargs):
        return {"session_id": kwargs["profile"]}

    def resume(self, **kwargs):
        return {"session_id": kwargs["session_id"]}

    def submit(self, **kwargs):
        self.calls.append((kwargs["profile"], kwargs["execution_generation"]))
        if kwargs["profile"] == "peer" and self.mode != "repaired":
            if self.callback:
                self.callback()
            raise PeerRunsHTTPError("inert refusal", retryable=True,
                                    not_admitted=self.mode != "unknown", ambiguous=self.mode == "unknown")
        kwargs["on_terminal"]({"status": "settled", "text": "one reply"})
        return {"accepted": True}

    def info(self, **kwargs):
        return {"active": False}

    def history(self, **kwargs):
        return []


def fixture(tmp_path, *, opt_in=True, member=True, mode="unavailable"):
    db = tmp_path / "state.db"
    binding = HostedRoomBinding("room", "gateway", 1)
    now = [time.time()]
    hosted_rooms.create_room(db, room_id="room", name="Room", authority_gateway_id="gateway",
        members=[{"profile": "peer", "handle": "peer"}, {"profile": "local", "handle": "local"}])
    rpc, publications = RPC(mode), []
    runtime = HostedRoomRuntime(db_path=db, rooms=[binding], rpc=rpc,
        turn_lock=lambda _: nullcontext(), clock=lambda: now[0],
        unavailable_retry_min_seconds=2, unavailable_retry_max_seconds=8,
        publish_terminal=lambda binding, task: publications.append((task["identity"], task["status"])),
        defer_not_admitted_members=opt_in)

    def admit(profile, suffix):
        identity = state.TaskIdentity("room", suffix, "thread-" + suffix, "turn-" + suffix)
        payload = dict(target_profile=profile, prompt="unchanged " + suffix, source_event_seq=1)
        if member:
            payload["target_member_id"] = profile
        state.admit_task(db, identity, payload=payload, clock=lambda: now[0])
        return identity
    peer = admit("peer", "a-peer")
    return db, binding, now, rpc, runtime, publications, peer, admit


def test_proven_nonadmission_releases_later_turns_without_spinning(tmp_path):
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(tmp_path)
    original = state.get_task(db, peer)["payload"]
    local = admit("local", "b-local")
    runtime._run_cycle()
    runtime._run_cycle()
    assert state.get_task(db, peer)["status"] == "deferred"
    assert (peer, "deferred") in publications
    assert state.get_task(db, local)["status"] == "settled"
    assert rpc.calls == [("peer", 1), ("local", 1)]
    # Later turns keep running during the member's backoff; the deferred one is not retried.
    later = admit("local", "c-local")
    for _ in range(3):
        runtime._run_cycle()
    assert state.get_task(db, later)["status"] == "settled"
    assert rpc.calls == [("peer", 1), ("local", 1), ("local", 1)]
    runtime.retry_indeterminate(peer)  # the explicit retry; nothing requeues it automatically
    rpc.mode = "repaired"
    runtime._run_cycle()
    assert len(rpc.calls) == 3  # an explicit retry still waits out the member's backoff
    now[0] += 3
    runtime._run_cycle()
    assert state.get_task(db, peer)["status"] == "settled"
    assert state.get_task(db, peer)["payload"] == original
    assert rpc.calls == [("peer", 1), ("local", 1), ("local", 1), ("peer", 2)]
    assert publications.count((local, "settled")) == 1


@pytest.mark.parametrize("fence", ["unknown", "cancel", "lease"])
def test_unknown_work_or_a_lost_fence_never_defers(tmp_path, fence):
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(
        tmp_path, mode="unknown" if fence == "unknown" else "unavailable")
    local = admit("local", "b-local")
    if fence == "cancel":
        rpc.callback = lambda: state.begin_task_cancel(
            db, peer, cancel_id="cancel", expected_cancel_generation=0, clock=lambda: now[0])
    elif fence == "lease":
        rpc.callback = lambda: now.__setitem__(0, now[0] + runtime.lease_ttl_seconds + 1)
    runtime._run_cycle()
    task = state.get_task(db, peer)
    assert task["status"] not in {"queued", "deferred", "settled"}
    assert task["execution_generation"] == 1
    assert runtime._unavailable_route_retries == {}
    assert publications == []
    assert state.get_task(db, local)["status"] == "queued"
    assert rpc.calls == [("peer", 1)]


@pytest.mark.parametrize("opt_in,member", [(False, True), (True, False)], ids=["not-opted-in", "not-a-member"])
def test_otherwise_the_turn_requeues_in_order_with_bounded_backoff(tmp_path, opt_in, member):
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(tmp_path, opt_in=opt_in, member=member)
    local = admit("local", "b-local")
    for _ in range(3):
        runtime._run_cycle()
    assert state.get_task(db, peer)["status"] == state.get_task(db, local)["status"] == "queued"
    assert rpc.calls == [("peer", 1)]
    rpc.mode = "repaired"
    now[0] += 3
    runtime._run_cycle()
    assert rpc.calls == [("peer", 1), ("peer", 2), ("local", 1)]
    assert state.get_task(db, local)["status"] == "settled"


def test_deferral_replays_only_for_the_exact_running_attempt(tmp_path):
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(tmp_path)
    lease = runtime._ensure_lease(binding)
    attempt = state.start_task(db, peer, lease, expected_cancel_generation=0, clock=lambda: now[0])
    first = state.defer_not_admitted_task(db, attempt, reason="member_unavailable", clock=lambda: now[0])
    replay = state.defer_not_admitted_task(db, attempt, reason="member_unavailable", clock=lambda: now[0])
    assert replay["idempotent"] is True
    assert {k: v for k, v in replay.items() if k != "idempotent"} == {
        k: v for k, v in first.items() if k != "idempotent"}
    for changed in (replace(attempt, execution_generation=2), replace(attempt, cancel_generation=1),
                    replace(attempt, lease=replace(lease, process_generation="foreign"))):
        with pytest.raises((state.StaleLeaseError, state.StaleTaskError)):
            state.defer_not_admitted_task(db, changed, reason="member_unavailable", clock=lambda: now[0])
    assert state.get_task(db, peer)["status"] == "deferred"


def test_only_the_producer_writes_the_proof_and_a_requeue_consumes_it(tmp_path):
    db, binding, now, rpc, runtime, publications, identity, admit = fixture(tmp_path)
    original = state.get_task(db, identity)
    runtime._run_cycle()
    saved = state.get_task(db, identity)
    proof = (saved["result"] or {}).get("nonadmission")
    assert isinstance(proof, dict), saved
    assert proof["disposition"] == "proven_nonadmission"
    assert proof["identity"] == asdict(identity)
    assert proof["execution_generation"] == saved["execution_generation"] == 1
    assert proof["cancel_generation"] == saved["cancel_generation"] == 0
    for key in ("run_gateway_id", "run_process_generation", "run_lease_generation"):
        assert proof[key] == saved[key]
    assert state.is_proven_nonadmission(saved)
    runtime.retry_indeterminate(identity)
    queued = state.get_task(db, identity)
    assert queued["execution_generation"] == 1 and queued["result"] is None
    assert not state.is_proven_nonadmission(queued)
    assert queued["payload"] == original["payload"]


@pytest.mark.parametrize("corruption", [
    "legacy", "boolean", "disposition", "identity", "extra-field", "epoch", "generation", "cancel",
    "lease", "run-gateway", "run-process", "status"])
def test_a_malformed_or_mismatched_proof_is_not_a_proof(tmp_path, corruption):
    db, binding, now, rpc, runtime, publications, identity, admit = fixture(tmp_path)
    runtime._run_cycle()
    task = state.get_task(db, identity)
    assert state.is_proven_nonadmission(task)
    proof = task["result"]["nonadmission"]
    corruptions = {
        "legacy": lambda: task["result"].pop("nonadmission"),
        "boolean": lambda: task["result"].update(nonadmission=True),
        "disposition": lambda: proof.update(disposition="unknown"),
        "identity": lambda: proof.update(identity={**proof["identity"], "turn_id": "other"}),
        "extra-field": lambda: proof.update(extra=True),
        "epoch": lambda: proof.update(authority_epoch=True),
        "generation": lambda: task.update(execution_generation=2),
        "cancel": lambda: proof.update(cancel_generation=1),
        "lease": lambda: task.update(run_lease_generation=task["run_lease_generation"] + 1),
        "run-gateway": lambda: proof.update(run_gateway_id="other-gateway"),
        "run-process": lambda: task.update(run_process_generation="other-process"),
        "status": lambda: task.update(status="queued"),
    }
    corruptions[corruption]()
    assert not state.is_proven_nonadmission(task)


def test_an_unknown_attempt_deferred_by_recovery_has_no_proof(tmp_path):
    db, binding, now, rpc, runtime, publications, identity, admit = fixture(tmp_path)
    lease = runtime._ensure_lease(binding)
    state.start_task(db, identity, lease, expected_cancel_generation=0, clock=lambda: now[0])
    now[0] += runtime.lease_ttl_seconds + 1
    successor = state.acquire_lease(db, room_id=binding.room_id, gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="successor", ttl_seconds=60,
        clock=lambda: now[0])
    state.recover_room(db, successor, clock=lambda: now[0])
    saved = state.defer_indeterminate_task(db, identity, successor, expected_execution_generation=1,
        expected_cancel_generation=0, reason="member_unavailable", clock=lambda: now[0])
    assert saved["result"] == {"reason": "member_unavailable", "retryable": True}
    assert not state.is_proven_nonadmission(saved)


class PreflightFailureRPC(FakeSessionRPC):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def submit(self, **_kwargs):
        self.attempts += 1
        failure = PeerRunsHTTPError("refresh refused", status_code=401, error_code="invalid_room_grant")
        failure.dispatch_not_attempted = True
        raise failure


@pytest.mark.parametrize("opt_in", [False, True])
def test_a_failure_before_sending_proves_only_a_fresh_generation(db, opt_in):
    identity = _identity()
    _admit(db, identity)
    rpc = PreflightFailureRPC()
    runtime = _runtime(db, rpc, defer_not_admitted_members=opt_in)
    runtime._run_cycle()
    task = state.get_task(db, identity)
    assert rpc.attempts == 1
    assert task["execution_generation"] == 1
    # The shared fixture's turns name no member, so even an opted-in runtime requeues them.
    assert task["status"] == "queued"


@pytest.mark.parametrize("stale_field", ["status", "generation"])
def test_a_reused_snapshot_does_not_prove_a_new_generation(db, stale_field):
    identity = _identity()
    _admit(db, identity)
    rpc = PreflightFailureRPC()
    runtime = _runtime(db, rpc)
    snapshot = state.get_task(db, identity)
    lease = runtime._ensure_lease(BINDING)
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=time.time)
    if stale_field == "status":
        snapshot["status"] = "running"
    else:
        snapshot["execution_generation"] = attempt.execution_generation
    runtime._execute_attempt(BINDING, snapshot, attempt)
    task = state.get_task(db, identity)
    assert rpc.attempts == 1
    assert task["status"] == "running"
    assert task["execution_generation"] == 1
    assert task["run_gateway_id"] == BINDING.gateway_id


def test_a_member_turn_that_was_never_sent_is_deferred_with_its_proof(tmp_path):
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(tmp_path)

    def refused_before_sending(**kwargs):
        rpc.calls.append((kwargs["profile"], kwargs["execution_generation"]))
        failure = RuntimeError("the grant could not be refreshed")
        failure.dispatch_not_attempted = True
        raise failure
    rpc.submit = refused_before_sending
    runtime._run_cycle()
    saved = state.get_task(db, peer)
    assert saved["status"] == "deferred" and state.is_proven_nonadmission(saved)
    assert publications == [(peer, "deferred")]
