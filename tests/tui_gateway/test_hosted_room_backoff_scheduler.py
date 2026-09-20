"""Manual SQLite driver invariants; no execution workers or real transports.

Member deferral/sibling progress preserves David Dudok de Wit's historical
691bb08a5cd310fffc5f3d01653dc93f394fc080 driver behavior. Generic queued retry
continues to use its existing bounded backoff; unknown work is never NEW work.
"""
from contextlib import nullcontext
import time

import pytest

from gateway import hosted_room_driver as state, hosted_rooms
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


def fixture(tmp_path, *, member, mode="unavailable"):
    db = tmp_path / "state.db"
    binding = HostedRoomBinding("room", "gateway", 1)
    now = [time.time()]
    hosted_rooms.create_room(db, room_id="room", name="Room", authority_gateway_id="gateway",
        members=[{"profile": "peer", "handle": "peer"}, {"profile": "local", "handle": "local"}])
    rpc, publications = RPC(mode), []
    runtime = HostedRoomRuntime(db_path=db, rooms=[binding], rpc=rpc,
        turn_lock=lambda _: nullcontext(), clock=lambda: now[0],
        unavailable_retry_min_seconds=2, unavailable_retry_max_seconds=8,
        publish_terminal=lambda binding, task: publications.append((task["identity"], task["status"])))
    def admit(profile, suffix):
        identity = state.TaskIdentity("room", suffix, "thread-" + suffix, "turn-" + suffix)
        payload = dict(target_profile=profile, prompt="unchanged " + suffix, source_event_seq=1)
        if member:
            payload["target_member_id"] = profile
        state.admit_task(db, identity, payload=payload, clock=lambda: now[0])
        return identity
    peer = admit("peer", "a-peer")
    return db, binding, now, rpc, runtime, publications, peer, admit


def test_proven_nonadmission_releases_healthy_siblings_without_spinning(tmp_path):
    member = True
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(tmp_path, member=member)
    original = state.get_task(db, peer)["payload"]
    local = admit("local", "b-local")
    runtime._run_cycle()
    runtime._run_cycle()
    assert state.get_task(db, peer)["status"] == ("deferred" if member else "queued")
    assert state.get_task(db, local)["status"] == "settled"
    assert rpc.calls == [("peer", 1), ("local", 1)]
    # Newly queued siblings remain reachable on later ticks during the backoff.
    later = admit("local", "c-local")
    for _ in range(3):
        runtime._run_cycle()
    assert state.get_task(db, later)["status"] == "settled"
    assert rpc.calls == [("peer", 1), ("local", 1), ("local", 1)]
    if member:
        assert (peer, "deferred") in publications
        runtime.retry_indeterminate(peer)  # existing explicit retry, not automatic NEW
    rpc.mode = "repaired"
    runtime._run_cycle()
    assert len(rpc.calls) == 3
    now[0] += 3
    runtime._run_cycle()
    assert state.get_task(db, peer)["status"] == "settled"
    assert state.get_task(db, peer)["payload"] == original
    assert rpc.calls == [("peer", 1), ("local", 1), ("local", 1), ("peer", 2)]
    assert publications.count((local, "settled")) == 1


@pytest.mark.parametrize("fence", ["unknown", "cancel", "lease"])
def test_progress_does_not_bypass_unknown_or_lost_fences(tmp_path, fence):
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(
        tmp_path, member=True, mode="unknown" if fence == "unknown" else "unavailable")
    local = admit("local", "b-local")
    if fence == "cancel":
        rpc.callback = lambda: state.begin_task_cancel(db, peer, cancel_id="cancel", expected_cancel_generation=0, clock=lambda: now[0])
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


@pytest.mark.parametrize("member", [False, True])
def test_runtime_without_policy_publisher_keeps_fifo_and_bounded_retry(tmp_path, member):
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(tmp_path, member=member)
    runtime.publish_terminal = None
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


def test_deferral_replay_requires_exact_running_attempt(tmp_path):
    from dataclasses import replace
    db, binding, now, rpc, runtime, publications, peer, admit = fixture(tmp_path, member=True)
    lease = runtime._ensure_lease(binding)
    attempt = state.start_task(db, peer, lease, expected_cancel_generation=0, clock=lambda: now[0])
    first = state.defer_not_admitted_task(db, attempt, reason="member_unavailable", clock=lambda: now[0])
    replay = state.defer_not_admitted_task(db, attempt, reason="member_unavailable", clock=lambda: now[0])
    assert replay["idempotent"] is True
    assert {k: v for k, v in replay.items() if k != "idempotent"} == {k: v for k, v in first.items() if k != "idempotent"}
    for changed in (replace(attempt, execution_generation=2), replace(attempt, cancel_generation=1),
                    replace(attempt, lease=replace(lease, process_generation="foreign"))):
        with pytest.raises((state.StaleLeaseError, state.StaleTaskError)):
            state.defer_not_admitted_task(db, changed, reason="member_unavailable", clock=lambda: now[0])
    assert state.get_task(db, peer)["status"] == "deferred"
