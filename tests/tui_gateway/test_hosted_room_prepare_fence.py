"""Preparation must fence a stopped turn before unlocking, without peer I/O."""
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from gateway.session_hosted_service import CanonicalHostedRoomService
from persistence.gateway.hosted_rooms import InvalidTaskTransitionError, StaleTaskError
from persistence_assembly import open_persistence, profile_scope
from tests.tui_gateway.test_hosted_room_peer_recovery import _peer_service
from tests.tui_gateway.test_hosted_room_service import _FakePeerClient, _FakeRPC
from tests.tui_gateway.test_hosted_room_approval_lifetime import (
    close_storage_handles, pending_approval as pending_approval_fixture)

pending_approval = pending_approval_fixture


class _Peer(_FakePeerClient):
    def __init__(self):
        super().__init__()
        self.calls, self.acknowledge = [], False

    def _record(self, operation):
        self.calls.append((operation, self.service._policy_lock._is_owned()))

    def recover_dispatch(self, **kwargs):
        self._record("recover")
        return {"status": "accepted"}

    def prepare(self, **kwargs):
        self._record("prepare")
        return self.session

    def history(self, **kwargs):
        self._record("history")
        return []

    def status(self, **kwargs):
        self._record("status")
        return {"active": True, "task_id": kwargs["task_id"],
                "execution_generation": kwargs["execution_generation"]}

    def stop_receipt(self, **kwargs):
        self._record("stop")
        return {"interrupted": self.acknowledge}


class _HandoffLock:
    """Let a waiting worker inspect durable state at the outermost release."""
    def __init__(self):
        self.lock, self.depth = threading.RLock(), threading.local()
        self.owner = threading.current_thread()
        self.waiting, self.done = threading.Event(), threading.Event()
        self.armed = False

    def _is_owned(self):
        return self.lock._is_owned()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            self.lock.acquire()
        self.depth.value = getattr(self.depth, "value", 0) + 1
        return self

    def __exit__(self, *exc):
        self.depth.value -= 1
        self.lock.release()
        if self.armed and not self.depth.value and threading.current_thread() is self.owner:
            assert self.done.wait(10), "waiting worker never inspected the fenced task"


@pytest.fixture
def preparation(tmp_path, monkeypatch, nested):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    peer = _Peer()
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seeded = _peer_service(db, peer)
        route = seeded.peer_routes[("room-1", "member-peer")]
        authority = SimpleNamespace(db=db, persistence=db, profile_id=str(tmp_path),
                                    epoch=db.runtime.begin_epoch(instance_id="test"))
        service = CanonicalHostedRoomService(authority, None)
        service.local_profiles = lambda: ("default",)
        service.peer_clients[("room-1", "member-peer")] = peer
        peer.service = service
        service.create_room(room_id="room-1", name="Preparation fencing", owner="alice", members=[
            {"member_id": "default", "profile": "default", "handle": "home"},
            {"member_id": "member-peer", "profile": "reviewer", "handle": "reviewer", "target": {
                "kind": "peer", "peer_id": "peer-review", "installation_id": "install-peer",
                "profile": "reviewer", "capability_digest": route.capability_digest}}])
        binding, = service.bindings()
        if nested:
            service.send(room_id="room-1", event_id="local", payload={"text": "@home first", "thread_id": "thread"})
            local, = driver.list_tasks(db, placement="profile", room_id="room-1")
            old = driver.acquire_lease(db, placement="profile", room_id="room-1", gateway_id=binding.gateway_id,
                authority_epoch=binding.authority_epoch, process_generation="old", ttl_seconds=1,
                clock=lambda: 1)
            driver.start_task(db, local["identity"], old, placement="profile", expected_cancel_generation=0, clock=lambda: 1)
            lease = service.runtime._ensure_lease(binding)
            driver.recover_room(db, lease, placement="profile", clock=service.runtime.clock)
            receipts = []
            local_rpc = SimpleNamespace(ref=SimpleNamespace(session_id="local-session"),
                discard=lambda **kwargs: receipts.append(kwargs) or {"discarded": True})
            resolve = service._resolve_member_transport
            monkeypatch.setattr(service, "_resolve_member_transport", lambda b, t:
                local_rpc if t["payload"]["target_member_id"] == "default" else resolve(b, t))

            def invoke():
                result = service.discard_room_task("room-1", member_id="default",
                    task_id=local["identity"].task_id, execution_generation=1)
                assert result["status"] == "cancelled" and result["cancel_id"] == "discard:1"
                assert receipts[0]["expected_task_id"] == local["identity"].task_id
        else:
            lease = service.runtime._ensure_lease(binding)
            invoke = lambda: service.prepare_room(binding)
        hosted_rooms.append_event(db, placement="profile", room_id="room-1", event_id="peer", kind="message.user",
            actor={"kind": "user", "id": "test"}, payload={"text": "@reviewer next", "thread_id": "thread"},
            authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch)
        service.runtime._wake.clear()
        yield SimpleNamespace(service=service, peer=peer, binding=binding, lease=lease, invoke=invoke)


@pytest.mark.parametrize("nested", [False, True], ids=["prepare", "discard"])
@pytest.mark.parametrize("running", [False, True], ids=["queued", "running"])
@pytest.mark.parametrize("stop_first", [False, True], ids=["post-admission", "pre-admission"])
def test_fenced_preparation_is_durable_before_unlock_and_worker_acknowledges(
        preparation, monkeypatch, running, stop_first):
    p = preparation
    service, db = p.service, p.service.store
    lock = service._policy_lock = _HandoffLock()
    admitted, task_ready = [], threading.Event()
    admit = driver.admit_task
    coordinator = CanonicalHostedRoomService(service.authority, None)

    def raced_admit(*args, **kwargs):
        if stop_first:
            assert coordinator.stop_room("room-1", cancel_id="racing-stop") == 0
        task = admit(*args, **kwargs)
        admitted.append(task)
        if running:
            driver.start_task(db, task["identity"], p.lease, placement="profile", expected_cancel_generation=0,
                              clock=service.runtime.clock)
        if not stop_first:
            hosted_rooms.request_room_stop(db, placement="profile", room_id="room-1", cancel_id="racing-stop",
                expected_gateway_id=p.binding.gateway_id, expected_epoch=p.binding.authority_epoch)
        lock.armed = True
        task_ready.set()
        assert lock.waiting.wait(10), "worker never reached preparation's held lock"
        return task

    def waiting_worker():
        try:
            assert task_ready.wait(10), "preparation never admitted the task"
            with lock:
                task = driver.get_task(db, admitted[0]["identity"], placement="profile")
                if not running:
                    with pytest.raises((InvalidTaskTransitionError, StaleTaskError)):
                        driver.start_task(db, task["identity"], p.lease, placement="profile",
                            expected_cancel_generation=0, clock=service.runtime.clock)
                return task
        finally:
            lock.done.set()

    monkeypatch.setattr(driver, "admit_task", raced_admit)
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(waiting_worker)
        p.invoke()
        observed = worker.result(timeout=10)
    assert p.peer.calls == [], "preparation performed peer I/O under its enclosing policy lock"
    assert observed["status"] == ("stopping" if running else "cancelled")
    fence = service._policy_snapshot(service._room("room-1")).stopped_through_seq
    assert observed["cancel_id"] == f"stop-fence:{fence}"
    assert observed["cancel_generation"] == 1
    assert service.runtime._wake.is_set()
    identity = observed["identity"]
    if running:
        # A second requester joins the first durable stop, including Stop's own scan.
        repeated = service.runtime.cancel(identity, cancel_id="another-request")
        assert repeated["status"] == "stopping"
        assert repeated["cancel_id"] == observed["cancel_id"]
        assert service.stop_room("room-1", cancel_id="operator-stop") == 1
        service.runtime._process_room(p.binding)
        assert driver.get_task(db, identity, placement="profile")["status"] == "stopping"
        p.peer.acknowledge = True
        service.runtime._process_room(p.binding)
        assert "stop" in [operation for operation, _ in p.peer.calls]
        assert not any(owned for _, owned in p.peer.calls)
    final = service.runtime.cancel(identity, cancel_id="replayed-after-ack")
    assert (final["status"], final["cancel_id"], final["cancel_generation"]) == (
        "cancelled", observed["cancel_id"], 1)


def test_local_cancel_still_acknowledges_synchronously(pending_approval):
    service, task, lease = pending_approval
    service.runtime.clock = lambda: lease.expires_at + 1
    service.rpc = service.runtime.rpc = _FakeRPC()
    for cancel_id in ("local-stop", "local-replay"):
        result = service.runtime.cancel(task["identity"], cancel_id=cancel_id)
        assert (result["status"], result["cancel_id"], result["cancel_generation"]) == ("cancelled", "local-stop", 1)
