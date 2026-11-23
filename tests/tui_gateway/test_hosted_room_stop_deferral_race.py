"""The worker can defer a turn behind its member's unconfirmed Stop while a room Stop or Disband is
already deciding about that turn. A turn that is deferred when the cancel commits needs the member's
confirmed Stop, whichever state the earlier snapshot showed; Disband refuses until it has it."""

from pathlib import Path

import pytest

from gateway import hosted_room_driver as driver
from persistence.gateway.hosted_rooms import RoomStopPendingError
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.tui_gateway.test_hosted_room_served_elsewhere_stop import _OtherDaemon
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server
from tui_gateway.hosted_room_service import HostedRoomService


def _stopping_behind_a_silent_daemon(tmp_path: Path):
    """A turn of a member another profile's daemon serves, stopping: that daemon does not answer."""
    now = [100.0]
    db = storage_handle(tmp_path)
    service = HostedRoomService(_server(), store=db, placement="profile")
    service.rpc = service.runtime.rpc = _FakeRPC()
    service.local_profiles = lambda: ("default", "ops")
    daemon = _OtherDaemon()
    service._resolve_member_transport = service.runtime.transport_resolver = lambda *_: daemon
    service.runtime.clock = lambda: now[0]
    service.runtime.lease_ttl_seconds, service.runtime.indeterminate_defer_seconds = 1000, 5
    service.create_room(room_id="room-1", name="Two daemons", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"}])
    service.send(room_id="room-1", event_id="user-1", payload={"text": "@ops inspect", "thread_id": "thread-1"})
    task, = driver.list_tasks(db, placement="profile", room_id="room-1", status="queued")
    lease = service.runtime._ensure_lease(service.bindings()[0])
    driver.start_task(db, task["identity"], lease, placement="profile", expected_cancel_generation=0,
                      clock=service.runtime.clock)
    daemon.add_session(profile="ops", title="Group: room-1", active=True, task_id=task["identity"].task_id)
    daemon.down = True
    assert service.stop_room("room-1", cancel_id="stop-1") == 1
    assert driver.get_task(db, task["identity"], placement="profile")["status"] == "stopping"
    return service, db, task["identity"], daemon


def _worker_defers(service, identity) -> None:
    """What the worker commits once the Stop stays unconfirmed past the deferral window."""
    task = driver.get_task(service.store, identity, placement=service.placement)
    if task["status"] == "stopping":
        lease = service.runtime._ensure_lease(service.bindings()[0])
        service.runtime._fenced(driver.defer_unconfirmed_stop, None, task, lease, reason="member_unavailable")


def test_a_turn_deferred_after_the_snapshot_keeps_disband_waiting(tmp_path: Path, monkeypatch):
    service, db, identity, daemon = _stopping_behind_a_silent_daemon(tmp_path)
    cancel = service.runtime.cancel

    def deferred_first(task_identity, **kwargs):
        _worker_defers(service, task_identity)  # between Stop's snapshot and its cancel
        return cancel(task_identity, **kwargs)

    monkeypatch.setattr(service.runtime, "cancel", deferred_first)
    with pytest.raises(RoomStopPendingError):
        service.stop_room("room-1", cancel_id="room-disbanded", require_acknowledged=True)
    assert driver.get_task(db, identity, placement="profile")["status"] == "deferred"
    daemon.down = False  # the daemon answers: its confirmed Stop settles the turn
    monkeypatch.setattr(service.runtime, "cancel", cancel)
    service.stop_room("room-1", cancel_id="room-disbanded", require_acknowledged=True)
    assert driver.get_task(db, identity, placement="profile")["status"] == "cancelled"


def test_a_turn_deferred_while_its_stop_is_being_asked_keeps_disband_waiting(tmp_path: Path, monkeypatch):
    service, db, identity, _daemon = _stopping_behind_a_silent_daemon(tmp_path)
    ask = service.runtime._peer_stop_acknowledged

    def deferred_meanwhile(binding, task):
        _worker_defers(service, task["identity"])  # during the confirmation calls
        return ask(binding, task)

    monkeypatch.setattr(service.runtime, "_peer_stop_acknowledged", deferred_meanwhile)
    with pytest.raises(RoomStopPendingError):
        service.stop_room("room-1", cancel_id="room-disbanded", require_acknowledged=True)
    assert driver.get_task(db, identity, placement="profile")["status"] == "deferred"
