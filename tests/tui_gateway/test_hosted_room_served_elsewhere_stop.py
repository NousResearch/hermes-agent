"""A local member another profile's daemon serves: a turn deferred behind its unconfirmed Stop
settles only once that daemon confirms the turn is not running, as for a member on another
gateway. A daemon that stays unreachable keeps the turn, and Disband refuses."""

from pathlib import Path

import pytest

from hermes_agent.gateway import hosted_room_driver as driver, hosted_rooms
from hermes_agent.tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.test_hosted_room_driver_runtime import FakeSessionRPC
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server


class _OtherDaemon(FakeSessionRPC):
    """The member's transport: another profile's daemon, whose control socket can go away."""
    served_elsewhere = True

    def __init__(self) -> None:
        super().__init__(auto_complete=False)
        self.down = False

    def __getattribute__(self, name):
        if name in {"resolve_exact", "resume", "history", "info", "interrupt"} and object.__getattribute__(self, "down"):
            def gone(**_):
                raise OSError("owner control socket is gone")
            return gone
        return super().__getattribute__(name)


def _deferred_behind_unconfirmed_stop(tmp_path: Path):
    now = [100.0]
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
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
    task, = driver.list_tasks(db, room_id="room-1", status="queued")
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)
    driver.start_task(db, task["identity"], lease, expected_cancel_generation=0, clock=service.runtime.clock)
    daemon.add_session(profile="ops", title="Group: room-1", active=True, task_id=task["identity"].task_id)
    daemon.down = True  # the daemon's control socket goes away while its agent keeps running
    assert service.stop_room("room-1", cancel_id="stop-1") == 1
    assert driver.get_task(db, task["identity"])["status"] == "stopping"
    now[0] = 110.0
    assert service.runtime._retry_stopping_tasks(binding, lease) is False
    assert driver.get_task(db, task["identity"])["status"] == "deferred"
    return service, db, task["identity"], daemon


def test_disband_refuses_while_the_members_daemon_cannot_confirm_the_stop(tmp_path: Path):
    service, db, identity, daemon = _deferred_behind_unconfirmed_stop(tmp_path)
    with pytest.raises(hosted_rooms.RoomStopPendingError):
        service.stop_room("room-1", cancel_id="room-disbanded", require_acknowledged=True)
    assert service.stop_room("room-1", cancel_id="stop-2") == 0
    assert driver.get_task(db, identity)["status"] == "deferred"
    assert not [c for c in daemon.calls if c[0] == "interrupt"]


def test_the_daemons_confirmed_stop_settles_the_deferred_turn(tmp_path: Path):
    service, db, identity, daemon = _deferred_behind_unconfirmed_stop(tmp_path)
    daemon.down = False  # back, with the turn still running there
    service.stop_room("room-1", cancel_id="room-disbanded", require_acknowledged=True)
    assert driver.get_task(db, identity)["status"] == "cancelled"
    assert [c[1]["expected_task_id"] for c in daemon.calls if c[0] == "interrupt"] == [identity.task_id]
