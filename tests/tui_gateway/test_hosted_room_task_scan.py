"""A room task scan reads the driver store once and keeps the per-status order."""

from types import SimpleNamespace

from gateway import hosted_rooms
from tui_gateway import hosted_room_service
from tui_gateway.hosted_room_service import HostedRoomService


def service(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db, room_id="one", name="one",
        members=[{"member_id": "writer", "profile": "writer", "handle": "writer"}],
        authority_gateway_id="gateway",
    )
    return HostedRoomService(SimpleNamespace(), db_path=db)


def test_terminal_scan_opens_the_driver_store_once(tmp_path, monkeypatch):
    instance = service(tmp_path)
    original, calls = hosted_room_service.driver.list_tasks, []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("status"))
        return original(*args, **kwargs)

    monkeypatch.setattr(hosted_room_service.driver, "list_tasks", counted)
    assert list(instance._list_tasks("one", hosted_room_service._TERMINAL_STATUSES)) == []
    assert calls == [None]


def test_scan_groups_by_requested_status_in_admission_order(tmp_path, monkeypatch):
    instance = service(tmp_path)
    admitted = [{"id": "a", "status": "settled"}, {"id": "b", "status": "failed"},
                {"id": "c", "status": "running"}, {"id": "d", "status": "settled"}]
    monkeypatch.setattr(hosted_room_service.driver, "list_tasks", lambda *args, **kwargs: admitted)
    scanned = [task["id"] for task in instance._list_tasks("one", ("failed", "settled"))]
    assert scanned == ["b", "a", "d"]
