"""``groups.disband`` on a quarantined room: refused until confirmed, then a local tombstone only."""

from __future__ import annotations

import sqlite3

import pytest

import tui_gateway.server as srv
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from tui_gateway import methods_groups

LOCAL_MEMBERS = [
    {"member_id": "default", "profile": "default", "handle": "hermes"},
    {"member_id": "ops", "profile": "ops", "handle": "ops"},
]


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    path.mkdir()
    (path / "profiles" / "ops").mkdir(parents=True)
    (path / "profiles" / "ops" / "config.yaml").write_text("{}\n")  # identity marker: local roster
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def _error(envelope):
    assert "error" in envelope, envelope
    return envelope["error"]


def _local_room(room_id):
    _result(srv._methods["groups.create"](1, {"room_id": room_id, "name": "Local room", "members": LOCAL_MEMBERS}))


def _quarantine(room_id):
    with sqlite3.connect(rooms.default_db_path()) as conn:
        conn.execute("INSERT INTO hosted_room_quarantine VALUES (?, 'unsafe_replica_promotion', 1)", (room_id,))


def _stale_queued_task(room_id):
    # Work admitted before the quarantine landed; the driver must never pick it up again.
    assert driver.list_tasks(rooms.default_db_path(), room_id=room_id) == []  # creates the driver schema
    with sqlite3.connect(rooms.default_db_path()) as conn:
        conn.execute("""INSERT INTO hosted_room_driver_tasks
            (room_id, task_id, thread_id, turn_id, source_event_seq, payload_json, payload_digest, status,
             created_at, updated_at)
            VALUES (?, 'task', 'thread', 'turn', 1, '{}', 'digest', 'queued', 1, 1)""", (room_id,))


def test_quarantined_room_disband_needs_confirmation_and_stops_or_sends_nothing(home, monkeypatch):
    _local_room("room-1")
    _quarantine("room-1")
    _stale_queued_task("room-1")
    history = _result(srv._methods["groups.log"](2, {"room_id": "room-1"}))["events"]
    service = methods_groups.get_hosted_room_service()

    def forbidden(*args, **kwargs):
        raise AssertionError("a quarantined Disband must not stop work or contact other gateways")

    monkeypatch.setattr(service, "stop_room", forbidden)
    monkeypatch.setattr(service, "revoke_room_routes", forbidden)

    refused = _error(srv._methods["groups.disband"](3, {"room_id": "room-1"}))
    assert refused["code"] == 4113
    assert refused["data"] == {"reason": "room_authority_quarantined"}
    assert "confirm_quarantined set to true" in refused["message"]

    tombstone = _result(srv._methods["groups.disband"](
        4, {"room_id": "room-1", "confirm_quarantined": True}))["tombstone"]
    assert tombstone["idempotent"] is False and "event" not in tombstone
    again = _result(srv._methods["groups.disband"](5, {"room_id": "room-1", "confirm_quarantined": True}))
    assert again["tombstone"] == {**tombstone, "idempotent": True}

    assert _result(srv._methods["groups.list"](6, {}))["rooms"] == []
    log = _result(srv._methods["groups.log"](7, {"room_id": "room-1", "include_disbanded": True}))
    assert log["events"] == history
    # The room left the driver's bindings, so nothing will ever run for it.
    assert all(binding.room_id != "room-1" for binding in service.bindings())
    with sqlite3.connect(rooms.default_db_path()) as conn:
        assert conn.execute("SELECT status FROM hosted_room_driver_tasks WHERE room_id='room-1'").fetchall() == [
            ("queued",)]


def test_confirmation_on_a_writable_room_still_takes_the_normal_disband(home, monkeypatch):
    _local_room("room-1")
    service = methods_groups.get_hosted_room_service()
    calls = []
    monkeypatch.setattr(service, "stop_room", lambda room_id, **kwargs: calls.append(("stop", room_id)) or 0)
    monkeypatch.setattr(service, "revoke_room_routes", lambda room_id: calls.append(("revoke", room_id)) or 0)

    tombstone = _result(srv._methods["groups.disband"](
        2, {"room_id": "room-1", "confirm_quarantined": True}))["tombstone"]
    # The usual Stop, route revocation and terminal room.disbanded event, as without the flag.
    assert calls == [("stop", "room-1"), ("revoke", "room-1")]
    assert tombstone["event"]["kind"] == "room.disbanded"
