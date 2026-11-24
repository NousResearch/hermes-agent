"""``groups.log`` serves member replies whose actor carries the profile/display/connection stamp."""

from __future__ import annotations

import pytest

import tui_gateway.server as srv
from gateway import hosted_rooms
from tui_gateway import methods_groups


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    (path / "profiles" / "ops").mkdir(parents=True)
    (path / "profiles" / "ops" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def test_log_with_a_stamped_member_actor_satisfies_its_contract(home):
    room = srv.handle_request({"id": 1, "method": "groups.create", "params": {
        "room_id": "room-1", "name": "Room",
        "members": [{"member_id": "default", "profile": "default", "handle": "hermes"},
                    {"member_id": "ops", "profile": "ops", "handle": "ops", "display_name": "Ops"}]}})
    actor = {"kind": "member", "id": "ops", "profile": "ops", "display_name": "Ops",
             "connection_id": "peer-1"}
    hosted_rooms.append_event(
        hosted_rooms.default_db_path(), room_id="room-1", event_id="reply-1", kind="message.member",
        actor=actor, payload={"text": "Done", "thread_id": "thread-1"},
        authority_gateway_id=room["result"]["room"]["authority_gateway_id"], authority_epoch=1)

    response = srv.handle_request(
        {"id": 2, "method": "groups.log", "params": {"room_id": "room-1", "since_seq": 0}})

    assert response["result"]["events"][-1]["actor"] == actor
