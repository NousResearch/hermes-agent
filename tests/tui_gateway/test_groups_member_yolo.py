"""A targeted room control uses one participant's existing session YOLO lifecycle (#91334)."""

from pathlib import Path

import pytest

from gateway import hosted_rooms
from hermes_state import SessionDB
from tools import approval
from tui_gateway import server
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def rooms(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    worker = home / "profiles" / "worker"
    worker.mkdir(parents=True)
    for path in (home, worker):
        (path / "config.yaml").write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_path", None)
    monkeypatch.setattr(server, "_cfg_sig", None)

    def unexpected_build(*args, **kwargs):
        pytest.fail("reading or setting participant YOLO must not build an agent")

    monkeypatch.setattr(server, "_schedule_agent_build", unexpected_build)
    monkeypatch.setattr(server, "_make_agent", unexpected_build)
    launch_db = SessionDB(home / "state.db")
    worker_db = SessionDB(worker / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    service = HostedRoomService(server)
    monkeypatch.setattr(server, "get_hosted_room_service", lambda: service)
    members = [
        {"member_id": "lead", "profile": "default", "handle": "lead"},
        {"member_id": "builder", "profile": "worker", "handle": "builder"},
    ]
    for room_id in ("room-a", "room-b"):
        service.create_room(room_id=room_id, name=room_id, members=members)
        for db, prefix in ((launch_db, "lead"), (worker_db, "builder")):
            key = f"{prefix}-{room_id}"
            db.create_session(key, source="bot_room")
            db.set_session_title(key, f"Group: {room_id}")
    worker_db.create_session("builder-chat", source="desktop")
    worker_db.set_session_title("builder-chat", "Bot Chat")
    yield service, launch_db, worker_db
    for db in (launch_db, worker_db):
        for row in db.list_sessions_rich():
            approval.clear_session(row["id"])
        db.close()


def _request(room_id="room-a", member_id="builder", **extra):
    return server.handle_request({
        "id": "member-control", "method": "groups.member.yolo",
        "params": {"room_id": room_id, "member_id": member_id, **extra},
    })


def _result(**extra):
    response = _request(**extra)
    assert "error" not in response, response
    return response["result"]


def test_yolo_is_isolated_idempotent_and_survives_restart_and_compression(rooms, monkeypatch):
    service, launch_db, worker_db = rooms
    before = {path: path.read_bytes() for path in service.root.rglob("config.yaml")}
    assert _result()["enabled"] is False
    assert server._sessions == {}
    assert _result(enabled=True) == {
        "room_id": "room-a", "member_id": "builder", "session_key": "builder-room-a",
        "enabled": True, "effective_enabled": True,
    }
    assert _result(enabled=True)["enabled"] is True  # retry is a set, not another toggle
    assert approval.is_session_yolo_enabled("builder-room-a")
    assert SessionDB.session_yolo_enabled(worker_db.get_session("builder-room-a"))
    assert _result(member_id="lead")["enabled"] is False
    assert _result(room_id="room-b")["enabled"] is False
    assert not SessionDB.session_yolo_enabled(worker_db.get_session("builder-chat"))
    assert not SessionDB.session_yolo_enabled(launch_db.get_session("lead-room-a"))
    assert all(path.read_bytes() == content for path, content in before.items())

    server._sessions.clear()
    approval.clear_session("builder-room-a")
    assert _result()["enabled"] is True
    assert not approval.is_session_yolo_enabled("builder-room-a")
    with monkeypatch.context() as patch:
        patch.setattr(server, "_schedule_agent_build", lambda sid: None)
        resumed = server.handle_request({
            "id": "worker-resume", "method": "session.resume",
            "params": {"profile": "worker", "session_id": "builder-room-a", "omit_messages": True},
        })
    assert "error" not in resumed, resumed
    assert approval.is_session_yolo_enabled("builder-room-a")
    runtime = resumed["result"]["session_id"]
    record = server._sessions[runtime]
    transport = record["transport"]
    pending = {"request_id": "pending-approval", "command": "rm -rf scratch"}
    from tools.approval_gateway_wait import _ApprovalEntry

    entry = _ApprovalEntry(pending)
    with approval._lock:
        approval._gateway_queues["builder-room-a"] = [entry]
    assert _result(enabled=False)["enabled"] is False
    assert _result(enabled=True)["enabled"] is True
    assert server._sessions[runtime] is record
    assert record["transport"] is transport
    assert approval.get_pending_gateway_approval("builder-room-a") == pending
    assert not entry.event.is_set()
    assert entry.result is None

    # A real stored compression lineage is resolved through the stock session lookup.
    from tools.approval_yolo import transfer_session_yolo, with_session_yolo

    worker_db.end_session("builder-room-a", "compression")
    worker_db.create_session(
        "builder-tip", source="bot_room", parent_session_id="builder-room-a",
        model_config=with_session_yolo({}, "builder-room-a"))
    transfer_session_yolo("builder-room-a", "builder-tip")
    assert _result()["session_key"] == "builder-tip"
    assert _result(enabled=False)["enabled"] is False
    assert not SessionDB.session_yolo_enabled(worker_db.get_session("builder-tip"))
    server._sessions.clear()
    approval.clear_session("builder-tip")
    assert _result()["enabled"] is False


def test_disabled_override_reports_profile_bypass_without_changing_it(rooms):
    service, _, _ = rooms
    config = service.root / "profiles" / "worker" / "config.yaml"
    config.write_text("approvals:\n  mode: off\n", encoding="utf-8")
    result = _result(enabled=False)
    assert result["enabled"] is False
    assert result["effective_enabled"] is True
    assert config.read_text(encoding="utf-8") == "approvals:\n  mode: off\n"


@pytest.mark.parametrize("enabled", ["true", 1, {}, []])
def test_non_boolean_update_is_rejected_before_effects(rooms, enabled):
    _, _, worker_db = rooms
    result = _request(enabled=enabled)
    assert result["error"]["code"] == 4002
    assert not SessionDB.session_yolo_enabled(worker_db.get_session("builder-room-a"))


def _remove_session(rooms):
    rooms[2].delete_session("builder-room-a")


def _wrong_source(rooms):
    db = rooms[2]
    db.delete_session("builder-room-a")
    db.create_session("unrelated", source="desktop")
    db.set_session_title("unrelated", "Group: room-a")


def _peer_room(rooms):
    hosted_rooms.create_room(
        rooms[0].db_path, room_id="remote", name="Remote",
        members=[{"member_id": "builder", "profile": "worker", "handle": "builder",
                  "target": {"kind": "peer"}}],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id())


def _foreign_room(rooms):
    hosted_rooms.create_room(
        rooms[0].db_path, room_id="foreign", name="Foreign",
        members=[], authority_gateway_id="another-gateway")


def _disband_room(rooms):
    hosted_rooms.disband_room(
        rooms[0].db_path, room_id="room-a",
        expected_gateway_id=hosted_rooms.local_authority_gateway_id(), expected_epoch=1)


@pytest.mark.parametrize(("room_id", "member_id", "prepare"), [
    pytest.param("missing", "builder", None, id="unknown-room"),
    pytest.param("room-a", "missing", None, id="unknown-member"),
    pytest.param("remote", "builder", _peer_room, id="peer"),
    pytest.param("room-a", "builder", _remove_session, id="missing-session"),
    pytest.param("room-a", "builder", _wrong_source, id="wrong-source"),
    pytest.param("foreign", "builder", _foreign_room, id="foreign-authority"),
    pytest.param("room-a", "builder", _disband_room, id="disbanded"),
])
def test_invalid_target_never_changes_another_session(rooms, room_id, member_id, prepare):
    if prepare is not None:
        prepare(rooms)
    response = _request(room_id, member_id, enabled=True)
    assert response["error"]["code"] == 4124, response
    for db in rooms[1:]:
        assert not any(SessionDB.session_yolo_enabled(row) for row in db.list_sessions_rich())
    assert server._sessions == {}


def test_another_worker_owner_is_refused_without_changing_the_session(rooms):
    from hermes_cli.active_sessions import try_acquire_active_session

    service, _, worker_db = rooms
    lease, refusal = try_acquire_active_session(
        session_id="builder-room-a", surface="cli", config={},
        registry_home=service.root / "profiles" / "worker")
    assert refusal is None
    try:
        response = _request(enabled=True)
        assert "owned by another worker" in response["error"]["message"]
        assert not SessionDB.session_yolo_enabled(worker_db.get_session("builder-room-a"))
        assert not approval.is_session_yolo_enabled("builder-room-a")
    finally:
        lease.release()


def test_failed_persistence_does_not_acknowledge_or_arm_a_cold_bypass(rooms, monkeypatch):
    def failed_write(*args):
        raise OSError("session store is read-only")

    monkeypatch.setattr(SessionDB, "set_session_yolo", failed_write)
    response = _request(enabled=True)
    assert "read-only" in response["error"]["message"]
    assert not approval.is_session_yolo_enabled("builder-room-a")
    assert _result()["enabled"] is False
