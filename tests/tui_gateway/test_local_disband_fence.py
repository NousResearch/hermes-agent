"""The local Disband admission boundary survives incomplete Stop and a cold service."""
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from tui_gateway import server
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def room(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "profiles" / "ops").mkdir(parents=True)
    service = HostedRoomService(server, db_path=tmp_path / "state.db")
    service.create_room(room_id="room", name="Local room", members=[
        {"member_id": name, "profile": name, "handle": name} for name in ("default", "ops")])
    monkeypatch.setattr(server, "get_hosted_room_service", lambda: service)
    return service


def _append(service, event_id, payload=None):
    room = hosted_rooms.room_state(service.db_path, room_id="room")
    return hosted_rooms.append_event(
        service.db_path, room_id="room", event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "desktop"}, payload=payload or {"text": "Input", "thread_id": event_id},
        authority_gateway_id=room["authority_gateway_id"], authority_epoch=room["authority_epoch"])


def _pause_disband(service, monkeypatch):
    def stopping(*args, **kwargs):
        raise RuntimeError("room work is still stopping")
    with monkeypatch.context() as patch:
        patch.setattr(service, "stop_room", stopping)
        result = server._methods["groups.disband"](1, {"room_id": "room"})
    assert result["error"]["code"] == 5114
    assert "disbanded_at" not in hosted_rooms.room_state(service.db_path, room_id="room")


@pytest.mark.parametrize("admission", [
    "send", "append", "prepare", "admit", "start", "submit", "retry",
    "requeue_deferred", "requeue_indeterminate", "requeue_not_admitted", "allow", "files_send",
])
def test_incomplete_disband_fences_every_new_local_admission(room, monkeypatch, admission):
    service, db = room, room.db_path
    if admission == "prepare":
        # Pending canonical input with no existing live task: an unfenced planner
        # must actually enqueue work, rather than returning early on a prior task.
        _append(service, "unplanned")
        binding = service.bindings()[0]
        assert driver.list_tasks(db, room_id="room") == []
        _pause_disband(service, monkeypatch)
        cold = HostedRoomService(server, db_path=db)
        cold.prepare_room(binding)
        assert driver.list_tasks(db, room_id="room") == []
        return
    event = service.send(room_id="room", event_id="first", payload={"text": "Discuss"})
    task = driver.list_tasks(db, room_id="room")[0]
    identity = task["identity"]
    binding = service.bindings()[0]
    lease = driver.acquire_lease(
        db, room_id="room", gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="test", ttl_seconds=600, clock=lambda: 10)
    attempt = None
    now = 13
    if admission in {"submit", "retry", "requeue_deferred", "requeue_indeterminate", "requeue_not_admitted"}:
        attempt = driver.start_task(db, identity, lease, expected_cancel_generation=0, clock=lambda: 11)
        if admission in {"retry", "requeue_deferred"}:
            driver.defer_not_admitted_task(db, attempt, reason="member_unavailable", clock=lambda: 12)
        elif admission == "requeue_indeterminate":
            lease = driver.acquire_lease(
                db, room_id="room", gateway_id=binding.gateway_id,
                authority_epoch=binding.authority_epoch, process_generation="successor",
                ttl_seconds=600, clock=lambda: 611)
            driver.recover_room(db, lease, clock=lambda: 612)
            now = 613
    upload = None
    if admission == "files_send":
        upload = service.put_attachment(
            room_id="room", upload_id="upload", kind="file", name="sample.txt", mime="text/plain", data=b"sample")
    _pause_disband(service, monkeypatch)
    # New object and new SQLite connections, not a process-local closing flag.
    cold = HostedRoomService(server, db_path=db)
    assert cold.peer_routes == {}
    assert cold.status("room")["counts"] == service.status("room")["counts"]
    assert _append(cold, "first", event["payload"])["idempotent"] is True
    assert driver.admit_task(db, identity, payload=task["payload"], clock=lambda: now)["idempotent"] is True
    before = driver.list_tasks(db, room_id="room")
    if admission == "allow":
        approvals = []
        cold.rpc = SimpleNamespace(approve=lambda **kwargs: approvals.append(kwargs) or {"resolved": 1})
        cold._set_pending_action("room", "default", {
            "request_id": "approval", "session_id": "native", "task_id": identity.task_id, "execution_generation": 1})
        def operation():
            return cold.approve_room_task(
                "room", member_id="default", task_id=identity.task_id, execution_generation=1,
                request_id="approval", choice="once")
    else:
        operations = {
            "send": lambda: cold.send(room_id="room", event_id="late", payload={"text": "Late"}),
            "append": lambda: _append(cold, "late"),
            "admit": lambda: driver.admit_task(
                db, driver.TaskIdentity("room", "new-task", "new-thread", "new-turn"),
                payload=task["payload"], clock=lambda: now),
            "start": lambda: driver.start_task(db, identity, lease, expected_cancel_generation=0, clock=lambda: now),
            "submit": lambda: driver.fence_task_admission(db, attempt, clock=lambda: now),
            "retry": lambda: cold.retry_room_task("room", task_id=identity.task_id),
            "requeue_deferred": lambda: driver.requeue_deferred_task(
                db, identity, lease, expected_execution_generation=1, expected_cancel_generation=0, clock=lambda: now),
            "requeue_indeterminate": lambda: driver.requeue_indeterminate_task(
                db, identity, lease, expected_execution_generation=1, expected_cancel_generation=0, clock=lambda: now),
            "requeue_not_admitted": lambda: driver.requeue_not_admitted_task(db, attempt, clock=lambda: now),
            "files_send": lambda: cold.send(room_id="room", event_id="late", payload={
                "text": "File", "attachments": [{key: upload[key] for key in
                    ("attachment_id", "kind", "name", "mime", "size")}]}),
        }
        operation = operations[admission]
    with pytest.raises((hosted_rooms.HostedRoomError, driver.RoomUnavailableError), match="being disbanded"):
        operation()
    assert driver.list_tasks(db, room_id="room") == before
    assert not any(item["event_id"] == "late" for item in hosted_rooms.read_events(db, room_id="room")["events"])
    if upload:
        assert cold.attachments.find_upload(room_id="room", upload_id="upload")["state"] == "uploaded"
    if admission == "allow":
        assert approvals == []
        cold.approve_room_task(
            "room", member_id="default", task_id=identity.task_id, execution_generation=1,
            request_id="approval", choice="deny")
        assert [item["choice"] for item in approvals] == ["deny"]


@pytest.mark.parametrize("failure", ["stop", "revoke"])
def test_disband_keeps_accepted_cleanup_and_replay_available(room, monkeypatch, failure):
    service = room
    service.send(room_id="room", event_id="first", payload={"text": "Discuss"})
    binding = service.bindings()[0]
    task = driver.list_tasks(service.db_path, room_id="room")[0]
    lease = driver.acquire_lease(
        service.db_path, room_id="room", gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="test", ttl_seconds=600, clock=lambda: 10)
    attempt = driver.start_task(service.db_path, task["identity"], lease, expected_cancel_generation=0, clock=lambda: 11)
    driver.fence_task_admission(service.db_path, attempt, clock=lambda: 12)
    def unavailable(*args, **kwargs):
        raise RuntimeError("cleanup pending")
    with monkeypatch.context() as patch:
        if failure == "stop":
            patch.setattr(service, "stop_room", unavailable)
        else:
            patch.setattr(service, "stop_room", lambda *args, **kwargs: 0)
            patch.setattr(service, "revoke_room_routes", unavailable)
        assert server._methods["groups.disband"](1, {"room_id": "room"})["error"]["code"] == 5114
    with pytest.raises(hosted_rooms.HostedRoomError, match="being disbanded"):
        _append(service, "late")
    settled = driver.settle_task(
        service.db_path, attempt, settlement_id="terminal", status="settled", result={"text": "Finished"}, clock=lambda: 13)
    service.publish_terminal(binding, settled)
    assert any(event["kind"] == "message.member" for event in
               hosted_rooms.read_events(service.db_path, room_id="room")["events"])
    assert not driver.list_tasks(service.db_path, room_id="room", status="queued")
    result = server._methods["groups.disband"](2, {"room_id": "room"})
    assert "result" in result, result
    replay = server._methods["groups.disband"](3, {"room_id": "room"})["result"]["tombstone"]
    assert replay["idempotent"] is True
    assert replay["event"]["event_id"] == result["result"]["tombstone"]["event"]["event_id"]
