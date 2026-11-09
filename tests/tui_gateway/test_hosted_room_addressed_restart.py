"""A completed silent member turn survives restart without duplicate execution."""

import time

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server


def _service(db, monkeypatch):
    service = HostedRoomService(_server(), db_path=db)
    rpc = _FakeRPC()
    monkeypatch.setattr(service, "rpc", rpc)
    monkeypatch.setattr(service.runtime, "rpc", rpc)
    monkeypatch.setattr(service, "local_profiles", lambda: ("default", "ops"))
    return service


@pytest.mark.parametrize("legacy_prompt", [False, True])
def test_restart_publishes_addressed_pass_once_without_reexecuting(
    tmp_path, monkeypatch, legacy_prompt
):
    db = tmp_path / "state.db"
    service = _service(db, monkeypatch)
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    send = {
        "room_id": "room-1",
        "event_id": "user-1",
        "payload": {"text": "@ops continue", "thread_id": "thread-1"},
    }
    service.send(**send)
    binding = service.bindings()[0]
    with monkeypatch.context() as patch:
        if legacy_prompt:
            # A queued pre-upgrade task keeps its original prompt and identity.
            patch.setattr(
                discussion,
                "_build_prompt",
                lambda **kwargs: 'Old room prompt: reply "(pass)" if idle.',
            )
        service.prepare_room(binding)
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="before-restart",
        ttl_seconds=30,
        clock=time.time,
    )
    attempt = driver.start_task(
        db, task["identity"], lease, expected_cancel_generation=0, clock=time.time
    )
    driver.settle_task(
        db,
        attempt,
        settlement_id="reply-1",
        status="settled",
        result={"text": "(pass)"},
        clock=time.time,
    )

    # Crash after the execution receipt, before its room publication.
    restored = _service(db, monkeypatch)
    restored.prepare_room(restored.bindings()[0])
    events = restored._events("room-1")
    failures = [e for e in events if e["kind"] == "turn.failed"]
    assert len(failures) == 1
    assert "directly addressed" in failures[0]["payload"]["error"]
    assert not any(e["kind"] == "message.member" for e in events)

    # Another restart and a duplicate delivery must reuse that same outcome.
    replayed = _service(db, monkeypatch)
    replayed.send(**send)
    replayed.prepare_room(replayed.bindings()[0])
    assert replayed._events("room-1") == events
    tasks = driver.list_tasks(db, room_id="room-1")
    assert len(tasks) == 1
    assert tasks[0]["identity"] == task["identity"]
    assert tasks[0]["execution_generation"] == 1
    assert tasks[0]["status"] == "settled"
