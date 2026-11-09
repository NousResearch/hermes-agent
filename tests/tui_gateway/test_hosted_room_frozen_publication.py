"""Frozen admissions reconstruct from canonical storage, not display caches."""

import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from tui_gateway.hosted_room_service import HostedRoomService


def service(db):
    instance = HostedRoomService(SimpleNamespace(_methods={}, _sessions={},
                                 _sessions_lock=threading.Lock()), db_path=db)
    instance.local_profiles = lambda: ("writer", "reviewer")
    return instance


@pytest.mark.parametrize("partial", [False, True])
def test_terminal_reopen_uses_frozen_input_after_cache_loss(tmp_path, monkeypatch, partial):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "home")
    db = tmp_path / "room.db"
    current = service(db)
    current.create_room(room_id="room", name="Room", members=[
        {"member_id": name, "profile": name, "handle": name}
        for name in ("writer", "reviewer")])
    current.send(room_id="room", event_id="input", payload={"text": "@writer original", "thread_id": "t"})
    task, = driver.list_tasks(db, room_id="room", status="queued")
    lease = driver.acquire_lease(db, room_id="room", gateway_id="home", authority_epoch=1,
        process_generation="test", ttl_seconds=300, clock=time.time)
    attempt = driver.start_task(db, task["identity"], lease, expected_cancel_generation=0, clock=time.time)
    settled = driver.settle_task(db, attempt, settlement_id="result", status="settled",
        result={"text": "original reply"}, clock=time.time)
    if partial:
        append = rooms.append_event
        def interrupted(*args, **kwargs):
            if kwargs.get("kind") == "turn.settled":
                raise OSError("lost terminal append")
            return append(*args, **kwargs)
        with monkeypatch.context() as crash:
            crash.setattr(rooms, "append_event", interrupted)
            with pytest.raises(OSError, match="lost terminal"):
                current._publish_terminal_tasks(current._room("room"))
    # Advance and evict the display projection without admitting any further work.
    for index in range(30):
        rooms.append_event(db, room_id="room", event_id=f"later-{index}", kind="message.user",
            actor={"kind": "user", "id": "home"}, authority_gateway_id="home", authority_epoch=1,
            payload={"text": "@reviewer later", "thread_id": "t" if partial else "other"})
    current.policy_checkpoint.snapshot(room_id="room", latest_seq=current._room("room")["latest_seq"])
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM hosted_room_policy_events WHERE room_id='room' AND seq=1")
        conn.execute("DELETE FROM hosted_room_policy_transcript WHERE room_id='room' AND seq=1")
    cold = service(db)
    monkeypatch.setattr(driver, "admit_task", lambda *a, **kw: pytest.fail("publication readmitted work"))
    assert cold._publish_terminal_tasks(cold._room("room"))
    cold.policy_checkpoint.sync(room_id="room", latest_seq=cold._room("room")["latest_seq"])
    own = [e for e in cold._events("room") if e["payload"].get("task_id") == task["identity"].task_id]
    assert [e["kind"] for e in own] == ["message.member", "turn.settled"]
    assert own[0]["payload"]["text"] == "original reply"
    assert driver.get_task(db, task["identity"]) == settled
    assert not cold._publish_terminal_tasks(cold._room("room"))
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_driver_tasks").fetchone()[0] == 1
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='room' AND seq=1")
    with pytest.raises(RuntimeError, match="input event is missing"):
        cold.policy_checkpoint.events_for_task(room_id="room", source_event_seq=1,
            input_context=task["payload"]["input_context"], task_id=task["identity"].task_id)
