"""The runtime holds a queued task back while its room's copies must still store it (majority mode)."""

import threading

from gateway import hosted_room_driver as state
from tests.tui_gateway.test_hosted_room_driver_runtime import (  # noqa: F401
    ROOM_ID, FakeSessionRPC, _admit, _identity, _runtime, _wait_for, db)


def test_a_queued_task_waits_until_dispatch_is_ready(db):
    identity = _identity()
    _admit(db, identity)
    ready, asked = threading.Event(), []

    def dispatch_ready(binding, task):
        asked.append((binding.room_id, task["identity"].task_id))
        return ready.is_set()

    runtime = _runtime(db, FakeSessionRPC(), dispatch_ready=dispatch_ready)
    runtime.start()
    _wait_for(lambda: len(asked) >= 2)
    assert state.get_task(db, identity)["status"] == "queued"  # held back, and asked again each pass
    ready.set()
    runtime.wakeup()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=5.0)
    assert asked[0] == (ROOM_ID, identity.task_id)
