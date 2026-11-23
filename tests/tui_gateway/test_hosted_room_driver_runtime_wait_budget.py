"""Sleep-budget invariant for ``_wait_for_terminal`` (hosted-room lease renewal).

Topical sibling of ``test_hosted_room_driver_runtime.py``: the FILE_LINES fix
prescribes new tests go in ``test_<stem>_<topic>.py`` so the facade stays
inside its file-line cap (see root AGENTS.md).
"""

from __future__ import annotations

import threading
from pathlib import Path

from tests.tui_gateway import test_hosted_room_driver_runtime as runtime_fixtures
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    FakeSessionRPC,
    _admit,
    _identity,
    _runtime,
    _wait_for,
)

db = runtime_fixtures.db
store = runtime_fixtures.store_fixture


def test_wait_for_terminal_sleep_budget_stays_inside_the_renewal_window(db: Path, store):
    identity = _identity()
    _admit(store, identity)
    rpc = FakeSessionRPC(auto_complete=False)
    runtime = _runtime(store, rpc, turn_timeout_seconds=30)
    waits: list[float] = []

    class RecordingWake(threading.Event):
        """Record each requested wait budget, then block like the real wake event."""

        def wait(self, timeout=None):
            waits.append(timeout if timeout is not None else float("inf"))
            return super().wait(timeout)

    runtime._wake = RecordingWake()
    runtime.start()
    try:
        assert rpc.submitted.wait(timeout=5.0)
        _wait_for(lambda: len([w for w in waits if w >= 0.05]) >= 3, timeout=5.0)
        cap = min(runtime.active_poll_interval_seconds, runtime.lease_ttl_seconds / 4)
        assert all(timeout <= cap + 1e-9 for timeout in waits)
    finally:
        if rpc.submitted.is_set():
            rpc.complete(identity.task_id, content="done")
        assert runtime.stop(timeout=5.0)
