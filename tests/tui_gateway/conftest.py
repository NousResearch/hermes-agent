"""tui_gateway test fixtures.

Several files here import ``tui_gateway.server`` inside a ``patch.dict("sys.modules", {"hermes_constants":
MagicMock(...)})`` window so the module binds a fixed home. The server's import graph reaches
``agent.process_bootstrap`` → ``hermes_bootstrap``, which is process boot: PM dependency activation reads
the real install root through ``hermes_constants`` and exits the process when that is a MagicMock.
Importing it once here, before any window opens, keeps boot out of the mocked import.
"""

import hermes_bootstrap

import threading
import time

import pytest

from tests.waiting import DEFAULT_TIMEOUT


_TURN_WORKER_PREFIXES = ("prompt-submit-", "prompt-turn-")


@pytest.fixture(autouse=True)
def _join_leaked_turn_workers(monkeypatch):
    """Join the turn workers a test's prompt.submit started, before its patches undo.

    prompt.submit returns while its worker (``prompt-submit-<sid>``, which then
    starts ``prompt-turn-<sid>``) is still running. Unjoined, a worker finishes
    in a LATER test and emits message.start/complete through that test's
    patched ``_emit`` (seen as a stray ``row-id-trunc-sid`` message.complete in
    test_notification_poller_live_loop_requeues_foreign_completion_for_owner
    under load). Depending on ``monkeypatch`` makes this teardown run first, so
    the worker ends against the fakes of the test that started it.
    """
    # A worker wedged by an earlier test is that test's failure, not every later test's.
    inherited = set(threading.enumerate())
    yield

    def own_workers():
        return [
            t
            for t in threading.enumerate()
            if t not in inherited and t.name.startswith(_TURN_WORKER_PREFIXES) and t.is_alive()
        ]

    deadline = time.monotonic() + DEFAULT_TIMEOUT
    while workers := own_workers():
        for thread in workers:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if time.monotonic() >= deadline and (still := own_workers()):
            pytest.fail(f"turn workers still running at teardown: {[t.name for t in still]}")
