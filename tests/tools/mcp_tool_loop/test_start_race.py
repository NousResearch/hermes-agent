"""``_ensure_mcp_loop`` must return a loop that is already running.

``Thread.start()`` returns once the new thread has begun, not once ``loop.run_forever`` has
marked the loop running. On a GIL build the new thread usually wins that race, but a loaded
host or a free-threaded build hands the caller the gap: ``loop.is_running()`` is still False.
In that gap the next ``_run_on_mcp_loop`` raises "MCP event loop is not running", a second
``_ensure_mcp_loop`` builds a second loop and orphans the first, and ``_stop_mcp_loop`` drives
the loop from the calling thread instead of draining on the loop's own.

The race is made deterministic here: the loop thread starts at once but is held before
``run_forever`` until the test has looked (or ``HOLD`` passes, which is what lets a fixed
``_ensure_mcp_loop`` return). That hold is exactly the window a busy scheduler can open.
"""
import asyncio
import threading

import pytest

# _run_on_mcp_loop imports these on its first call; a cold import must not use up the hold.
import agent.async_utils
import tools.interrupt
from tools import mcp_tool as _origin
from tools import mcp_tool_lifecycle as _lifecycle
from tools import mcp_tool_loop as _loop_mod

# Upper bound on the hold; a fixed tree waits it out once per start, so keep it short.
HOLD = 2.0


class _LateThread(threading.Thread):
    """Has begun when ``start()`` returns; reaches its target once ``release`` is set or ``HOLD``
    has passed."""

    release = threading.Event()

    def run(self):
        self.release.wait(HOLD)
        super().run()


@pytest.fixture
def late_loop_thread(monkeypatch):
    """No MCP loop at the start; the loop thread starts late; every loop built is torn down."""
    made = []
    real_new_event_loop = asyncio.new_event_loop

    def tracking_new_event_loop():
        loop = real_new_event_loop()
        made.append(loop)
        return loop

    class _Asyncio:
        new_event_loop = staticmethod(tracking_new_event_loop)

        def __getattr__(self, name):
            return getattr(asyncio, name)

    class _Threading:
        Thread = _LateThread

        def __getattr__(self, name):
            return getattr(threading, name)

    monkeypatch.setattr(_LateThread, "release", threading.Event())
    monkeypatch.setattr(_loop_mod, "asyncio", _Asyncio())
    monkeypatch.setattr(_loop_mod, "threading", _Threading())
    monkeypatch.setattr(_origin, "_mcp_loop", None)
    monkeypatch.setattr(_origin, "_mcp_thread", None)
    monkeypatch.setattr(_lifecycle, "_kill_orphaned_mcp_children", lambda *a, **k: None)
    before = set(threading.enumerate())
    yield made
    # A loop whose thread is still held has not started yet: let it start; the queued stop ends it.
    _LateThread.release.set()
    for loop in made:
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
    for thread in set(threading.enumerate()) - before:
        if thread.name == "mcp-event-loop":
            thread.join(timeout=5)
    for loop in made:
        if not loop.is_running() and not loop.is_closed():
            loop.close()


def _halt():
    loop, thread = _origin._mcp_loop, _origin._mcp_thread
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()
    _origin._mcp_loop = _origin._mcp_thread = None


def test_ensure_returns_a_loop_that_is_running(late_loop_thread):
    _loop_mod._ensure_mcp_loop()
    assert _origin._mcp_loop.is_running(), "_ensure_mcp_loop returned before the loop was running"
    _halt()


def test_a_call_right_after_the_start_is_served(late_loop_thread):
    async def seven():
        return 7

    _loop_mod._ensure_mcp_loop()
    assert _loop_mod._run_on_mcp_loop(seven, timeout=10) == 7
    _halt()


def test_a_second_ensure_reuses_the_first_loop(late_loop_thread):
    _loop_mod._ensure_mcp_loop()
    _loop_mod._ensure_mcp_loop()
    assert len(late_loop_thread) == 1, "a second loop was built and the first one orphaned"
    _halt()


def test_a_stop_right_after_the_start_drains_on_the_loop_thread(late_loop_thread, monkeypatch):
    drained_on = []
    real_drain = _lifecycle._drain_mcp_loop_tasks

    async def recording_drain(**kwargs):
        drained_on.append(threading.get_ident())
        await real_drain(**kwargs)

    def refuse_to_drive_the_loop(coro):
        # What _stop_mcp_loop does with a loop it believes is stopped: drive it from the CALLING
        # thread while the loop thread is about to run it too. Record the call, never make it.
        drained_on.append(("run_until_complete", threading.get_ident()))
        coro.close()

    monkeypatch.setattr(_lifecycle, "_drain_mcp_loop_tasks", recording_drain)
    _loop_mod._ensure_mcp_loop()
    loop, thread = _origin._mcp_loop, _origin._mcp_thread
    loop.run_until_complete = refuse_to_drive_the_loop
    assert _loop_mod._stop_mcp_loop()
    assert drained_on == [thread.ident], f"drained on {drained_on}, not on the loop thread {thread.ident}"
    assert not thread.is_alive()
