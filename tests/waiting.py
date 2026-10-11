"""Wait for a condition instead of the clock (the HX013 fix).

A fixed ``time.sleep(0.2)`` before an assertion is a bet that the code under test finishes within
0.2 s; on a loaded CI runner it loses and the test flakes. Poll the condition with a generous
deadline instead: a fast run returns in milliseconds, a slow one still passes, and a real hang
fails with a message naming what never happened.

    from tests.waiting import wait_until, async_wait_until

    wait_until(lambda: worker.done, what="worker finished")
    row = wait_until(lambda: db.get(key), timeout=10)      # returns the truthy value
    await async_wait_until(lambda: adapter.sent, what="message sent")

Prefer an Event the code under test sets (``threading.Event`` / ``asyncio.Event`` with
``.wait(timeout=...)``) when there is one; poll when the only signal is observable state.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

DEFAULT_TIMEOUT = 10.0  # a deadline, not a delay: only a hang ever waits this long
DEFAULT_INTERVAL = 0.01


def _timed_out(what: str | None, timeout: float) -> AssertionError:
    return AssertionError(f"timed out after {timeout}s waiting for {what or 'the condition'}")


def wait_until(predicate: Callable[[], T], timeout: float = DEFAULT_TIMEOUT,
               interval: float = DEFAULT_INTERVAL, what: str | None = None) -> T:
    """Return ``predicate()``'s first truthy value; raise AssertionError after ``timeout`` s."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise _timed_out(what, timeout)
        time.sleep(interval)


async def async_wait_until(predicate: Callable[[], T], timeout: float = DEFAULT_TIMEOUT,
                           interval: float = DEFAULT_INTERVAL, what: str | None = None) -> T:
    """``wait_until`` for a coroutine: yields to the event loop between polls."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise _timed_out(what, timeout)
        await asyncio.sleep(interval)
