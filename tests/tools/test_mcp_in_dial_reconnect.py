"""A reconnect requested mid-dial must not tear down the session it produced.

`_signal_reconnect_and_wait` clears `_ready` and sets `_reconnect_event`, then
polls for a fresh session. When the lifecycle loop is ALREADY rebuilding, that
request lands mid-dial: the dial completes, publishes the new session and sets
`_ready`, and `_wait_for_lifecycle_event()` is entered with the event still set
-- so `asyncio.wait` returns "reconnect" immediately and destroys the session
that was established microseconds earlier.

Meanwhile `_wait_for_server_session_ready()` only checks
`session is not old_session and _ready.is_set()`, so it certifies that same
doomed session and the single session-expired retry is spent on a transport
that is already unwinding. Observed in production as, milliseconds apart:

    reconnect requested - tearing down HTTP session
    retry after session reconnect failed: Connection closed

The request is satisfied BY that dial, so it must be consumed when the session
becomes ready rather than left to fire against it.
"""
import asyncio

import pytest

from tools.mcp_tool import MCPServerTask


def _ready_server(name: str = "test") -> MCPServerTask:
    """A task in the state a dial leaves behind: session published, _ready set."""
    srv = MCPServerTask(name)
    srv._config = {"keepalive_interval": 300}
    srv.session = object()
    srv._ready.set()
    return srv


@pytest.mark.asyncio
async def test_in_dial_reconnect_does_not_tear_down_the_new_session():
    """An event set BEFORE readiness is consumed, not acted on."""
    srv = _ready_server()
    srv._reconnect_event.set()          # requested while the dial was in flight

    task = asyncio.ensure_future(srv._wait_for_lifecycle_event())
    await asyncio.sleep(0.1)

    assert not task.done(), (
        "brand-new session torn down by the reconnect request that built it"
    )
    assert not srv._reconnect_event.is_set(), (
        "the in-dial request must be consumed so it cannot fire later either"
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_reconnect_requested_after_ready_still_reconnects():
    """The real signal path (keepalive failure, OAuth recovery) still works."""
    srv = _ready_server()

    task = asyncio.ensure_future(srv._wait_for_lifecycle_event())
    await asyncio.sleep(0.05)           # let it enter the wait
    srv._reconnect_event.set()

    assert await asyncio.wait_for(task, timeout=3.0) == "reconnect"


@pytest.mark.asyncio
async def test_shutdown_after_ready_still_shuts_down():
    srv = _ready_server()

    task = asyncio.ensure_future(srv._wait_for_lifecycle_event())
    await asyncio.sleep(0.05)
    srv._shutdown_event.set()

    assert await asyncio.wait_for(task, timeout=3.0) == "shutdown"


@pytest.mark.asyncio
async def test_shutdown_still_wins_when_both_set_after_ready():
    srv = _ready_server()

    task = asyncio.ensure_future(srv._wait_for_lifecycle_event())
    await asyncio.sleep(0.05)
    srv._reconnect_event.set()
    srv._shutdown_event.set()

    assert await asyncio.wait_for(task, timeout=3.0) == "shutdown"
