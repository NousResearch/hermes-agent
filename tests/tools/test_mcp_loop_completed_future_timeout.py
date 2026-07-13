"""Regression: a completed future whose stored exception is a real
TimeoutError must surface once, promptly — not spin.

On Py>=3.8 concurrent.futures.TimeoutError IS builtin TimeoutError, so
_run_on_mcp_loop's poll ``except concurrent.futures.TimeoutError`` also
catches a DONE future whose coroutine raised a real TimeoutError (e.g. an
inner asyncio.wait_for hitting mcp_servers.<srv>.timeout). Before the fix,
re-resolving the done future re-raised the same exception every poll with no
sleep — a tight spin appending frames to __traceback__ unboundedly, growing
RSS until the gateway OOM'd. The loop now resolves a done future once.
"""
import concurrent.futures
import time

import pytest


@pytest.fixture
def mcp_loop():
    import tools.mcp_tool as mcp_tool

    mcp_tool._ensure_mcp_loop()
    yield mcp_tool
    mcp_tool._stop_mcp_loop()


def test_inner_timeout_surfaces_once_without_spinning(mcp_loop):
    import asyncio

    async def inner():
        # Completes (does not stay pending) with a real TimeoutError well
        # before the outer _run_on_mcp_loop deadline below.
        return await asyncio.wait_for(asyncio.sleep(60), timeout=0.2)

    start = time.monotonic()
    with pytest.raises((concurrent.futures.TimeoutError, TimeoutError)) as exc:
        # Generous outer timeout: a fixed loop returns the inner TimeoutError
        # in ~0.2s; the buggy spin would instead burn until this 10s deadline
        # and raise the "MCP call timed out after ..." wrapper message.
        mcp_loop._run_on_mcp_loop(inner, timeout=10)
    elapsed = time.monotonic() - start

    assert elapsed < 5, f"poll spun instead of resolving promptly ({elapsed:.1f}s)"
    assert "MCP call timed out after" not in str(exc.value)


def test_success_race_after_poll_timeout_returns_value(mcp_loop):
    """A future that completes successfully between a poll's timeout raise and
    the done() check must return its value — not a re-raised poll timeout."""
    import asyncio

    async def slow_ok():
        await asyncio.sleep(0.25)
        return "ok"

    # wait_timeout starts at 0.1s, so the first few polls time out while the
    # coroutine is still sleeping; the loop must eventually return the value.
    assert mcp_loop._run_on_mcp_loop(slow_ok, timeout=10) == "ok"
