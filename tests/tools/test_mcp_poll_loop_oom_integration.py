"""End-to-end coverage for the MCP poll-loop OOM spin (#63892).

The unit tests in ``test_mcp_tool.py`` hand-construct completed futures to lock
the ``_run_on_mcp_loop`` contract deterministically. This complements them by
exercising the actual field trigger through the *live* MCP loop: an inner
``asyncio.wait_for`` expiry produces a real ``TimeoutError`` stored on a real
future scheduled via ``run_coroutine_threadsafe``. On Python >= 3.8 that class
is the builtin ``TimeoutError``, so before the fix the poll loop swallowed the
completed future's stored exception and spun with no sleep -- burning CPU,
growing the exception traceback ~108 MB/s until the gateway OOM'd, and finally
masking the real error behind the generic "MCP call timed out after <full
timeout>" wrapper.

The fixed loop must surface the real exception once, promptly -- long before
the outer deadline.
"""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.fixture
def mcp_loop():
    import tools.mcp_tool as mcp_tool

    mcp_tool._ensure_mcp_loop()
    yield mcp_tool
    mcp_tool._stop_mcp_loop()


def test_inner_wait_for_timeout_surfaces_promptly_without_spinning(mcp_loop):
    async def inner():
        # Real TimeoutError from wait_for, completing far before the outer
        # deadline below -- the exact shape of an MCP call_tool wrapped in the
        # server's configured mcp_servers.<srv>.timeout.
        await asyncio.wait_for(asyncio.sleep(60), timeout=0.05)

    start = time.monotonic()
    with pytest.raises(TimeoutError) as exc:
        mcp_loop._run_on_mcp_loop(inner, timeout=10)
    elapsed = time.monotonic() - start

    # Fixed: the real inner TimeoutError surfaces within a couple poll ticks.
    # Broken: the loop swallows it and spins with no sleep until the 10s
    # deadline, then raises the generic wrapper message instead. A generous
    # (>= 2s) bound keeps this stable on a slow CI runner while still failing
    # hard on a spin-to-deadline regression.
    assert elapsed < 5.0, (
        f"poll loop spun for {elapsed:.1f}s instead of resolving the completed "
        "future once (#63892 regression)"
    )
    assert "MCP call timed out after" not in str(exc.value), (
        "the real inner TimeoutError must surface, not the outer poll deadline"
    )


def test_successful_call_still_returns_through_real_loop(mcp_loop):
    """Guard the done-future path against regressing normal success returns."""

    async def inner():
        await asyncio.sleep(0.25)  # first polls time out while still pending
        return {"ok": True}

    assert mcp_loop._run_on_mcp_loop(inner, timeout=10) == {"ok": True}
