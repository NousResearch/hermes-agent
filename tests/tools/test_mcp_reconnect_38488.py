"""Regression test for #38488 — MCP server never permanently gives up."""
import asyncio
import pytest
from unittest.mock import patch, AsyncMock


def test_run_does_not_give_up_after_max_reconnect_retries():
    """After _MAX_RECONNECT_RETRIES failures, run() must keep slow-reprobing
    rather than returning, so a backend that comes back later self-heals.
    Regression for #38488.
    """
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("flaky")
    # Force HTTP path; we'll patch _run_http directly so config shape is moot.

    # Mark _ready so we're in the "reconnect" branch (not initial connect)
    server._ready.set()

    call_count = {"n": 0}
    # Fail 7 times (more than _MAX_RECONNECT_RETRIES=5), then succeed by
    # returning cleanly (transport done -> loop continues -> shutdown set).
    async def fake_run_http(self, _cfg):
        call_count["n"] += 1
        if call_count["n"] <= 7:
            raise ConnectionError(f"boom {call_count['n']}")
        # On the 8th attempt the backend is "back"; return cleanly.
        # Trigger shutdown so the run loop exits after this success.
        server._shutdown_event.set()
        return

    with patch.object(MCPServerTask, "_run_http", new=fake_run_http), \
         patch.object(MCPServerTask, "_is_http", lambda self: True), \
         patch("tools.mcp_tool._MAX_BACKOFF_SECONDS", 0.01):
        async def _go():
            await asyncio.wait_for(server.run({"url": "http://x"}), timeout=5)
        asyncio.run(_go())

    # All 8 attempts must have happened — i.e. the server didn't return
    # after attempt 6.
    assert call_count["n"] == 8, (
        f"Expected slow re-probe to keep retrying past _MAX_RECONNECT_RETRIES, "
        f"but run() exited after only {call_count['n']} attempts"
    )


def test_successful_cycle_resets_retry_budget():
    """After a successful transport cycle, future failures should start
    fresh fast-retry counters (not inherit old retries). Regression for #38488.
    """
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("recovering")
    server._ready.set()

    phases = {"i": 0}
    # Pattern: fail 3x, succeed (reconnect_event triggered), fail 3x more,
    # then shutdown. If counters reset, second batch is "fast" attempts 1-3.
    async def fake_run_http(self, _cfg):
        phases["i"] += 1
        i = phases["i"]
        if i in (1, 2, 3, 5, 6, 7):
            raise ConnectionError(f"boom {i}")
        if i == 4:
            # Successful cycle: transport returns cleanly, loop continues.
            return
        # i == 8: shutdown
        server._shutdown_event.set()
        return

    with patch.object(MCPServerTask, "_run_http", new=fake_run_http), \
         patch.object(MCPServerTask, "_is_http", lambda self: True), \
         patch("tools.mcp_tool._MAX_BACKOFF_SECONDS", 0.01):
        async def _go():
            await asyncio.wait_for(server.run({"url": "http://x"}), timeout=5)
        asyncio.run(_go())

    assert phases["i"] == 8
