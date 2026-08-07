"""Keepalive must not probe while an RPC is in flight (``_rpc_lock`` held).

A single-threaded stdio server (e.g. the mac_worker facade) reads one
message, dispatches it, and only then reads the next — so it cannot
answer ``ping`` while executing a long tool call. Probing mid-call
guarantees a false-positive reconnect that tears down the transport and
kills the very call it blames (delegation deleg_e8af57bc, 2026-08-02: a
240s ``mac_worker_wait`` was destroyed at t+48s and its completion
receipt never reached the Pi). The in-flight round-trip is itself the
liveness signal; its success path marks the session proven.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest


def test_keepalive_skips_probe_while_rpc_in_flight(monkeypatch):
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.01)

    async def _scenario():
        server = MCPServerTask("srv")
        server._config = {"command": "x", "keepalive_interval": 0.02}
        ping = AsyncMock(return_value=None)
        server.session = AsyncMock()
        server.session.send_ping = ping

        async with server._rpc_lock:
            lifecycle = asyncio.ensure_future(server._wait_for_lifecycle_event())
            # Many keepalive intervals elapse while the RPC is in flight.
            await asyncio.sleep(0.3)
            assert ping.await_count == 0, (
                f"probe fired {ping.await_count}x while _rpc_lock was held"
            )
            assert not server._reconnect_event.is_set()
            assert not lifecycle.done()

        # Lock released — probing must resume.
        await asyncio.sleep(0.3)
        assert ping.await_count > 0, "probe never resumed after lock release"
        assert not server._reconnect_event.is_set()

        server._shutdown_event.set()
        assert await asyncio.wait_for(lifecycle, timeout=5) == "shutdown"

    asyncio.run(_scenario())


def test_keepalive_still_reconnects_on_real_ping_failure(monkeypatch):
    """The guard must not mask genuine idle-session death."""
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.01)

    async def _scenario():
        server = MCPServerTask("srv")
        server._config = {"command": "x", "keepalive_interval": 0.02}
        server.session = AsyncMock()
        server.session.send_ping = AsyncMock(side_effect=ConnectionError("dead"))

        result = await asyncio.wait_for(
            server._wait_for_lifecycle_event(), timeout=5
        )
        assert result == "reconnect"

    asyncio.run(_scenario())
