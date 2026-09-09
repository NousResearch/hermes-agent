"""A stalled ``send_text`` must not hold the writer lock forever (#106369).

Without a send deadline, socket backpressure parks ``_safe_send_many`` inside
``_send_lock`` and ``_closed`` never latches, so reconnect recovery cannot
start. With the fix, a stalled send times out, latches the transport closed,
and a queued second batch returns immediately instead of blocking forever.
"""

from __future__ import annotations

import asyncio

import tui_gateway.ws as ws_mod
from tui_gateway.ws import WSTransport


class _StalledWS:
    """``send_text`` never completes — models socket backpressure."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._release = asyncio.Event()

    async def send_text(self, line: str) -> None:
        await self._release.wait()  # never set during the test
        self.sent.append(line)

    async def close(self, code: int = 1000) -> None:
        self._release.set()


class _FastWS:
    """``send_text`` completes immediately — guards against the timeout
    breaking the happy path."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, line: str) -> None:
        self.sent.append(line)


def test_stalled_send_latches_closed() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        ws = _StalledWS()
        transport = WSTransport(ws, loop, peer="127.0.0.1:1")
        orig = ws_mod._WS_WRITE_TIMEOUT_S
        ws_mod._WS_WRITE_TIMEOUT_S = 0.05
        try:
            await transport._safe_send_many(["first"])
        finally:
            ws_mod._WS_WRITE_TIMEOUT_S = orig
        assert transport.closed is True, "stalled send must latch the transport closed"
        assert ws.sent == [], "the stalled frame must not be recorded as sent"

    asyncio.run(_run())


def test_queued_batch_returns_immediately_after_timeout() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        ws = _StalledWS()
        transport = WSTransport(ws, loop, peer="127.0.0.1:1")
        orig = ws_mod._WS_WRITE_TIMEOUT_S
        ws_mod._WS_WRITE_TIMEOUT_S = 0.05
        try:
            await transport._safe_send_many(["first"])
            # _closed is now True; the second batch must not queue on _send_lock.
            await transport._safe_send_many(["second"])
        finally:
            ws_mod._WS_WRITE_TIMEOUT_S = orig
        assert transport.closed is True
        assert ws.sent == []

    asyncio.run(_run())


def test_normal_send_completes_within_timeout() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        ws = _FastWS()
        transport = WSTransport(ws, loop, peer="127.0.0.1:1")
        orig = ws_mod._WS_WRITE_TIMEOUT_S
        ws_mod._WS_WRITE_TIMEOUT_S = 0.05
        try:
            await transport._safe_send_many(["a", "b", "c"])
        finally:
            ws_mod._WS_WRITE_TIMEOUT_S = orig
        assert transport.closed is False, "a fast send must not latch the transport closed"
        assert ws.sent == ["a", "b", "c"]

    asyncio.run(_run())
