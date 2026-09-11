"""A stalled ``send_text`` must have a bounded lifetime (#106369, #108147).

``_safe_send_many`` awaits the socket while holding the connection-wide ``_send_lock``. Without a
deadline, one send parked by socket backpressure trapped every later event and RPC reply behind the
lock on an apparently open connection, so reconnect recovery never started. The deadline alone is
not a memory bound: later batches can still retain serialized frames while they wait for that lock.
"""

from __future__ import annotations

import asyncio
import json

from tui_gateway.ws import WSTransport


class _StalledWS:
    """``send_text`` never completes (kernel backpressure); ``close`` is observable."""

    def __init__(self) -> None:
        self.closed_with: list[int] = []
        self.entered = asyncio.Event()
        self._release = asyncio.Event()

    async def send_text(self, line: str) -> None:
        self.entered.set()
        await self._release.wait()

    async def close(self, code: int = 1000) -> None:
        self.closed_with.append(code)
        self._release.set()


class _FastWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, line: str) -> None:
        self.sent.append(line)


async def _wait_for_accounting_release(transport: WSTransport) -> None:
    for _ in range(50):
        if transport._pending_send_frames == 0:
            return
        await asyncio.sleep(0)
    assert transport._pending_send_frames == 0


def test_stalled_send_closes_socket_and_releases_queued_reply(monkeypatch):
    # raising=False: on a base without the deadline the test must fail on the SYMPTOM (sends never terminate).
    monkeypatch.setattr("tui_gateway.ws._WS_SEND_DEADLINE_S", 0.05, raising=False)

    async def _run() -> None:
        ws = _StalledWS()
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:1")
        progress = asyncio.create_task(transport.write_async({"method": "event", "params": {"type": "tool.progress"}}))
        await asyncio.sleep(0)  # progress is now inside send_text, holding _send_lock
        reply = asyncio.create_task(transport.write_async({"id": "submit", "result": {"status": "streaming"}}))
        # The loop stays responsive; both sends must still terminate within the deadline (not the 2s cap).
        results = await asyncio.wait_for(asyncio.gather(progress, reply), timeout=2.0)
        assert results == [False, False], "a send that missed the deadline must report failure, not success"
        assert transport.closed, "the transport must latch closed so handle_ws teardown/reconnect can run"
        await asyncio.sleep(0)  # let the scheduled close task run
        assert ws.closed_with == [1011], "the stalled socket must be closed, not left half-open"
        await _wait_for_accounting_release(transport)
        assert transport._pending_send_bytes == 0

    asyncio.run(_run())


def test_stalled_send_backlog_is_frame_bounded(monkeypatch):
    monkeypatch.setattr("tui_gateway.ws._WS_SEND_DEADLINE_S", 5.0)
    monkeypatch.setattr("tui_gateway.ws._WS_BACKLOG_MAX_FRAMES", 3)
    monkeypatch.setattr("tui_gateway.ws._WS_BACKLOG_MAX_BYTES", 1_000_000)

    async def _run() -> None:
        ws = _StalledWS()
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:2")
        active = asyncio.create_task(transport.write_async({"idx": 0}))
        await ws.entered.wait()  # first frame holds _send_lock

        # Two more frames reach the configured occupancy; the next is rejected immediately instead
        # of creating another task that retains its serialized batch until the 5s socket deadline.
        assert [transport.write({"idx": i}) for i in range(1, 4)] == [True, True, False]
        assert await asyncio.wait_for(active, timeout=1.0) is False
        await _wait_for_accounting_release(transport)

        assert transport.closed
        assert ws.closed_with == [1011]
        assert transport._pending_send_bytes == 0

    asyncio.run(_run())


def test_single_oversized_frame_cannot_grow_a_backlog(monkeypatch):
    monkeypatch.setattr("tui_gateway.ws._WS_SEND_DEADLINE_S", 5.0)
    monkeypatch.setattr("tui_gateway.ws._WS_BACKLOG_MAX_FRAMES", 256)
    monkeypatch.setattr("tui_gateway.ws._WS_BACKLOG_MAX_BYTES", 1)

    async def _run() -> None:
        ws = _StalledWS()
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:3")

        # A first frame larger than the cap is already materialized and remains sendable. Nothing else
        # may queue behind it, so an isolated large history response works without permitting growth.
        active = asyncio.create_task(transport.write_async({"payload": "large"}))
        await ws.entered.wait()
        assert transport.write({"payload": "second"}) is False
        assert await asyncio.wait_for(active, timeout=1.0) is False
        await _wait_for_accounting_release(transport)

        assert transport.closed
        assert ws.closed_with == [1011]
        assert transport._pending_send_bytes == 0

    asyncio.run(_run())


def test_progress_then_final_ordering_preserved_on_healthy_socket():
    async def _run() -> None:
        ws = _FastWS()
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:1")
        assert await transport.write_async({"method": "event", "params": {"type": "tool.progress"}})
        assert await transport.write_async({"method": "event", "params": {"type": "message.complete"}})
        assert await transport.write_async({"id": "submit", "result": {"status": "done"}})
        assert not transport.closed
        assert transport._pending_send_frames == 0
        assert transport._pending_send_bytes == 0
        assert [json.loads(s).get("params", {}).get("type", "reply") for s in ws.sent] == [
            "tool.progress", "message.complete", "reply",
        ]

    asyncio.run(_run())
