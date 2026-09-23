"""A stalled ``send_text`` must have a bounded lifetime (#106369).

``_safe_send_many`` awaits the socket while holding the connection-wide ``_send_lock``. Without a
deadline, one send parked by socket backpressure trapped every later event and RPC reply behind the
lock on an apparently open connection, so reconnect recovery never started.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tui_gateway import ws as ws_mod
from tui_gateway.ws import WSTransport


class _StalledWS:
    """``send_text`` never completes (kernel backpressure); ``close`` is observable."""

    def __init__(self) -> None:
        self.closed_with: list[int] = []
        self.send_started = asyncio.Event()
        self._release = asyncio.Event()

    async def send_text(self, line: str) -> None:
        self.send_started.set()
        await self._release.wait()

    async def close(self, code: int = 1000) -> None:
        self.closed_with.append(code)
        self._release.set()

    def release(self) -> None:
        self._release.set()


class _FastWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, line: str) -> None:
        self.sent.append(line)


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

    asyncio.run(_run())


@pytest.mark.parametrize("streaming", [False, True])
def test_stalled_send_backlog_is_bounded_across_all_buffers(monkeypatch, streaming):
    monkeypatch.setattr(ws_mod, "_WS_MAX_PENDING_FRAMES", 100, raising=False)
    monkeypatch.setattr(ws_mod, "_WS_MAX_PENDING_BYTES", 256, raising=False)
    monkeypatch.setattr(ws_mod, "_WS_SEND_DEADLINE_S", 60)

    async def _run() -> None:
        ws = _StalledWS()
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:1")
        first = asyncio.create_task(transport.write_async({"id": "history", "result": "é" * 200}))
        await ws.send_started.wait()
        assert not transport.closed, "an oversized first frame must remain admissible"

        frame = {"method": "event", "params": {"type": "message.delta", "text": "x"}}
        if not streaming:
            frame = {"id": "queued", "result": "x"}
        try:
            admitted = [transport.write(frame) for _ in range(20)]
            assert False in admitted, "the over-budget additional frame must be rejected"
            assert transport._pending_frame_count <= 100 and transport._pending_byte_count >= 0
            await asyncio.wait_for(first, timeout=1.0)
            await asyncio.sleep(0)
            assert ws.closed_with == [1011]
            assert (transport._pending_frame_count, transport._pending_byte_count) == (0, 0)
        finally:
            ws.release()
            await asyncio.wait_for(first, timeout=1.0)

    asyncio.run(_run())


def test_progress_then_final_ordering_preserved_on_healthy_socket():
    async def _run() -> None:
        ws = _FastWS()
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:1")
        token = {"method": "event", "params": {"type": "message.delta", "text": "é"}}
        assert transport.write(token)
        token_line = json.dumps(token, ensure_ascii=False)
        assert transport._pending_frame_count == 1
        assert transport._pending_byte_count == len(token_line.encode("utf-8"))
        assert await transport.write_async({"method": "event", "params": {"type": "message.complete"}})
        assert await transport.write_async({"id": "submit", "result": {"status": "done"}})
        assert not transport.closed
        assert [json.loads(s).get("params", {}).get("type", "reply") for s in ws.sent] == [
            "message.delta", "message.complete", "reply",
        ]
        assert transport._pending_frame_count == 0
        assert transport._pending_byte_count == 0

    asyncio.run(_run())
