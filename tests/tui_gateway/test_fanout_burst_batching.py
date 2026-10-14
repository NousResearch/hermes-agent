"""The per-peer fanout drain sends what is queued as ONE batch on the socket's loop, not one
cross-thread round trip per frame: a producer with no per-frame I/O (an in-process turn streaming
tokens) otherwise outruns a healthy WebSocket and overflows its bounded mailbox
(``session.replay_gap`` mid-turn). Real asyncio loop thread, real FanoutTransport; the socket
only records frames and holds its first send until the test has queued a backlog."""
import asyncio
import threading

from tests.waiting import wait_until
from tui_gateway.transport import FanoutTransport


class _LoopSocket:
    """Duck-typed like WSTransport: its async sends complete on ``_loop``."""

    def __init__(self, loop):
        self._loop = loop
        self.frames = []
        self.sends = 0
        self.first_send_started = threading.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def _send(self, frames):
        self.sends += 1
        self.first_send_started.set()
        await self.release.wait()
        self.frames.extend(frames)
        return True

    async def write_async(self, frame):
        return await self._send([frame])

    async def write_many_async(self, frames):
        return await self._send(list(frames))

    def write(self, frame):
        raise AssertionError('a loop-bound socket is written from its loop, not the drain thread')

    def abort(self):
        self.closed = True

    close = abort


def _delta(index):
    return {'jsonrpc': '2.0', 'method': 'event', 'params': {
        'type': 'message.delta', 'session_id': 's', 'payload': {'text': str(index)}}}


def test_the_drain_sends_a_queued_backlog_as_one_batch():
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    try:
        socket = _LoopSocket(loop)
        fanout = FanoutTransport(socket)
        gaps = []
        fanout.write(_delta(0), overflow=lambda transport: gaps.append(transport) or None)
        assert socket.first_send_started.wait(10), 'the drain never reached the socket'
        backlog = FanoutTransport._MAX_PENDING_FRAMES - 1
        for index in range(1, backlog + 1):
            fanout.write(_delta(index), overflow=lambda transport: gaps.append(transport) or None)
        loop.call_soon_threadsafe(socket.release.set)
        wait_until(lambda: len(socket.frames) == backlog + 1, timeout=30, what='the backlog delivered')
        assert not gaps and not socket.closed
        assert [f['params']['payload']['text'] for f in socket.frames] == [str(i) for i in range(backlog + 1)]
        # One send in flight when the backlog formed, then the whole backlog in one batch (a
        # per-frame drain makes ``backlog + 1`` sends, one loop round trip each).
        assert socket.sends <= 2, f'{socket.sends} socket round trips for {backlog + 1} frames'
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(5)
        loop.close()
