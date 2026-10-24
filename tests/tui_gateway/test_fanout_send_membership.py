"""Session detach revokes queued fanout frames without closing a shared socket."""

import asyncio
import io
import json
import threading

import pytest

from tests.waiting import async_wait_until
from tui_gateway.transport import FanoutTransport, StdioTransport
from tui_gateway.ws import WSTransport


def _event(text, session_id="session"):
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "message.delta",
            "session_id": session_id,
            "payload": {"text": text},
        },
    }


class _Socket:
    def __init__(self, *held):
        self.frames = []
        self.entered = {text: asyncio.Event() for text in held}
        self.released = {text: asyncio.Event() for text in held}

    async def send_text(self, text):
        frame = json.loads(text)
        label = frame.get("params", {}).get("payload", {}).get("text")
        if label in self.entered:
            self.entered[label].set()
            await self.released[label].wait()
        self.frames.append(frame)


def _idle(fanout):
    with fanout._lock:
        return not any(peer.writing for peer in fanout._peers)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    ["detach", "reattach", "close", "overflow", "overflow-reattach", "overflow-detach"],
)
async def test_detach_discards_batch_waiting_for_send_lock(monkeypatch, action):
    socket = _Socket()
    transport = WSTransport(socket, asyncio.get_running_loop())
    fanout = FanoutTransport(transport)
    other_session = FanoutTransport(transport)
    entered = asyncio.Event()
    send_many = transport._safe_send_many

    async def observed_send(lines, **kwargs):
        entered.set()
        await send_many(lines, **kwargs)

    monkeypatch.setattr(transport, "_safe_send_many", observed_send)
    # Keep another producer's buffered token in this exact captured send batch.
    monkeypatch.setattr(transport, "_arm_token_flush", lambda: None)
    buffered = _event("buffered", "other-session")
    reply = {"jsonrpc": "2.0", "id": "private-reply", "result": {"ok": True}}
    notice = _event("resume-required")
    await transport._send_lock.acquire()
    try:
        assert transport.write(buffered)
        assert fanout.write(_event("stale"))
        await asyncio.wait_for(entered.wait(), timeout=10)
        if action.startswith("overflow"):
            monkeypatch.setattr(fanout, "_MAX_PENDING_FRAMES", 2)
            for index in range(3):
                fanout.write(_event(str(index)), overflow=lambda _peer: notice)
            assert not fanout.contains(transport)
            if action == "overflow-reattach":
                assert fanout.attach(transport)
                assert fanout.write(_event("fresh"))
            elif action == "overflow-detach":
                assert not fanout.detach(transport)
        elif action == "close":
            fanout.close()
        else:
            assert fanout.detach(transport)
            if action == "reattach":
                assert fanout.attach(transport)
                assert fanout.write(_event("fresh"))
        assert other_session.write(_event("other", "other-session"))
        response = asyncio.create_task(transport.write_async(reply))
        transport._send_lock.release()
        assert await asyncio.wait_for(response, timeout=10)
        await async_wait_until(
            lambda: _idle(fanout) and _idle(other_session),
            what="both fanout writers drained",
        )
        expected = [buffered, reply, _event("other", "other-session")]
        if action in {"reattach", "overflow-reattach"}:
            expected.append(_event("fresh"))
        if action == "overflow":
            expected.append(notice)
        assert sorted(map(json.dumps, socket.frames)) == sorted(
            map(json.dumps, expected)
        )
        assert not transport.closed
    finally:
        fanout.close()
        other_session.close()
        if transport._send_lock.locked():
            transport._send_lock.release()
        transport.close()
        await async_wait_until(
            lambda: _idle(fanout) and _idle(other_session), what="fanout cleanup"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reattach", [False, True])
async def test_detach_discards_unsent_tail_of_captured_batch(reattach):
    socket = _Socket("seed", "first")
    transport = WSTransport(socket, asyncio.get_running_loop())
    fanout = FanoutTransport(transport)
    reply = {"jsonrpc": "2.0", "id": "reply", "result": {"ok": True}}
    try:
        assert fanout.write(_event("seed"))
        await asyncio.wait_for(socket.entered["seed"].wait(), timeout=10)
        assert fanout.write(_event("first"))
        assert fanout.write(_event("stale-tail"))
        socket.released["seed"].set()
        await asyncio.wait_for(socket.entered["first"].wait(), timeout=10)
        assert fanout.detach(transport)
        if reattach:
            assert fanout.attach(transport)
            assert fanout.write(_event("fresh"))
        response = asyncio.create_task(transport.write_async(reply))
        socket.released["first"].set()
        assert await asyncio.wait_for(response, timeout=10)
        await async_wait_until(
            lambda: _idle(fanout), what="captured fanout batch drained"
        )
        # A send already handed to the socket can finish, but its unsent sibling cannot.
        expected = [_event("seed"), _event("first"), reply]
        if reattach:
            expected.append(_event("fresh"))
        assert sorted(map(json.dumps, socket.frames)) == sorted(
            map(json.dumps, expected)
        )
        assert not transport.closed
    finally:
        fanout.close()
        for released in socket.released.values():
            released.set()
        transport.close()
        await async_wait_until(lambda: _idle(fanout), what="fanout cleanup")


@pytest.mark.asyncio
@pytest.mark.parametrize("reattach", [False, True])
async def test_sync_peer_discards_unsent_tail_of_captured_batch(reattach):
    entered = {text: threading.Event() for text in ("seed", "first")}
    released = {text: threading.Event() for text in entered}

    class GatedStream(io.StringIO):
        def write(self, text):
            label = json.loads(text)["params"]["payload"]["text"]
            if label in entered:
                entered[label].set()
                assert released[label].wait(timeout=10), (
                    "test did not release the stream"
                )
            return super().write(text)

    stream = GatedStream()
    transport = StdioTransport(lambda: stream, threading.Lock())
    fanout = FanoutTransport(transport)
    try:
        assert fanout.write(_event("seed"))
        await async_wait_until(entered["seed"].is_set, what="first stdio send")
        assert fanout.write(_event("first"))
        assert fanout.write(_event("stale-tail"))
        released["seed"].set()
        await async_wait_until(entered["first"].is_set, what="captured stdio batch")
        assert fanout.detach(transport)
        if reattach:
            assert fanout.attach(transport)
            assert fanout.write(_event("fresh"))
        released["first"].set()
        await async_wait_until(lambda: _idle(fanout), what="stdio fanout drained")
        expected = [_event("seed"), _event("first")]
        if reattach:
            expected.append(_event("fresh"))
        assert list(map(json.loads, stream.getvalue().splitlines())) == expected
        assert not stream.closed
    finally:
        fanout.close()
        for gate in released.values():
            gate.set()
        await async_wait_until(lambda: _idle(fanout), what="stdio fanout cleanup")
        stream.close()
