"""Shared-session routing and backpressure, exercised through real OS pipes."""
import asyncio
import json
import os
import queue
import socket
import threading
import time
from contextlib import ExitStack, suppress

import pytest

from tui_gateway import server
from tui_gateway.transport import FanoutTransport, StdioTransport
from tui_gateway.ws import WSTransport


class RecordingTransport:
    """In-process Transport: optional write gate lets one peer overflow without a kernel pipe."""

    def __init__(self, *, delay=0.0):
        self.frames, self.closed, self.write_delay = [], False, delay
        self._released = threading.Event()

    def write(self, obj):
        if self.write_delay:
            self._released.wait(timeout=self.write_delay)
        self.frames.append(obj)
        return True

    def close(self):
        self.closed = True
        self._released.set()

    def release(self):
        self._released.set()


def _is_overflow_signal(obj) -> bool:
    blob = json.dumps(obj, ensure_ascii=False)
    return any(token in blob for token in ("detach", "overflow", "resubscribe", "replay", "backlog"))


def _await_frame_count(transport, count, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(transport.frames) >= count:
            return
        time.sleep(0.001)
    raise AssertionError(f"expected {count} frames, got {len(transport.frames)}")


def _overflow_slow_peer(fan, healthy, slow):
    """Emit until the slow mailbox overflows; pace healthy one receipt per emit."""
    for n in range(FanoutTransport._MAX_PENDING_FRAMES + 64):
        frame = {"params": {"type": "message.delta", "n": n}}
        assert fan.write(frame)
        _await_frame_count(healthy, n + 1)
        if not fan.contains(slow):
            return n
    raise AssertionError("slow peer never overflowed")


class PipeClient:
    def __init__(self, stack, *, reading=True):
        read_fd, write_fd = os.pipe()
        self.reader = stack.enter_context(os.fdopen(read_fd, "r", encoding="utf-8"))
        self.writer = os.fdopen(write_fd, "w", encoding="utf-8")
        stack.callback(self._cleanup)
        self.transport = StdioTransport(lambda: self.writer, threading.Lock())
        self._closed = False
        self.writes = 0
        self.frames = queue.Queue()
        if reading:
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()

    def _cleanup(self):
        with suppress(BrokenPipeError):
            self.writer.close()
        if hasattr(self, "thread"):
            self.thread.join(timeout=5)
            assert not self.thread.is_alive()

    def _read(self):
        for line in self.reader:
            self.frames.put(json.loads(line))

    def write(self, obj):
        self.writes += 1
        return self.transport.write(obj)

    def close(self):
        self._closed = True

    def receive(self):
        return self.frames.get(timeout=5)


class SocketClient(WSTransport):
    """Real WSTransport with its ASGI send backed by a kernel socketpair."""
    def __init__(self, stack, *, reading=True):
        self.reader, self.writer = socket.socketpair()
        self.writer.setblocking(False)
        loop = asyncio.new_event_loop()
        super().__init__(self, loop)
        self.writes = 0
        self.loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.frames = queue.Queue()
        self.read_thread = None
        if reading:
            self.read_thread = threading.Thread(target=self._read, daemon=True)
            self.read_thread.start()
        stack.callback(self._cleanup)

    async def send_text(self, payload):
        self.writes += 1
        await self._loop.sock_sendall(self.writer, (payload + "\n").encode())

    def _read(self):
        with self.reader.makefile("r", encoding="utf-8") as stream:
            for line in stream:
                self.frames.put(json.loads(line))

    def receive(self):
        return self.frames.get(timeout=5)

    def _cleanup(self):
        async def cancel_sends():
            self.close()
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.run_coroutine_threadsafe(cancel_sends(), self._loop).result(timeout=5)
        self.writer.close()
        if self.read_thread:
            self.read_thread.join(timeout=5)
            assert not self.read_thread.is_alive()
        self.reader.close()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self.loop_thread.join(timeout=5)
        self._loop.close()


def _session(transport):
    return dict(transport=transport, agent=None, session_key="fanout-invariant",
                history=[], history_lock=threading.Lock(), history_version=0,
                running=False, attached_images=[])


@pytest.mark.parametrize("attachment", ["direct", "flattened"])
def test_membership_preserves_terminal_delivery_and_revokes_departed_peers(monkeypatch, attachment):
    with ExitStack() as stack:
        a, b, stranger = [PipeClient(stack) for _ in range(3)]
        session = _session(a)
        monkeypatch.setitem(server._sessions, "fanout-invariant", session)
        newcomer = b if attachment == "direct" else FanoutTransport(a, b)
        assert server._attach_session_transport(session, newcomer)
        assert server._attach_session_transport(session, b)
        for kind in ("message.start", "message.delta", "message.complete"):
            server._emit(kind, "fanout-invariant", {"text": "α"})
            first, second = a.receive(), b.receive()
            assert first == second
            assert first["params"]["type"] == kind
        for client, allowed in ((a, True), (b, True), (stranger, False)):
            token = server.bind_transport(client)
            try:
                assert (server._current_session_steer_authority("fanout-invariant")[0] is client) == allowed
            finally:
                server.reset_transport(token)
        # RPC replies stay on their request transport, never the subscriber set.
        token = server.bind_transport(a)
        try:
            assert server.write_json({"jsonrpc": "2.0", "id": "private", "result": "owner only"})
        finally:
            server.reset_transport(token)
        assert a.receive()["id"] == "private"
        assert b.frames.empty()
        from gateway import browser_control_broker as broker_module
        monkeypatch.setattr(broker_module, "browser_control_enabled", lambda: True)
        for client in (a, b, stranger):
            client.auth_identity = {"user_id": "fanout-owner", "provider": "fixture"}
        session["profile"] = "default"

        def controller(client, action, **params):
            return server.dispatch({"jsonrpc": "2.0", "id": 1,
                                    "method": "browser.controller." + action,
                                    "params": {"session_id": "fanout-invariant", **params}}, client)

        registered = controller(a, "register", controller_id="invariant",
                                browser_profile_id="fixture", capabilities=["controller.noop"],
                                protocol_version=broker_module.BROWSER_CONTROL_PROTOCOL_VERSION)
        assert "result" in registered, registered
        try:
            assert controller(a, "heartbeat")["result"] == {"ok": True}
            assert controller(b, "heartbeat")["error"]["message"] == "controller is not owned by this transport"
            assert controller(stranger, "heartbeat")["error"]["message"] == "session is not owned by this transport"
        finally:
            controller(a, "detach")
        session["viewers"] = {b: object(), stranger: object()}
        assert server._detach_session_transport(session, b)
        assert not session["viewers"]
        assert not server._session_transport_contains(session, b)
        server._emit("message.complete", "fanout-invariant", {"text": "only A"})
        assert a.receive()["params"]["payload"]["text"] == "only A"
        assert b.frames.empty()
        assert server._attach_session_transport(session, b)
        server._emit("message.complete", "fanout-invariant", {"text": "reattached"})
        assert a.receive() == b.receive()
        b.close()
        # A stale queued envelope must not restore a dead peer's authority.
        assert not server._attach_session_transport(session, b)
        assert not server._session_transport_contains(session, b)
        assert not server._attach_session_transport(session, server._stdio_transport)
        assert server._close_sessions_for_transport(b) == (0, 0)
        assert server._close_sessions_for_transport(a) == (0, 1)
        assert session["transport"] is server._detached_ws_transport


@pytest.mark.linux_only
@pytest.mark.parametrize("client_type", [PipeClient, SocketClient])
@pytest.mark.parametrize("slow_first", [True, False])
@pytest.mark.parametrize("on_loop", [True, False])
def test_backpressure_never_blocks_later_frames_or_other_subscribers(slow_first, on_loop, client_type, monkeypatch):
    monkeypatch.setattr("tui_gateway.ws._WS_WRITE_TIMEOUT_S", 0.01)
    monkeypatch.setattr("tui_gateway.ws._TOKEN_COALESCE_S", 0)
    with ExitStack() as stack:
        healthy, slow = client_type(stack), client_type(stack, reading=False)
        # Fill the actual kernel pipe, not a fake wait in a transport.write().
        fd = slow.writer.fileno()
        os.set_blocking(fd, False)
        try:
            while True:
                os.write(fd, b"x" * 4096)
        except BlockingIOError:
            pass
        finally:
            os.set_blocking(fd, client_type is PipeClient)
        fan = FanoutTransport(*((slow, healthy) if slow_first else (healthy, slow)))
        returned = threading.Event()
        errors = []

        def emit():
            try:
                for kind in ("message.start", "message.delta", "message.complete"):
                    assert fan.write({"params": {"type": kind}})
            except BaseException as exc:
                errors.append(exc)
            finally:
                returned.set()

        async def loop_emit():
            emit()

        worker = threading.Thread(target=(lambda: asyncio.run(loop_emit())) if on_loop else emit, daemon=True)
        worker.start()
        try:
            assert returned.wait(3), "slow subscriber blocked the emitting turn"
            assert not errors
            assert [healthy.receive()["params"]["type"] for _ in range(3)] == [
                "message.start", "message.delta", "message.complete"]
            # Exhaust only the slow peer's bounded backlog; pace the healthy
            # reader by receipts so scheduler latency cannot make it overflow.
            for n in range(1024):
                frame = {"params": {"type": "message.delta", "n": n}}
                assert fan.write(frame)
                assert healthy.receive() == frame
                if not fan.contains(slow):
                    break
            assert not fan.contains(slow), "slow backlog grew without bound"
            for _ in range(16):
                assert fan.attach(slow)
                assert fan.write({"reattach": True})
                assert healthy.receive() == {"reattach": True}
                fan.detach(slow)
            assert slow.writes == 1, "reattach spawned more writers behind blocked I/O"
            assert fan.contains(healthy)
            assert fan.write({"params": {"type": "message.complete"}})
            assert healthy.receive()["params"]["type"] == "message.complete"
        finally:
            # Closing the real reader releases any blocked writer even on RED.
            slow.reader.close()
            worker.join(timeout=15)
            fan.close()
        assert not worker.is_alive()
        assert not fan.write({"after": "close"})


def test_overflow_signals_the_detached_subscriber():
    healthy = RecordingTransport()
    slow = RecordingTransport(delay=30.0)
    fan = FanoutTransport(healthy, slow)
    try:
        _overflow_slow_peer(fan, healthy, slow)
        assert not fan.contains(slow)
        assert fan.contains(healthy)
        assert slow.closed or any(_is_overflow_signal(frame) for frame in slow.frames)
    finally:
        slow.release()
        fan.close()


def test_fanout_close_does_not_close_peer_sockets():
    peer = RecordingTransport()
    fan = FanoutTransport(peer)
    assert fan.contains(peer)
    fan.close()
    assert peer.closed is False
    assert not fan.contains(peer)


def test_fanout_detach_does_not_close_peer_sockets():
    slow = RecordingTransport()
    fan = FanoutTransport(slow)
    assert fan.detach(slow)
    assert slow.closed is False
    assert not fan.contains(slow)


def test_healthy_peer_receives_frames_after_slow_overflow():
    healthy = RecordingTransport()
    slow = RecordingTransport(delay=30.0)
    fan = FanoutTransport(healthy, slow)
    try:
        last_n = _overflow_slow_peer(fan, healthy, slow)
        assert not fan.contains(slow)
        assert fan.contains(healthy)
        after = {"params": {"type": "message.complete", "n": last_n + 1}}
        assert fan.write(after)
        _await_frame_count(healthy, last_n + 2)
        assert healthy.frames[-1] == after
        assert fan.contains(healthy)
    finally:
        slow.release()
        fan.close()


def test_overflow_close_error_does_not_fail_emit_or_other_peers():
    class BoomTransport(RecordingTransport):
        def close(self):
            super().close()
            raise RuntimeError("overflow close exploded")

    healthy = RecordingTransport()
    slow = BoomTransport(delay=30.0)
    fan = FanoutTransport(healthy, slow)
    try:
        last_n = _overflow_slow_peer(fan, healthy, slow)
        assert not fan.contains(slow)
        assert fan.contains(healthy)
        after = {"params": {"type": "message.complete", "n": last_n + 1}}
        assert fan.write(after)
        _await_frame_count(healthy, last_n + 2)
        assert healthy.frames[-1] == after
    finally:
        slow.release()
        fan.close()
