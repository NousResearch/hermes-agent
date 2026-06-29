import asyncio
import threading
import time

from tui_gateway import server
from tui_gateway import ws as ws_mod


def _run_disconnect(monkeypatch, seed):
    """Drive handle_ws to its disconnect `finally`, seeding sessions against the
    live WSTransport the moment it exists. Returns nothing; inspect _sessions."""
    # Disable the grace-reap Timer: detached sessions normally schedule a
    # threading.Timer via _schedule_ws_orphan_reap, which would outlive the test
    # and fire _reap during interpreter teardown — touching _sessions/DB and
    # producing spurious post-run errors under the per-file CI runner. Grace=0
    # short-circuits the Timer (see _schedule_ws_orphan_reap) so the test leaves
    # no lingering thread.
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)

    # Mirror the real _finalize_session chokepoint: it is the single place that
    # closes the slash-worker (#38095). Stub it but keep that behavior so the
    # disconnect-reap path still exercises worker teardown.
    def _fake_finalize(s, end_reason="tui_close"):
        w = s.get("slash_worker")
        if w:
            w.close()

    monkeypatch.setattr(server, "_finalize_session", _fake_finalize)

    created = []
    real_transport = ws_mod.WSTransport
    monkeypatch.setattr(
        ws_mod, "WSTransport",
        lambda ws, loop, **kw: created.append(real_transport(ws, loop, **kw)) or created[-1],
    )

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            seed(created[0])  # transport now exists; attach it to sessions
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))


def test_ws_disconnect_reaps_flagged_session_and_closes_worker(monkeypatch):
    closed = []

    class FakeWorker:
        def close(self):
            closed.append(True)

    server._sessions.clear()
    try:
        _run_disconnect(
            monkeypatch,
            lambda t: server._sessions.update(
                flagged={
                    "transport": t,
                    "close_on_disconnect": True,
                    "slash_worker": FakeWorker(),
                    "session_key": "k",
                }
            ),
        )
        assert "flagged" not in server._sessions
        assert closed == [True]
    finally:
        server._sessions.clear()


def test_ws_disconnect_preserves_and_repoints_reconnectable_session(monkeypatch):
    server._sessions.clear()
    try:
        _run_disconnect(
            monkeypatch,
            lambda t: server._sessions.update(
                plain={"transport": t, "close_on_disconnect": False, "session_key": "k"}
            ),
        )
        assert server._sessions["plain"]["transport"] is server._detached_ws_transport
    finally:
        server._sessions.clear()


def _teardown(transport, loop, thread):
    """Close transport, cancel writer, stop loop, join thread."""
    if transport is not None:
        transport.close()
        w = transport._writer
        if w is not None and not w.done():
            loop.call_soon_threadsafe(w.cancel)
    time.sleep(0.05)  # let the loop process the cancel / sentinel
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def test_ws_write_does_not_block_on_slow_event_loop():
    """Writes must never block the caller, even when the event loop is
    stalled.  The queue-based transport enqueues instantly and the writer
    task drains frames when the loop recovers — no timeout, no latch.

    This replaces the old _WS_WRITE_TIMEOUT_S test: the property it
    verified (slow loop doesn't permanently break the transport) is now
    inherent in the design because write() never touches the event loop.
    """
    sent = []

    class FakeWS:
        async def send_text(self, line):
            sent.append(line)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    transport = None
    try:
        transport = ws_mod.WSTransport(FakeWS(), loop, peer="stall-test")
        # Start the writer task on the loop.
        loop.call_soon_threadsafe(transport.start_writer)

        # Stall the event loop for 0.3s — during this time, writes from
        # this (non-loop) thread must return immediately.
        loop.call_soon_threadsafe(time.sleep, 0.3)

        # These writes return True instantly despite the stalled loop.
        assert transport.write({"a": 1}) is True
        assert transport.write({"b": 2}) is True
        assert transport._closed is False

        # Once the loop breathes, the writer drains the queue and both
        # frames reach the socket in order.
        deadline = time.time() + 2
        while len(sent) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(sent) == 2
        assert transport._closed is False
    finally:
        _teardown(transport, loop, thread)


def test_ws_write_returns_false_when_closed():
    """After close(), write() returns False — the 'peer is gone' signal."""
    sent = []

    class FakeWS:
        async def send_text(self, line):
            sent.append(line)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    transport = None
    try:
        transport = ws_mod.WSTransport(FakeWS(), loop, peer="close-test")
        loop.call_soon_threadsafe(transport.start_writer)

        # Before close — writes succeed.
        assert transport.write({"x": 1}) is True

        # Let the writer drain.
        deadline = time.time() + 2
        while len(sent) < 1 and time.time() < deadline:
            time.sleep(0.01)

        # After close — writes return False.
        transport.close()
        assert transport.write({"y": 2}) is False
    finally:
        _teardown(transport, loop, thread)


def test_ws_writer_serializes_sends():
    """The writer task calls send_text one frame at a time — no concurrent
    sends on the same socket.  Verify ordering is preserved under
    concurrent writes from multiple threads."""
    sent = []
    send_lock = threading.Lock()

    class FakeWS:
        async def send_text(self, line):
            # Simulate a slow socket: each send takes a few ms.
            await asyncio.sleep(0.005)
            with send_lock:
                sent.append(line)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    transport = None
    try:
        transport = ws_mod.WSTransport(FakeWS(), loop, peer="order-test")
        loop.call_soon_threadsafe(transport.start_writer)

        # Hammer writes from multiple threads.
        def writer_thread(n):
            for i in range(n):
                transport.write({"thread": threading.current_thread().name, "i": i})

        threads = [threading.Thread(target=writer_thread, args=(5,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for all frames to be sent.
        deadline = time.time() + 5
        while len(sent) < 20 and time.time() < deadline:
            time.sleep(0.01)

        assert len(sent) == 20
        # Each frame should be valid JSON (no interleaving corruption).
        import json
        for line in sent:
            obj = json.loads(line)
            assert "thread" in obj and "i" in obj
    finally:
        _teardown(transport, loop, thread)
