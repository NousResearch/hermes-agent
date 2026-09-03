import asyncio
import concurrent.futures
import json
import threading
import time

from hermes_cli import mcp_startup
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




def test_ws_connection_registers_then_disconnect_unregisters_live_transport(monkeypatch):
    """A connected client must be tracked in the live-transport registry so a
    session-less global broadcast (skin.changed from the background watcher)
    reaches it, and dropped on disconnect so no stale write targets a dead peer.
    This is the WS half of the cross-surface live-theme fix."""
    server._sessions.clear()
    server._live_transports.clear()
    seen = {}
    try:
        _run_disconnect(
            monkeypatch,
            lambda t: seen.__setitem__("registered", t in server._live_transports),
        )
        # Seeded at receive_text time — i.e. after gateway.ready registered it.
        assert seen["registered"] is True
        # handle_ws's finally must have unregistered it.
        assert not server._live_transports
    finally:
        server._sessions.clear()
        server._live_transports.clear()


def test_ws_disconnect_releases_wake_word_owner(monkeypatch):
    released = []
    created = []
    monkeypatch.setattr(
        server,
        "_release_wake_for_transport",
        lambda transport: released.append(transport) or True,
    )

    _run_disconnect(monkeypatch, lambda transport: created.append(transport))

    assert released == created




def test_ws_starts_mcp_discovery_before_ready(monkeypatch):
    import tui_gateway.entry as entry

    calls = []
    events = []

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: calls.append("mcp"))

    class FakeWS:
        async def accept(self):
            events.append("accept")

        async def send_text(self, line):
            if '"gateway.ready"' in line:
                events.append(f"ready_after_{len(calls)}")

        async def receive_text(self):
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))

    # Discovery moved to profile-aware agent construction. WebSocket transport
    # should not start MCP discovery before a profile has been bound.
    assert calls == []
    assert events == ["accept", "ready_after_0"]


def test_ws_ready_advertises_heartbeat_and_ping_is_inline(monkeypatch):
    sent = []
    inbound = iter(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "heartbeat-1",
                    "method": "gateway.ping",
                    "params": {},
                }
            )
        ]
    )
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            sent.append(json.loads(line))

        async def receive_text(self):
            try:
                return next(inbound)
            except StopIteration:
                raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))

    ready = sent[0]["params"]
    assert ready["type"] == "gateway.ready"
    assert ready["payload"]["heartbeat"] is True
    assert sent[1] == {
        "jsonrpc": "2.0",
        "result": {"ok": True},
        "id": "heartbeat-1",
    }


def test_ws_transport_serializes_concurrent_sends():
    active_sends = 0
    max_active_sends = 0
    sent = []

    class FakeWS:
        async def send_text(self, line):
            nonlocal active_sends, max_active_sends
            active_sends += 1
            max_active_sends = max(max_active_sends, active_sends)
            try:
                await asyncio.sleep(0.05)
                sent.append(line)
            finally:
                active_sends -= 1

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        transport = ws_mod.WSTransport(FakeWS(), loop, peer="serialize-test")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(transport.write, {"idx": 1}),
                pool.submit(transport.write, {"idx": 2}),
            ]
            assert [f.result(timeout=2) for f in futures] == [True, True]

        assert len(sent) == 2
        assert max_active_sends == 1
        assert transport._closed is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_ws_transport_preserves_cross_batch_order():
    async def scenario():
        entered = []
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()

        class FakeWS:
            async def send_text(self, line):
                entered.append(line)
                if line == "A1":
                    first_entered.set()
                    await release_first.wait()

        transport = ws_mod.WSTransport(
            FakeWS(), asyncio.get_running_loop(), peer="batch-order-test"
        )
        first = asyncio.create_task(transport._safe_send_many(["A1", "A2"]))
        await first_entered.wait()

        async def send_second():
            second_started.set()
            await transport._safe_send_many(["B1", "B2"])

        second = asyncio.create_task(send_second())
        await second_started.wait()

        # The second task has reached the transport. Without whole-batch
        # serialization it runs B1/B2 before this task can resume.
        assert entered == ["A1"]

        release_first.set()
        await asyncio.gather(first, second)
        assert entered == ["A1", "A2", "B1", "B2"]

    asyncio.run(scenario())


def test_ws_transport_anchors_coalesced_token_flush():
    """The coalesce timer empties _pending_tokens before it creates the send
    task, so that task holds the only surviving reference to the batch. A bare
    asyncio.create_task() is only weakly referenced by the loop, so the
    transport must keep a strong reference of its own — otherwise a pending
    flush can be collected mid-flight and those streamed tokens never reach the
    client, with nothing left to retry or re-queue them."""

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        sent = []

        class FakeWS:
            async def send_text(self, line):
                entered.set()
                await release.wait()
                sent.append(line)

        transport = ws_mod.WSTransport(
            FakeWS(), asyncio.get_running_loop(), peer="anchor-test"
        )
        with transport._token_lock:
            transport._pending_tokens.extend(["T1", "T2"])
            transport._token_flush_armed = True

        transport._flush_tokens()

        # The batch is no longer reachable through the buffer.
        assert transport._pending_tokens == []

        await entered.wait()
        in_flight = [t for t in transport._background_tasks if not t.done()]
        assert len(in_flight) == 1

        release.set()
        await asyncio.wait_for(asyncio.gather(*in_flight), timeout=5)
        assert sent == ["T1", "T2"]
        # The done callback releases the reference, so the set cannot grow
        # across a streamed turn's hundreds of flushes.
        assert not transport._background_tasks

    asyncio.run(scenario())


def test_ws_transport_close_cancels_in_flight_batch_send():
    """close() latches _closed, but _safe_send_many only re-checks that between
    frames: a send already suspended inside ws.send_text() on a wedged socket
    never observes it. Now that the transport anchors that task, teardown has to
    cancel it, or the task and the transport keep each other alive after
    handle_ws has returned."""

    async def scenario():
        entered = asyncio.Event()
        wedged = asyncio.Event()  # never set: the socket never completes a send

        class FakeWS:
            async def send_text(self, line):
                entered.set()
                await wedged.wait()

        transport = ws_mod.WSTransport(
            FakeWS(), asyncio.get_running_loop(), peer="close-cancel-test"
        )
        with transport._token_lock:
            transport._pending_tokens.append("T1")
            transport._token_flush_armed = True

        transport._flush_tokens()

        await entered.wait()
        in_flight = [t for t in transport._background_tasks if not t.done()]
        assert len(in_flight) == 1
        task = in_flight[0]

        transport.close()

        # Tracking is dropped synchronously rather than one done callback at a
        # time, so close() leaves no transport -> set -> task -> transport cycle
        # behind for the loop to unpick later.
        assert not transport._background_tasks

        _done, pending = await asyncio.wait({task}, timeout=5)
        assert not pending, "close() left an in-flight batch send pending"
        assert task.cancelled()

    asyncio.run(scenario())


def test_ws_transport_close_drops_the_coalesce_buffer():
    """close() cancels the coalesce TimerHandle, which is the only caller that
    would ever have drained _pending_tokens. Anything still buffered there is
    undeliverable from that moment on, so close() must release it instead of
    pinning the frames on a dead transport."""

    async def scenario():
        class FakeWS:
            async def send_text(self, line):
                raise AssertionError("a closed transport must not send")

        transport = ws_mod.WSTransport(
            FakeWS(), asyncio.get_running_loop(), peer="close-buffer-test"
        )
        with transport._token_lock:
            transport._pending_tokens.extend(["T1", "T2"])
            transport._token_flush_armed = True

        transport.close()

        assert transport._pending_tokens == []

    asyncio.run(scenario())


