import asyncio
import concurrent.futures
import datetime
import json
import threading

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


def _slow_dispatch_harness(monkeypatch):
    """handle_ws over a scripted FakeWS whose ``slow`` RPC blocks inside dispatch() until released.

    Returns (inbound queue, sent frames, event log, release Event). Push ``None`` to disconnect."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    sent, log, release = [], [], threading.Event()

    def fake_dispatch(req, transport):
        method = req.get("method")
        log.append(f"dispatch:{method}")
        if method == "slow":
            release.wait(5)
        log.append(f"done:{method}")
        return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"method": method}}

    def fake_close_sessions(transport, end_reason):
        log.append("teardown")
        return 0, 0

    monkeypatch.setattr(server, "dispatch", fake_dispatch)
    monkeypatch.setattr(server, "_close_sessions_for_transport", fake_close_sessions)
    inbound: asyncio.Queue = asyncio.Queue()

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            sent.append(json.loads(line))

        async def receive_text(self):
            frame = await inbound.get()
            if frame is None:
                raise ws_mod._WebSocketDisconnect()
            return json.dumps(frame)

        async def close(self):
            pass

    return FakeWS(), inbound, sent, log, release


async def _wait_for(predicate, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _rpc(req_id, method):
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": {}}


def test_ws_ping_is_answered_while_an_earlier_rpc_blocks_dispatch(monkeypatch):
    """#108325: a handler blocked for minutes (lock wait behind a long compaction, GIL-heavy turn) must not
    starve gateway.ping — the client's 45s heartbeat deadline would otherwise tear down a busy but healthy
    backend. Non-ping RPCs keep their serial arrival order."""
    ws, inbound, sent, log, release = _slow_dispatch_harness(monkeypatch)
    ids = lambda: [f.get("id") for f in sent if "id" in f]

    async def scenario():
        task = asyncio.create_task(ws_mod.handle_ws(ws))
        try:
            for frame in (_rpc("r1", "slow"), _rpc("r2", "fast"), _rpc("heartbeat-1", "gateway.ping")):
                inbound.put_nowait(frame)
            ping_answered = await _wait_for(lambda: "heartbeat-1" in ids(), timeout=1.0)
            assert ping_answered, f"gateway.ping went unanswered while dispatch was busy; sent ids={ids()}"
            assert "done:slow" not in log and "dispatch:fast" not in log
        finally:
            release.set()
        assert await _wait_for(lambda: {"r1", "r2"} <= set(ids()))
        inbound.put_nowait(None)
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert log.index("done:slow") < log.index("dispatch:fast")
    assert ids().index("r1") < ids().index("r2")


def test_ws_disconnect_teardown_waits_for_in_flight_dispatch(monkeypatch):
    """A client that drops while a handler runs must not have its sessions torn down under that handler, and
    frames it sent before dropping are still dispatched (as the serial read loop did)."""
    ws, inbound, sent, log, release = _slow_dispatch_harness(monkeypatch)

    async def scenario():
        task = asyncio.create_task(ws_mod.handle_ws(ws))
        inbound.put_nowait(_rpc("r1", "slow"))
        assert await _wait_for(lambda: "dispatch:slow" in log)
        inbound.put_nowait(_rpc("r2", "queued"))
        inbound.put_nowait(None)
        await asyncio.sleep(0.1)
        assert "teardown" not in log
        release.set()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert log == ["dispatch:slow", "done:slow", "dispatch:queued", "done:queued", "teardown"]


def test_ws_failed_reply_from_dispatcher_ends_the_connection(monkeypatch):
    """A response the dispatcher cannot send ends the connection even while the reader waits on the socket."""
    ws, inbound, sent, log, release = _slow_dispatch_harness(monkeypatch)
    real_send = type(ws).send_text

    async def send_text(self, line):
        if json.loads(line).get("id") == "r1":
            raise RuntimeError("peer gone")
        await real_send(self, line)

    monkeypatch.setattr(type(ws), "send_text", send_text)

    async def scenario():
        inbound.put_nowait(_rpc("r1", "fast"))
        await asyncio.wait_for(ws_mod.handle_ws(ws), 5)

    asyncio.run(scenario())
    assert log == ["dispatch:fast", "done:fast", "teardown"]


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


def test_ws_transport_replies_with_error_for_unserializable_response(caplog):
    """#92506: an unserializable payload (datetime from profile.yaml ui_meta) must surface as a
    JSON-RPC error frame with the original id plus a log line — on both the worker-thread
    ``write`` and the loop-side ``write_async`` twin — and leave the transport open, instead of
    killing the pool worker silently so the client waits forever."""
    sent = []

    class FakeWS:
        async def send_text(self, line):
            sent.append(json.loads(line))

    bad = {"jsonrpc": "2.0", "id": "profiles", "result": {"created": datetime.datetime(2026, 8, 22)}}

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        transport = ws_mod.WSTransport(FakeWS(), loop, peer="serialization-test")
        assert transport.write(bad) is True
        assert transport.write({"jsonrpc": "2.0", "id": "next", "result": {}}) is True
        assert asyncio.run_coroutine_threadsafe(transport.write_async(bad), loop).result(5) is True
        assert transport.closed is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    assert [m.get("id") for m in sent] == ["profiles", "next", "profiles"]
    for frame in (sent[0], sent[2]):
        assert frame["error"]["code"] == -32603
        assert frame["error"]["message"].startswith("response serialization error")
        assert "datetime" in frame["error"]["message"]
    assert sent[1] == {"jsonrpc": "2.0", "id": "next", "result": {}}
    assert caplog.text.count("frame serialization failed") == 2


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


