import asyncio
import json
import threading
import time

from hermes_cli import mcp_startup
from tui_gateway import server
from tui_gateway import ws as ws_mod


def test_ws_startup_starts_background_mcp_discovery(monkeypatch):
    """The desktop app and dashboard chat reach the agent through this WS
    sidecar, not through tui_gateway.entry.main() (which spawns the discovery
    thread for the stdio TUI). handle_ws must start discovery itself, otherwise
    _make_agent's wait_for_mcp_discovery no-ops and the agent snapshots an
    MCP-less tool list. Regression test for #38945."""
    calls = []
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda **kw: calls.append(kw),
    )

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    server._sessions.clear()
    try:
        asyncio.run(ws_mod.handle_ws(FakeWS()))
    finally:
        server._sessions.clear()

    assert calls == [{"logger": ws_mod._log, "thread_name": "tui-ws-mcp-discovery"}]


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


def test_ws_write_loop_stall_does_not_latch_transport(monkeypatch):
    """A worker-thread write while the loop is stalled enqueues without latching
    the transport closed. The single writer flushes after the loop breathes."""
    monkeypatch.setattr(ws_mod, "_WS_WRITE_TIMEOUT_S", 0.05)
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
        loop.call_soon_threadsafe(time.sleep, 0.3)
        assert transport.write({"a": 1}) is True
        assert transport._closed is False

        assert transport.write({"b": 2}) is True
        deadline = time.time() + 2
        while len(sent) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(sent) == 2
        assert transport._closed is False
    finally:
        if transport is not None:
            transport.close()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def _event(event_type, sid="sid", payload=None):
    params = {"type": event_type, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    return {"jsonrpc": "2.0", "method": "event", "params": params}


async def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def _frame_types(lines):
    return [json.loads(line).get("params", {}).get("type") for line in lines]


def test_ws_transport_overload_bounds_noncritical_queue(monkeypatch):
    monkeypatch.setattr(ws_mod, "_WS_NONCRITICAL_MAX_FRAMES", 8)
    monkeypatch.setattr(ws_mod, "_TOKEN_COALESCE_S", 60.0)

    class FakeWS:
        async def send_text(self, line):
            raise AssertionError("streaming frames should remain queued during this test")

    async def scenario():
        transport = ws_mod.WSTransport(FakeWS(), asyncio.get_running_loop(), peer="bounded")
        try:
            for i in range(40):
                assert transport.write(_event("message.delta", payload={"text": str(i)})) is True
            with transport._lock:
                queued = len(transport._streaming) + len(transport._normal) + len(transport._snapshots)
                dropped = transport._dropped_noncritical
            assert queued <= ws_mod._WS_NONCRITICAL_MAX_FRAMES
            assert dropped > 0
        finally:
            transport.close()

    asyncio.run(scenario())


def test_ws_transport_coalesces_latest_only_snapshots(monkeypatch):
    monkeypatch.setattr(ws_mod, "_WS_SEND_BURST", 1000)
    sent = []

    class FakeWS:
        async def send_text(self, line):
            sent.append(line)

    async def scenario():
        transport = ws_mod.WSTransport(FakeWS(), asyncio.get_running_loop(), peer="snapshots")
        try:
            for i in range(25):
                assert transport.write(_event("status.update", payload={"kind": "run", "text": f"step {i}"})) is True
            assert await _wait_until(lambda: len(sent) == 1)
        finally:
            transport.close()

    asyncio.run(scenario())

    assert _frame_types(sent) == ["status.update"]
    assert json.loads(sent[0])["params"]["payload"]["text"] == "step 24"


def test_ws_transport_preserves_critical_during_noncritical_overload(monkeypatch):
    monkeypatch.setattr(ws_mod, "_WS_NONCRITICAL_MAX_FRAMES", 4)
    monkeypatch.setattr(ws_mod, "_TOKEN_COALESCE_S", 60.0)
    sent = []

    class FakeWS:
        async def send_text(self, line):
            sent.append(line)

    async def scenario():
        transport = ws_mod.WSTransport(FakeWS(), asyncio.get_running_loop(), peer="critical")
        try:
            for i in range(30):
                assert transport.write(_event("message.delta", payload={"text": str(i)})) is True
            assert transport.write(_event("approval.request", payload={"id": "approve-1"})) is True
            assert await _wait_until(lambda: "approval.request" in _frame_types(sent))
        finally:
            transport.close()

    asyncio.run(scenario())

    assert "approval.request" in _frame_types(sent)


def test_ws_transport_critical_overflow_closes_explicitly(monkeypatch):
    monkeypatch.setattr(ws_mod, "_WS_CRITICAL_MAX_FRAMES", 2)

    class FakeWS:
        async def send_text(self, line):
            raise AssertionError("critical overflow is checked before drain")

    async def scenario():
        transport = ws_mod.WSTransport(FakeWS(), asyncio.get_running_loop(), peer="critical-overflow")
        try:
            assert transport.write(_event("approval.request", payload={"id": 1})) is True
            assert transport.write(_event("approval.request", payload={"id": 2})) is True
            assert transport.write(_event("approval.request", payload={"id": 3})) is False
            assert transport._closed is True
        finally:
            transport.close()

    asyncio.run(scenario())


def test_ws_transport_close_cancels_writer_timer_and_clears_noncritical(monkeypatch):
    monkeypatch.setattr(ws_mod, "_TOKEN_COALESCE_S", 60.0)

    class FakeWS:
        async def send_text(self, line):
            raise AssertionError("close should cancel queued streaming send")

    async def scenario():
        transport = ws_mod.WSTransport(FakeWS(), asyncio.get_running_loop(), peer="close")
        assert transport.write(_event("message.delta", payload={"text": "queued"})) is True
        await asyncio.sleep(0)
        transport.close()
        await asyncio.sleep(0)
        with transport._lock:
            assert not transport._streaming
            assert not transport._normal
            assert not transport._snapshots
        assert transport._stream_flush_handle is None
        assert transport._writer_task is None or transport._writer_task.done() or transport._writer_task.cancelled()

    asyncio.run(scenario())
