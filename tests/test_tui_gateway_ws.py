import asyncio
import concurrent.futures
import json
import threading
import time
from types import SimpleNamespace

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


def test_ws_ready_does_not_wait_for_skin_resolution(monkeypatch):
    """A slow/default-executor skin lookup must not delay the liveness frame.

    The Desktop starts its ready timeout as soon as the socket opens.  Waiting
    for ``resolve_skin`` here turns unrelated worker saturation into a false
    "Runtime not ready" failure and every reconnect adds another queued lookup.
    """
    skin_started = threading.Event()
    release_skin = threading.Event()

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    # These attributes are introduced by the fix.  ``raising=False`` keeps this
    # test runnable against the pre-fix implementation for the RED assertion.
    monkeypatch.setattr(server, "get_cached_skin_payload", lambda: {}, raising=False)
    monkeypatch.setattr(ws_mod, "_skin_refresh_task", None, raising=False)
    monkeypatch.setattr(ws_mod, "_skin_refresh_loop", None, raising=False)
    monkeypatch.setattr(server, "_note_cached_skin_broadcast", lambda _revision: True)
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *_args: None)

    def blocking_resolve_skin_snapshot():
        skin_started.set()
        assert release_skin.wait(timeout=2)
        return {"name": "late-skin"}, 1

    monkeypatch.setattr(
        server, "resolve_skin_snapshot", blocking_resolve_skin_snapshot, raising=False
    )

    async def scenario():
        ready_seen = asyncio.Event()
        disconnect = asyncio.Event()

        class FakeWS:
            async def accept(self):
                pass

            async def send_text(self, line):
                frame = json.loads(line)
                if frame.get("params", {}).get("type") == "gateway.ready":
                    ready_seen.set()

            async def receive_text(self):
                await disconnect.wait()
                raise ws_mod._WebSocketDisconnect()

            async def close(self):
                pass

        task = asyncio.create_task(ws_mod.handle_ws(FakeWS()))
        try:
            # The ready frame must beat the blocked skin lookup.  This times out
            # on the old implementation, which awaited to_thread(resolve_skin).
            await asyncio.wait_for(ready_seen.wait(), timeout=2)
            assert not release_skin.is_set()
            assert await asyncio.to_thread(skin_started.wait, 2)
        finally:
            release_skin.set()
            disconnect.set()
            await asyncio.wait_for(task, timeout=2)
            refresh = getattr(ws_mod, "_skin_refresh_task", None)
            if refresh is not None:
                await asyncio.wait_for(asyncio.shield(refresh), timeout=2)

    asyncio.run(scenario())


def test_concurrent_cold_ws_connections_share_one_skin_refresh(monkeypatch):
    """Reconnect bursts must collapse cache misses into one background lookup."""
    calls = 0
    calls_lock = threading.Lock()
    skin_started = threading.Event()
    release_skin = threading.Event()

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "get_cached_skin_payload", lambda: {}, raising=False)
    monkeypatch.setattr(ws_mod, "_skin_refresh_task", None, raising=False)
    monkeypatch.setattr(ws_mod, "_skin_refresh_loop", None, raising=False)
    broadcasts = []
    monkeypatch.setattr(server, "_note_cached_skin_broadcast", lambda _revision: True)
    monkeypatch.setattr(
        server,
        "_broadcast_global_event",
        lambda event, payload: broadcasts.append((event, payload)),
    )

    def blocking_resolve_skin_snapshot():
        nonlocal calls
        with calls_lock:
            calls += 1
        skin_started.set()
        assert release_skin.wait(timeout=2)
        return {"name": "shared-skin"}, 1

    monkeypatch.setattr(
        server, "resolve_skin_snapshot", blocking_resolve_skin_snapshot, raising=False
    )

    async def scenario():
        ready_events = [asyncio.Event(), asyncio.Event()]
        disconnect = asyncio.Event()

        class FakeWS:
            def __init__(self, ready):
                self.ready = ready

            async def accept(self):
                pass

            async def send_text(self, line):
                frame = json.loads(line)
                if frame.get("params", {}).get("type") == "gateway.ready":
                    self.ready.set()

            async def receive_text(self):
                await disconnect.wait()
                raise ws_mod._WebSocketDisconnect()

            async def close(self):
                pass

        tasks = [
            asyncio.create_task(ws_mod.handle_ws(FakeWS(ready)))
            for ready in ready_events
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*(ready.wait() for ready in ready_events)),
                timeout=2,
            )
            assert await asyncio.to_thread(skin_started.wait, 2)
            assert calls == 1
        finally:
            release_skin.set()
            disconnect.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
            refresh = getattr(ws_mod, "_skin_refresh_task", None)
            if refresh is not None:
                await asyncio.wait_for(asyncio.shield(refresh), timeout=2)

    asyncio.run(scenario())
    assert broadcasts == [("skin.changed", {"name": "shared-skin"})]


def test_resolve_skin_snapshot_replaces_cache_and_preserves_last_good_on_failure(
    monkeypatch,
):
    """Only a complete skin resolution may replace the process snapshot."""
    import hermes_cli.skin_engine as skin_engine

    skin = SimpleNamespace(
        name="fresh",
        colors={"accent": "#123456"},
        light_colors={},
        dark_colors={},
        branding={"help_header": "Fresh"},
        banner_logo="logo",
        banner_hero="hero",
        tool_prefix="tool",
    )
    monkeypatch.setattr(server, "_skin_payload_cache", {"name": "old"})
    monkeypatch.setattr(server, "_skin_payload_sig", ("old", 1.0))
    monkeypatch.setattr(server, "_skin_payload_revision", 7)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"skin": "fresh"}})
    monkeypatch.setattr(server, "_skin_sig_from_config", lambda _cfg: ("fresh", 2.0))
    monkeypatch.setattr(skin_engine, "init_skin_from_config", lambda _cfg: None)
    monkeypatch.setattr(skin_engine, "get_active_skin", lambda: skin)

    payload, revision = server.resolve_skin_snapshot()

    assert payload["name"] == "fresh"
    assert payload["help_header"] == "Fresh"
    assert revision == 8
    assert server.get_cached_skin_payload() == payload
    assert server._skin_payload_sig == ("fresh", 2.0)
    monkeypatch.setattr(server, "_last_skin_sig", ("old", 1.0))
    assert server._note_cached_skin_broadcast(7) is False
    assert server._last_skin_sig == ("old", 1.0)
    assert server._note_cached_skin_broadcast(8) is True
    assert server._last_skin_sig == ("fresh", 2.0)

    def fail_resolution():
        raise RuntimeError("resolution failed")

    monkeypatch.setattr(skin_engine, "get_active_skin", fail_resolution)
    failed, failed_revision = server.resolve_skin_snapshot()

    assert (failed, failed_revision) == ({}, -1)
    assert server.get_cached_skin_payload() == payload
    assert server._skin_payload_sig == ("fresh", 2.0)
    assert server._skin_payload_revision == 8


def test_skin_watcher_detects_config_changed_after_startup_prime(monkeypatch):
    """Watcher baseline must describe the cache, not a newer unread config."""
    broadcasts = []

    class NoopThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(server, "_skin_payload_cache", {"name": "old"})
    monkeypatch.setattr(server, "_skin_payload_sig", ("old", 1.0))
    monkeypatch.setattr(server, "_skin_payload_revision", 1)
    monkeypatch.setattr(server, "_last_skin_sig", None)
    monkeypatch.setattr(server, "_skin_watcher_started", False)
    monkeypatch.setattr(server.threading, "Thread", NoopThread)
    monkeypatch.setattr(server, "_skin_sig", lambda: ("new", 2.0))
    monkeypatch.setattr(server, "resolve_skin", lambda: {"name": "new"})
    monkeypatch.setattr(
        server,
        "_broadcast_global_event",
        lambda event, payload: broadcasts.append((event, payload)),
    )

    server._ensure_skin_watcher()
    assert server._last_skin_sig == ("old", 1.0)

    server._broadcast_skin_if_changed()
    assert broadcasts == [("skin.changed", {"name": "new"})]
    assert server._last_skin_sig == ("new", 2.0)


def test_skin_refresh_task_is_recreated_for_a_new_event_loop(monkeypatch):
    """A process-global task must never be reused from a different loop."""
    monkeypatch.setattr(server, "resolve_skin_snapshot", lambda: ({"name": "x"}, 1))
    monkeypatch.setattr(server, "_note_cached_skin_broadcast", lambda _revision: True)
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *_args: None)
    first_loop = asyncio.new_event_loop()

    async def never_finishes():
        await asyncio.Future()

    first_task = first_loop.create_task(never_finishes())
    monkeypatch.setattr(ws_mod, "_skin_refresh_task", first_task)
    monkeypatch.setattr(ws_mod, "_skin_refresh_loop", first_loop)
    try:
        async def make_second_task():
            task = ws_mod._ensure_skin_cache_refresh()
            await task
            return task

        second_task = asyncio.run(make_second_task())

        assert second_task is not first_task
        assert second_task.get_loop() is not first_task.get_loop()
        assert first_task.done() is False
    finally:
        first_task.cancel()
        first_loop.run_until_complete(
            asyncio.gather(first_task, return_exceptions=True)
        )
        first_loop.close()


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


