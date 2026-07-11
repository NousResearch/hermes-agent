"""Tests for the tui-gateway observer hooks.

Covers the three observer hooks that let plugins watch gateway traffic
without patching core:

  - ``post_emit_event``        — after every emitted gateway event
  - ``post_frame_write``       — after a session-owned event frame is routed
  - ``on_ws_transport_change`` — WS client attach/detach

Behavior contracts:
  1. All three are members of ``VALID_HOOKS`` (a plugin registering them
     gets no unknown-hook warning).
  2. The gateway call sites invoke them with the documented kwargs.
  3. A raising callback never breaks the emit/write path (isolation).
  4. With no callbacks registered the paths behave exactly as before
     (strict no-op).
"""

from unittest import mock

import pytest

from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture()
def hook_manager():
    """Register test callbacks against the real manager, restore after."""
    mgr = get_plugin_manager()
    before = {k: list(v) for k, v in mgr._hooks.items()}
    yield mgr
    mgr._hooks.clear()
    mgr._hooks.update(before)


def test_gateway_observer_hooks_are_valid_hooks():
    for hook in ("post_emit_event", "post_frame_write", "on_ws_transport_change"):
        assert hook in VALID_HOOKS, f"{hook} missing from VALID_HOOKS"


def test_post_emit_event_receives_documented_kwargs(hook_manager):
    from tui_gateway import server

    seen = []
    hook_manager._hooks.setdefault("post_emit_event", []).append(
        lambda event=None, session_id=None, payload=None, **kw: seen.append(
            (event, session_id, payload)
        )
    )
    server._emit("turn.complete", "sess-1", {"ok": True})
    assert seen == [("turn.complete", "sess-1", {"ok": True})]


def test_post_emit_event_raising_callback_is_isolated(hook_manager):
    from tui_gateway import server

    def _boom(**kw):
        raise RuntimeError("plugin bug")

    hook_manager._hooks.setdefault("post_emit_event", []).append(_boom)
    # Must not raise — a broken plugin can't break the gateway emit path.
    server._emit("turn.complete", "sess-1", None)


def test_post_frame_write_fires_only_for_session_owned_event_frames(hook_manager):
    from tui_gateway import server

    seen = []
    hook_manager._hooks.setdefault("post_frame_write", []).append(
        lambda frame=None, session_id=None, owner_transport=None, **kw: seen.append(
            ((frame or {}).get("method"), session_id, owner_transport)
        )
    )

    write_order = []

    class _Transport:
        def write(self, obj):
            write_order.append("write")
            return True

    t = _Transport()
    with mock.patch.dict(server._sessions, {"sid-9": {"transport": t}}, clear=False):
        # Session-owned event frame -> hook fires with the owner transport.
        server.write_json({"method": "event", "params": {"session_id": "sid-9"}})
        # Event frame for an unknown session -> no owner -> no hook.
        server.write_json({"method": "event", "params": {"session_id": "nope"}})
        # Non-event frame -> no hook.
        server.write_json({"method": "result", "params": {"session_id": "sid-9"}})

    assert seen == [("event", "sid-9", t)]
    # "post" contract: the owner write happens BEFORE the observer fires.
    assert write_order == ["write"]


def test_post_frame_write_observes_after_write_and_preserves_result(hook_manager):
    from tui_gateway import server

    order = []
    hook_manager._hooks.setdefault("post_frame_write", []).append(
        lambda **kw: order.append("hook")
    )

    class _FailingTransport:
        def write(self, obj):
            order.append("write")
            return False  # send failure must still reach the caller

    t = _FailingTransport()
    with mock.patch.dict(server._sessions, {"sid-f": {"transport": t}}, clear=False):
        result = server.write_json({"method": "event", "params": {"session_id": "sid-f"}})

    assert order == ["write", "hook"]
    assert result is False  # hook must not mask the transport's return value


def test_on_ws_transport_change_fired_and_isolated(hook_manager):
    from tui_gateway import server

    seen = []
    hook_manager._hooks.setdefault("on_ws_transport_change", []).append(
        lambda action=None, transport=None, **kw: seen.append(action)
    )
    hook_manager._hooks.setdefault("on_ws_transport_change", []).append(
        lambda **kw: (_ for _ in ()).throw(RuntimeError("bad plugin"))
    )
    fake_transport = object()
    # Fired via the same helper the WS handler uses; must not raise.
    server._notify_gateway_observers(
        "on_ws_transport_change", action="connect", transport=fake_transport
    )
    server._notify_gateway_observers(
        "on_ws_transport_change", action="disconnect", transport=fake_transport
    )
    assert seen == ["connect", "disconnect"]


def test_unregistered_hooks_are_noops():
    """With nothing registered the gateway paths behave exactly as before."""
    from tui_gateway import server

    class _Transport:
        def __init__(self):
            self.frames = []

        def write(self, obj):
            self.frames.append(obj)
            return True

    t = _Transport()
    with mock.patch.dict(server._sessions, {"sid-1": {"transport": t}}, clear=False):
        assert server.write_json({"method": "event", "params": {"session_id": "sid-1"}})
    assert len(t.frames) == 1


def _drain_observer_pool():
    """Wait for the async observer pool to flush (tests only)."""
    from tui_gateway import server

    if server._OBSERVER_POOL is not None:
        server._OBSERVER_POOL.submit(lambda: None).result(timeout=5)


def test_handle_ws_fires_connect_and_disconnect(hook_manager, monkeypatch):
    """FakeWS through the REAL handle_ws(): both lifecycle notifications
    must be delivered (connect on accept, disconnect in the finally block)."""
    import asyncio

    from hermes_cli import mcp_startup
    from tui_gateway import server
    from tui_gateway import ws as ws_mod

    monkeypatch.setattr(
        mcp_startup, "start_background_mcp_discovery", lambda **kw: None
    )
    seen = []
    hook_manager._hooks.setdefault("on_ws_transport_change", []).append(
        lambda **kw: seen.append(kw["action"])
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
    _drain_observer_pool()
    assert seen == ["connect", "disconnect"]


def test_slow_ws_observer_does_not_block_handle_ws(hook_manager, monkeypatch):
    """Regression: a slow/blocking callback must not stall the WS event
    loop — handle_ws() completes immediately while the callback runs on
    the observer pool."""
    import asyncio
    import threading
    import time

    from hermes_cli import mcp_startup
    from tui_gateway import server
    from tui_gateway import ws as ws_mod

    monkeypatch.setattr(
        mcp_startup, "start_background_mcp_discovery", lambda **kw: None
    )
    release = threading.Event()
    calls = []

    def slow_callback(**kw):
        calls.append(kw["action"])
        release.wait(timeout=10)

    hook_manager._hooks.setdefault("on_ws_transport_change", []).append(
        slow_callback
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
    t0 = time.monotonic()
    try:
        asyncio.run(ws_mod.handle_ws(FakeWS()))
    finally:
        server._sessions.clear()
    elapsed = time.monotonic() - t0
    release.set()
    _drain_observer_pool()
    # handle_ws returned without waiting on the blocked callback.
    assert elapsed < 5, f"handle_ws blocked for {elapsed:.1f}s on a slow observer"
    assert calls and calls[0] == "connect"
