"""Regression tests for dead stdio subprocess recovery (#95626 salvage).

The #81995 fast-fail gate detects a dead stdio subprocess but the transport
failure never cleared ``server.session``, so the transport-down reconnect path
(which only fires when the session is gone/not-ready) never ran. The call
failed fast — correctly — but nothing asked the server task to respawn the
subprocess (#95626 added the reconnect signal).

Signalling alone still lost the call: a gateway restart kills every MCP stdio
child, and the first call from a surviving agent session (or a cron run
spanning the restart) failed in 0.00s while the subprocess was respawned
seconds later. Both fast-fail sites now respawn AND retry once:

- pre-call gate (children already dead when the call arrives);
- mid-call watcher race (children die while the RPC is in flight).

Both must recover transparently, and both must stop after ONE retry so a
server that keeps dying parks via run()'s rapid-drop budget instead of
hot-cycling respawns forever. The error text must never claim a timeout —
that wording is what misdirected the original investigation.
"""

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")
from tools import mcp_tool_loop as _mcp_loop  # noqa: E402


def _success_result():
    result = MagicMock()
    result.is_error = False
    block = MagicMock()
    block.text = "ok"
    result.content = [block]
    result.structured_content = None
    result.meta = None
    return result


def _install_stub_server(mcp_tool_module, name: str, call_tool_impl,
                         *, children_dead, on_reconnect=None):
    """Fake MCP server with real-bool stdio liveness and a countable
    reconnect event (mirrors tests/tools/test_mcp_circuit_breaker.py).

    ``on_reconnect`` runs on the MCP loop thread when the reconnect event is
    set — the hook tests use to simulate the server task respawning the
    subprocess and publishing a fresh session.
    """
    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl
    server.session = session

    ready_flag = threading.Event()
    ready_flag.set()

    class _ReconnectAdapter:
        def __init__(self):
            self.set_calls = 0

        def set(self):
            self.set_calls += 1
            if on_reconnect is not None:
                on_reconnect(server)

    server._reconnect_event = _ReconnectAdapter()
    server._ready = ready_flag
    server._is_recycled_stdio.return_value = False
    # The fast-fail gate requires a callable returning a real bool
    # (MagicMock's truthy Mock is deliberately ignored).
    server._stdio_children_dead = children_dead

    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup(mcp_tool_module, name: str) -> None:
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)


def test_precall_dead_children_respawn_and_retry(monkeypatch, tmp_path):
    """Dead-at-call-time subprocess (the gateway-restart case): respawn,
    retry once, and hand the model a normal result — no error at all."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    called = {"n": 0}
    alive = {"v": False}

    async def _call_tool(*a, **kw):
        called["n"] += 1
        return _success_result()

    def _respawn(server):
        # What the server task does after a gateway restart: fresh child,
        # fresh session object, _ready re-armed.
        alive["v"] = True
        new_session = MagicMock()
        new_session.call_tool = _call_tool
        server.session = new_session
        server._ready.set()

    server = _install_stub_server(
        mcp_tool, "srv-dead", _call_tool,
        children_dead=lambda: not alive["v"],
        on_reconnect=_respawn,
    )
    _mcp_loop._ensure_mcp_loop()
    try:
        handler = _make_tool_handler("srv-dead", "tool1", 10.0)
        parsed = json.loads(handler({}))
        assert "error" not in parsed, parsed
        assert parsed["result"] == "ok", parsed
        assert server._reconnect_event.set_calls == 1
        assert called["n"] == 1, "exactly one RPC — the retry after respawn"
        assert mcp_tool._server_error_counts.get("srv-dead", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv-dead")


def test_midcall_child_exit_respawn_and_retry(monkeypatch, tmp_path):
    """Subprocess dies while the RPC is in flight → respawn and retry once,
    so the caller still gets its result."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    alive = {"v": True}

    async def _hanging_call(*a, **kw):
        await asyncio.sleep(30)

    async def _good_call(*a, **kw):
        return _success_result()

    async def _watch_children():
        # Resolves immediately while the child is dead; never while alive.
        while alive["v"]:
            await asyncio.sleep(0.05)

    def _respawn(server):
        alive["v"] = True
        new_session = MagicMock()
        new_session.call_tool = _good_call
        server.session = new_session
        server._ready.set()

    server = _install_stub_server(
        mcp_tool, "srv-midcall", _hanging_call,
        children_dead=lambda: not alive["v"],
        on_reconnect=_respawn,
    )
    server._watch_stdio_children = _watch_children
    _mcp_loop._ensure_mcp_loop()
    try:
        handler = _make_tool_handler("srv-midcall", "tool1", 10.0)
        # The child dies once the RPC is in flight.
        alive["v"] = False
        parsed = json.loads(handler({}))
        assert "error" not in parsed, parsed
        assert parsed["result"] == "ok", parsed
        assert server._reconnect_event.set_calls == 1
    finally:
        _cleanup(mcp_tool, "srv-midcall")


def test_dead_child_never_returning_is_not_reported_as_a_timeout(
    monkeypatch, tmp_path,
):
    """No fresh session inside the respawn window → a clean error that says
    the subprocess exited, never that something timed out (the
    old "failing the call fast instead of waiting 300s" wording sent the
    investigation into a healthy remote backend)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    monkeypatch.setattr(mcp_tool, "_STDIO_RESPAWN_WAIT_SEC", 1.0)
    called = {"n": 0}

    async def _call_tool(*a, **kw):
        called["n"] += 1
        return _success_result()

    server = _install_stub_server(
        mcp_tool, "srv-gone", _call_tool, children_dead=lambda: True,
    )
    _mcp_loop._ensure_mcp_loop()
    try:
        handler = _make_tool_handler("srv-gone", "tool1", 300.0)
        parsed = json.loads(handler({}))
        assert "error" in parsed, parsed
        message = parsed["error"]
        assert "exited" in message, message
        for forbidden in ("TimeoutError", "300s", "timed out"):
            assert forbidden not in message, message
        assert server._reconnect_event.set_calls == 1
        assert called["n"] == 0, "RPC must not be attempted on a dead transport"
        assert mcp_tool._server_error_counts.get("srv-gone", 0) == 1
    finally:
        _cleanup(mcp_tool, "srv-gone")


def test_child_dying_again_after_respawn_does_not_hot_cycle(
    monkeypatch, tmp_path,
):
    """A server whose child dies immediately after every respawn gets ONE
    retry per call, not an endless respawn loop — run()'s rapid-drop budget
    is what parks it, and this path must not fight that."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools import mcp_tool_loop as _mcp_loop
    from tools.mcp_tool_handlers import _make_tool_handler

    monkeypatch.setattr(mcp_tool, "_STDIO_RESPAWN_WAIT_SEC", 1.0)
    called = {"n": 0}

    async def _call_tool(*a, **kw):
        called["n"] += 1
        return _success_result()

    def _respawn_then_die(server):
        # Fresh session object (so the readiness wait succeeds) whose child
        # is already dead again by the time the retry dispatches.
        new_session = MagicMock()
        new_session.call_tool = _call_tool
        server.session = new_session
        server._ready.set()

    server = _install_stub_server(
        mcp_tool, "srv-flap", _call_tool,
        children_dead=lambda: True,
        on_reconnect=_respawn_then_die,
    )
    _mcp_loop._ensure_mcp_loop()
    try:
        handler = _make_tool_handler("srv-flap", "tool1", 10.0)
        parsed = json.loads(handler({}))
        assert "error" in parsed, parsed
        assert "exited again" in parsed["error"], parsed
        assert "do NOT retry" in parsed["error"], parsed
        assert server._reconnect_event.set_calls == 1, (
            "one respawn request per tool call — never a retry loop"
        )
        assert called["n"] == 0
        assert mcp_tool._server_error_counts.get("srv-flap", 0) == 1
    finally:
        _cleanup(mcp_tool, "srv-flap")
