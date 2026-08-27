"""Regression tests for stdio fast-fail reconnect signaling (#95626 salvage).

The #81995 fast-fail gate detects a dead stdio subprocess but the transport
failure never cleared ``server.session``, so the transport-down reconnect path
(which only fires when the session is gone/not-ready) never ran. The call
failed fast — correctly — but nothing asked the server task to respawn the
subprocess, so every subsequent call kept failing until the idle keepalive
probe eventually noticed. Both fast-fail sites must signal a reconnect:

- pre-call gate (children already dead when the call arrives): return a clean
  "reconnecting" tool error and set ``_reconnect_event``;
- mid-call watcher race (children die while the RPC is in flight): raise the
  fast-fail TimeoutError and set ``_reconnect_event``.
"""

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")


def _install_stub_server(mcp_tool_module, name: str, call_tool_impl,
                         *, children_dead):
    """Fake MCP server with real-bool stdio liveness and a countable
    reconnect event (mirrors tests/tools/test_mcp_circuit_breaker.py)."""
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


def test_precall_dead_children_signal_reconnect(monkeypatch, tmp_path):
    """Dead-at-call-time subprocess → clean reconnecting error + reconnect
    signal, instead of a bare fast-fail that leaves the server dead."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    called = {"n": 0}

    async def _call_tool(*a, **kw):
        called["n"] += 1
        return MagicMock(is_error=False, content=[])

    server = _install_stub_server(
        mcp_tool, "srv-dead", _call_tool, children_dead=lambda: True
    )
    mcp_tool._ensure_mcp_loop()
    try:
        handler = _make_tool_handler("srv-dead", "tool1", 10.0)
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed, parsed
        assert "reconnect" in parsed["error"].lower(), parsed
        assert server._reconnect_event.set_calls == 1
        assert called["n"] == 0, "RPC must not be attempted on a dead transport"
        # The error payload flows through the handler's JSON parse, which
        # bumps the breaker exactly once (no double-bump at the gate).
        assert mcp_tool._server_error_counts.get("srv-dead", 0) == 1
    finally:
        _cleanup(mcp_tool, "srv-dead")


def test_midcall_child_exit_signals_reconnect(monkeypatch, tmp_path):
    """Subprocess dies while the RPC is in flight → fast-fail error AND a
    reconnect signal so the next call lands on a respawned transport."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    async def _hanging_call(*a, **kw):
        await asyncio.sleep(30)

    server = _install_stub_server(
        mcp_tool, "srv-midcall", _hanging_call, children_dead=lambda: False
    )

    async def _watch_children():
        return  # children die immediately → watcher resolves first

    server._watch_stdio_children = _watch_children
    mcp_tool._ensure_mcp_loop()
    try:
        handler = _make_tool_handler("srv-midcall", "tool1", 10.0)
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed, parsed
        assert "exited mid-call" in parsed["error"], parsed
        assert server._reconnect_event.set_calls == 1
    finally:
        _cleanup(mcp_tool, "srv-midcall")
