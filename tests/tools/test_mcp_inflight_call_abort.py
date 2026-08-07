"""Session teardown must abort in-flight tool calls, not orphan them.

Companion to test_mcp_keepalive_rpc_lock_guard.py (both from the
deleg_e8af57bc investigation, 2026-08-02). When a transport genuinely
dies mid-call, the dying session's teardown never cancels foreign
awaiters: before the fix the pending ``session.call_tool`` hung forever
holding ``_rpc_lock``, starving every later call to that server (a
read-only status call queued 403s behind a dead wait) until an outer
watchdog interrupted the agent. Teardown now cancels registered
in-flight calls so they fail fast with ``McpCallAbortedError`` and
release the lock.
"""

import asyncio
import threading
import time

import pytest


def _install_server(name, call_tool):
    """Register a real MCPServerTask backed by a stub session."""
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    mcp_tool._ensure_mcp_loop()

    class _Session:
        pass

    session = _Session()
    session.call_tool = call_tool

    server = MCPServerTask(name)
    server._config = {"command": "x"}
    server.session = session
    mcp_tool._servers[name] = server
    mcp_tool._server_error_counts.pop(name, None)
    return server


def _run_handler_in_thread(handler):
    box = {}

    def _target():
        box["result"] = handler({})

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, box


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_teardown_aborts_inflight_call_and_releases_lock():
    """An aborted call returns a verify-first error and frees _rpc_lock."""
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    async def _never_answers(tool_name, arguments=None):
        await asyncio.Event().wait()

    server = _install_server("abort-srv", _never_answers)
    handler = _make_tool_handler("abort-srv", "slow_tool", tool_timeout=30)
    thread, box = _run_handler_in_thread(handler)

    assert _wait_for(lambda: server._inflight_calls), "call never registered"
    mcp_tool._mcp_loop.call_soon_threadsafe(server._abort_inflight_calls)

    thread.join(timeout=10)
    assert not thread.is_alive(), "handler still blocked after abort"
    assert "connection was reset" in box["result"]
    assert "read-only" in box["result"]
    assert not server._rpc_lock.locked(), "_rpc_lock leaked after abort"


def test_next_call_proceeds_after_abort():
    """The lock released by an abort must admit the next queued caller."""
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    state = {"calls": 0}

    async def _first_hangs(tool_name, arguments=None):
        state["calls"] += 1
        if state["calls"] == 1:
            await asyncio.Event().wait()

        class _Result:
            isError = False
            content = []
            structuredContent = {"ok": True}

        return _Result()

    server = _install_server("abort-then-ok", _first_hangs)
    handler = _make_tool_handler("abort-then-ok", "tool", tool_timeout=30)

    first_thread, first_box = _run_handler_in_thread(handler)
    assert _wait_for(lambda: server._inflight_calls)

    second_thread, second_box = _run_handler_in_thread(handler)
    assert _wait_for(lambda: server._rpc_lock.locked())

    mcp_tool._mcp_loop.call_soon_threadsafe(server._abort_inflight_calls)

    first_thread.join(timeout=10)
    second_thread.join(timeout=10)
    assert "connection was reset" in first_box["result"]
    assert second_box["result"] == '{"result": {"ok": true}}'


def test_lifecycle_exit_aborts_inflight_calls():
    """Every _wait_for_lifecycle_event exit path sweeps in-flight calls."""
    from tools.mcp_tool import MCPServerTask

    async def _scenario():
        server = MCPServerTask("srv")
        server._config = {"command": "x", "keepalive_interval": 30}
        server.session = object()

        inner = asyncio.ensure_future(asyncio.Event().wait())
        server._inflight_calls.add(inner)

        server._shutdown_event.set()
        result = await asyncio.wait_for(
            server._wait_for_lifecycle_event(), timeout=5
        )
        assert result == "shutdown"
        with pytest.raises(asyncio.CancelledError):
            await inner

    asyncio.run(_scenario())


def test_user_interrupt_keeps_semantics_and_cancels_rpc(monkeypatch):
    """A user interrupt must keep its semantics: the handler returns the
    interrupted-call result (not McpCallAbortedError), and the detached
    RPC task is cancelled rather than left running holding resources."""
    import tools.interrupt as interrupt_mod
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    state = {"rpc_cancelled": False, "started": False, "interrupted": False}

    async def _hangs_until_cancelled(tool_name, arguments=None):
        state["started"] = True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            state["rpc_cancelled"] = True
            raise

    server = _install_server("interrupt-srv", _hangs_until_cancelled)
    monkeypatch.setattr(
        interrupt_mod, "is_interrupted", lambda *a, **kw: state["interrupted"]
    )

    handler = _make_tool_handler("interrupt-srv", "tool", tool_timeout=30)
    thread, box = _run_handler_in_thread(handler)

    assert _wait_for(lambda: state["started"])
    state["interrupted"] = True

    thread.join(timeout=10)
    assert not thread.is_alive(), "handler did not honor the interrupt"
    assert "interrupted" in box["result"]
    assert "connection was reset" not in box["result"]
    assert _wait_for(lambda: state["rpc_cancelled"]), (
        "RPC task left running after interrupt"
    )
    assert not server._rpc_lock.locked()
