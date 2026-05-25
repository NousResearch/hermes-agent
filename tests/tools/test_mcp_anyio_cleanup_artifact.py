"""Regression tests for benign anyio cleanup artifacts during MCP reconnect.

See #31987.  When ``async with streamable_http_client(...)`` unwinds for
a planned reconnect on an HTTP/StreamableHTTP MCP server, anyio's
TaskGroup teardown can raise ``RuntimeError("The current task is not
holding this lock")`` even though the session shut down cleanly.  The
previous reconnect loop treated this as a real connection failure and
burned a slot in the reconnect budget, so every keepalive-driven
reconnect cost one retry attempt and the server was permanently
disconnected after ``_MAX_RECONNECT_RETRIES`` cleanup events.
"""

import asyncio
from unittest.mock import patch


def test_anyio_cleanup_marker_matches_bare_runtime_error():
    """``not holding this lock`` is recognized on a bare RuntimeError."""
    from tools.mcp_tool import _is_anyio_cleanup_artifact

    exc = RuntimeError("The current task is not holding this lock")
    assert _is_anyio_cleanup_artifact(exc) is True


def test_anyio_cleanup_marker_unwraps_exception_group():
    """``ExceptionGroup`` wrapping the anyio RuntimeError still matches."""
    from tools.mcp_tool import _is_anyio_cleanup_artifact

    inner = RuntimeError("The current task is not holding this lock")
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [inner])
    assert _is_anyio_cleanup_artifact(group) is True


def test_anyio_cleanup_marker_unwraps_nested_groups():
    """Nested ``ExceptionGroup`` chains are walked, not just one level."""
    from tools.mcp_tool import _is_anyio_cleanup_artifact

    inner = RuntimeError("The current task is not holding this lock")
    nested = BaseExceptionGroup("inner group", [inner])
    outer = BaseExceptionGroup("outer group", [nested])
    assert _is_anyio_cleanup_artifact(outer) is True


def test_anyio_cleanup_marker_rejects_unrelated_runtime_error():
    """Plain ``RuntimeError`` without the marker is not an anyio artifact."""
    from tools.mcp_tool import _is_anyio_cleanup_artifact

    assert _is_anyio_cleanup_artifact(RuntimeError("DNS resolution failed")) is False
    assert _is_anyio_cleanup_artifact(ConnectionError("ECONNRESET")) is False


def test_reconnect_loop_ignores_anyio_cleanup_artifact():
    """Anyio cleanup error after the server is ready does not burn a retry.

    Reproduces the #31987 failure mode: each reconnect causes a
    ``RuntimeError`` from ``streamable_http_client.__aexit__``.  Without
    the fix, the retry counter would exhaust quickly and the server
    would be given up on.  With the fix, the loop treats it as benign
    cleanup, resets backoff, and reconnects immediately.
    """
    from tools.mcp_tool import MCPServerTask, _MAX_RECONNECT_RETRIES

    call_count = 0

    async def _run():
        nonlocal call_count
        server = MCPServerTask("test-anyio-cleanup")

        # First call: succeed (marks server ready), then raise the anyio
        # cleanup artifact several times in a row, then finally exit cleanly
        # on shutdown.
        async def fake_run_http(self_inner, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Mark as initially connected so subsequent failures hit
                # the "session was ready" branch, not the initial-connect
                # retry budget.
                self_inner._ready.set()
                raise RuntimeError("The current task is not holding this lock")
            if call_count <= _MAX_RECONNECT_RETRIES + 2:
                # More cleanup-artifact failures than the reconnect budget
                # would normally tolerate.  Each must NOT count.
                raise BaseExceptionGroup(
                    "unhandled errors in a TaskGroup",
                    [RuntimeError("The current task is not holding this lock")],
                )
            # Eventually exit cleanly on shutdown.
            await self_inner._shutdown_event.wait()

        _orig_sleep = asyncio.sleep

        async def _instant_sleep(*_a, **_k):
            await _orig_sleep(0)

        with patch.object(MCPServerTask, "_run_http", fake_run_http), \
             patch.object(MCPServerTask, "_is_http", lambda self_inner: True), \
             patch("asyncio.sleep", new=_instant_sleep):
            task = asyncio.ensure_future(server.run({"url": "https://example.test/mcp"}))
            await server._ready.wait()

            # Yield enough times for the cleanup-artifact failures to cycle.
            for _ in range(_MAX_RECONNECT_RETRIES + 5):
                await asyncio.sleep(0)

            server._shutdown_event.set()
            server._reconnect_event.set()
            await task

        # The loop survived more cleanup events than _MAX_RECONNECT_RETRIES
        # would have permitted if they had counted as real failures.
        assert call_count > _MAX_RECONNECT_RETRIES + 1, (
            f"Expected >{_MAX_RECONNECT_RETRIES + 1} attempts (cleanup did not "
            f"burn the retry budget); got {call_count}"
        )
        # And no error was latched onto the server.
        assert server._error is None

    asyncio.run(_run())


def test_reconnect_loop_still_counts_real_failures():
    """Non-anyio reconnect failures still count against the retry budget."""
    from tools.mcp_tool import MCPServerTask, _MAX_RECONNECT_RETRIES

    call_count = 0

    async def _run():
        nonlocal call_count
        server = MCPServerTask("test-real-failures")

        async def fake_run_http(self_inner, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Mark as initially connected, then start failing for real.
                self_inner._ready.set()
                raise ConnectionError("ECONNRESET")
            raise ConnectionError("ECONNRESET")

        _orig_sleep = asyncio.sleep

        async def _instant_sleep(*_a, **_k):
            await _orig_sleep(0)

        with patch.object(MCPServerTask, "_run_http", fake_run_http), \
             patch.object(MCPServerTask, "_is_http", lambda self_inner: True), \
             patch("asyncio.sleep", new=_instant_sleep):
            task = asyncio.ensure_future(server.run({"url": "https://example.test/mcp"}))
            # Server should ready on first attempt, then exhaust retries.
            await server._ready.wait()
            await task

        # Real failures should exhaust the reconnect budget — the first
        # attempt sets ready then raises, and each subsequent failure
        # increments ``retries`` until ``retries > _MAX_RECONNECT_RETRIES``
        # triggers "give up".  That bounds total ``_run_http`` calls at
        # ``_MAX_RECONNECT_RETRIES + 1``.
        assert call_count == _MAX_RECONNECT_RETRIES + 1, (
            f"Expected {_MAX_RECONNECT_RETRIES + 1} attempts on real "
            f"failures; got {call_count}"
        )

    asyncio.run(_run())
