"""Regression tests for the MCP keepalive / in-flight tool-call race.

Background
==========
An MCP stdio session is a SINGLE JSON-RPC stream. The idle keepalive
(``list_tools`` / ``send_ping``) could fire WHILE a normal ``call_tool`` was
in flight, wedging the stream so the in-flight call timed out. That timeout
triggered a false reconnect, and the SDK does not always fail the pending
``call_tool`` when its streams close — so its ``run_coroutine_threadsafe``
future never resolved and the calling agent thread polled to the full
``tool_timeout`` (up to hours).

The fix:
  * the keepalive skips a cycle when a call is in flight (and otherwise runs
    under the same ``_rpc_lock`` tool calls use, so the two can't overlap);
  * a reconnect/shutdown teardown calls ``_fail_inflight_calls`` to cancel the
    pending call tasks; and
  * ``_call`` converts that deliberate cancellation into a clean, retryable
    error so the agent recovers on the freshly rebuilt session.

These tests exercise the in-flight bookkeeping and the teardown behavior
directly (no live MCP server required).
"""

from __future__ import annotations

import asyncio


def test_new_server_starts_with_empty_inflight_state():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("init-test")
    assert server._inflight_tasks == set()
    assert server._reconnecting is False


def test_fail_inflight_calls_is_noop_when_nothing_in_flight():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("noop-test")
    # No in-flight tasks: must not flip the teardown flag (so a later genuine
    # cancel isn't misread as a deliberate reconnect).
    server._fail_inflight_calls("reconnect")
    assert server._reconnecting is False


def test_fail_inflight_calls_cancels_pending_and_flags_teardown():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("cancel-test")

    async def drive():
        async def _long():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_long())
        server._inflight_tasks.add(task)
        await asyncio.sleep(0)  # let the task start

        server._fail_inflight_calls("reconnect")
        assert server._reconnecting is True

        # The pending task must have been cancelled.
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            return "cancelled"
        except asyncio.TimeoutError:
            return "still_running"
        return "completed"

    assert asyncio.run(drive()) == "cancelled"


def test_inflight_task_tracking_add_and_discard():
    """The in-flight set tracks a running task and discards it on completion,
    mirroring the add/finally-discard bookkeeping in ``_call``."""
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("track-test")

    async def drive():
        async def _work():
            task = asyncio.current_task()
            server._inflight_tasks.add(task)
            try:
                assert task in server._inflight_tasks
            finally:
                server._inflight_tasks.discard(task)

        await asyncio.create_task(_work())
        return server._inflight_tasks

    assert asyncio.run(drive()) == set()


def test_reconnecting_flag_distinguishes_deliberate_teardown():
    """``_reconnecting`` is the signal ``_call`` reads to convert a cancellation
    into a retryable error vs. re-raising a genuine (external) cancel."""
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("flag-test")
    assert server._reconnecting is False
    # Simulate what a teardown does when there IS an in-flight task.

    async def drive():
        async def _long():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_long())
        server._inflight_tasks.add(task)
        await asyncio.sleep(0)
        server._fail_inflight_calls("shutdown")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return server._reconnecting

    assert asyncio.run(drive()) is True
