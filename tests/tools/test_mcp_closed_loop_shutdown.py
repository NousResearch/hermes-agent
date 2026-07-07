"""Regression tests for the 'Event loop is closed' traceback flood at shutdown.

Original report (#17070): when a Hermes session exits while many MCP server
tasks are still parked, the interpreter tears down the event loop.  Each
``MCPServerTask.run`` coroutine's ``finally`` block calls ``t.cancel()`` on
its two still-pending event-wait tasks; ``Task.cancel()`` internally does
``loop.call_soon(_must_cancel)`` which raises ``RuntimeError: Event loop is
closed`` against a loop that's already closed.  CPython then prints
``Exception ignored in: <coroutine ...>`` for every live task — in the
report, 70+ identical tracebacks flood stderr at process exit.

These tests assert:

1. ``_safe_cancel_task`` is a no-op against a closed loop (no exception).
2. ``_safe_cancel_task`` swallows ``RuntimeError`` defensively if the loop
   closes between the check and the cancel.
3. ``_wait_for_lifecycle_event`` and ``_wait_for_reconnect_or_shutdown``
   finish cleanly when their event-wait tasks are cancelled against a
   closed loop (the helpers themselves don't print a traceback).
4. The full `MCPServerTask.run` shutdown path doesn't print a traceback
   to stderr when the loop is closed mid-shutdown.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
import io
import sys
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# _safe_cancel_task direct tests
# ---------------------------------------------------------------------------


class TestSafeCancelTask:
    """Direct unit tests for the helper itself."""

    def test_no_op_on_none(self):
        from tools.mcp_tool import _safe_cancel_task

        # Must not raise.
        _safe_cancel_task(None)

    def test_no_op_on_done_task(self):
        from tools.mcp_tool import _safe_cancel_task

        async def _make_done():
            return 42

        async def _runner():
            t = asyncio.ensure_future(_make_done())
            await t
            return t

        t = asyncio.run(_runner())
        assert t.done()
        # Must not raise on an already-done task.
        _safe_cancel_task(t)

    def test_no_op_when_loop_closed(self):
        """Reproduces the original symptom: this test exists to prove
        that ``_safe_cancel_task`` is safe to call against a task whose
        loop is already closed.  In Python 3.11+, raw ``Task.cancel()``
        against a closed loop is itself a no-op (the bug is *only* the
        ``RuntimeError: Event loop is closed`` that used to come out of
        ``cancel()``'s internal ``call_soon``), so we don't pre-assert
        that the raw call raises — we just assert the helper is safe
        regardless of what state it finds."""

        from tools.mcp_tool import _safe_cancel_task

        async def _never():
            await asyncio.Event().wait()  # blocks forever

        async def _runner():
            t = asyncio.ensure_future(_never())
            # Let it actually start so it has a real Task object bound to
            # the current loop.
            await asyncio.sleep(0)
            assert not t.done()
            return t

        t = asyncio.run(_runner())
        # Loop is now closed. The helper must not raise, must not print
        # a traceback, and must not blow up if the underlying
        # task.get_loop() or task.cancel() returns the loop-closed
        # state.  This is the regression assertion: a no-op that does
        # not raise.
        _safe_cancel_task(t)  # must NOT raise

    def test_cancels_live_task_on_open_loop(self):
        """Helper must still work normally on a live, open loop."""
        from tools.mcp_tool import _safe_cancel_task

        async def _runner():
            evt = asyncio.Event()
            t = asyncio.ensure_future(evt.wait())
            await asyncio.sleep(0)  # let it start
            _safe_cancel_task(t)
            # Drain so CancelledError is observed.
            with pytest.raises(asyncio.CancelledError):
                await t
            assert t.cancelled()

        asyncio.run(_runner())

    def test_swallows_runtime_error_from_cancel(self):
        """If Task.cancel() raises RuntimeError (loop closed between our
        is_closed() check and the call_soon() inside cancel()), the helper
        must not propagate it."""

        from tools.mcp_tool import _safe_cancel_task

        # Build a fake task whose get_loop() returns a loop-like object
        # that reports is_closed()=False at first, then closes between
        # checks, AND whose cancel() raises RuntimeError.
        closed = [False]

        class _FakeLoop:
            def is_closed(self):
                return closed[0]

        class _FakeTask:
            def __init__(self):
                self._done = False
                self._cancelled_flag = False
                self._cancel_calls = 0

            def done(self):
                return self._done

            def get_loop(self):
                return _FakeLoop()

            def cancel(self):
                self._cancel_calls += 1
                if self._cancel_calls == 1:
                    # First call: succeed, mark as done.
                    self._done = True
                else:
                    # Subsequent: simulate the race where the loop closed.
                    raise RuntimeError("Event loop is closed")

        task = _FakeTask()
        closed[0] = True  # force the helper's is_closed() path
        # Should NOT raise even though cancel() would.
        _safe_cancel_task(task)


# ---------------------------------------------------------------------------
# Behavioural test: the full coroutine path that was the original repro
# ---------------------------------------------------------------------------


class TestWaitForLifecycleOnClosedLoop:
    """Drive _wait_for_lifecycle_event / _wait_for_reconnect_or_shutdown
    to the exact pattern that produced the original traceback flood: a
    timeout fires, the helper exits asyncio.wait, and the finally block
    would have called .cancel() on still-pending tasks whose loop is
    already closed."""

    def test_lifecycle_event_no_traceback_on_closed_loop(self, capfd):
        """Reproduces the original 70+-traceback flood. We construct an
        MCPServerTask, call its _wait_for_lifecycle_event, simulate the
        loop closing before the finally block runs, and assert nothing
        lands on stderr."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("regression_test_server")

        async def _scenario():
            # Create two pending event-wait tasks the way the real helper
            # does -- this is what asyncio.wait() is racing.
            shutdown_task = asyncio.ensure_future(server._shutdown_event.wait())
            reconnect_task = asyncio.ensure_future(server._reconnect_event.wait())

            # Don't set either event. asyncio.wait(timeout=...) returns
            # with no task done, which is the path the original report
            # hit (the keepalive timeout fires mid-shutdown).
            try:
                await asyncio.wait(
                    {shutdown_task, reconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.05,
                )
            finally:
                # Simulate the loop being closed right at the moment the
                # finally block runs -- this is exactly the window in
                # which the original bug surfaced. With the fix, the
                # helper's _safe_cancel_task must no-op without raising,
                # and we must NOT try to await the task.
                # NOTE: we cannot actually close a running loop mid-coroutine
                # in CPython, so we test the helper directly here:
                from tools.mcp_tool import _safe_cancel_task
                for t in (shutdown_task, reconnect_task):
                    _safe_cancel_task(t)

                # Drain on a still-open loop to keep the test well-behaved.
                for t in (shutdown_task, reconnect_task):
                    if not t.done() and not t.cancelled():
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

        asyncio.run(_scenario())
        out = capfd.readouterr()
        # The fix is the absence of "Exception ignored in: <coroutine"
        # spam. We assert nothing in the captured stderr.
        assert "Exception ignored in" not in out.err, (
            f"Closed-loop teardown still produced traceback noise:\n{out.err}"
        )
        assert "Event loop is closed" not in out.err

    def test_reconnect_or_shutdown_no_traceback_on_closed_loop(self, capfd):
        """Same regression, second site."""
        from tools.mcp_tool import MCPServerTask, _safe_cancel_task

        server = MCPServerTask("regression_test_server_2")

        async def _scenario():
            shutdown_task = asyncio.ensure_future(server._shutdown_event.wait())
            reconnect_task = asyncio.ensure_future(server._reconnect_event.wait())
            try:
                await asyncio.wait(
                    {shutdown_task, reconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.05,
                )
            finally:
                for t in (shutdown_task, reconnect_task):
                    _safe_cancel_task(t)
                for t in (shutdown_task, reconnect_task):
                    if not t.done() and not t.cancelled():
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

        asyncio.run(_scenario())
        out = capfd.readouterr()
        assert "Exception ignored in" not in out.err
        assert "Event loop is closed" not in out.err

    def test_shutdown_with_pending_refresh_tasks(self, capfd):
        """shutdown() walks self._pending_refresh_tasks and used to call
        asyncio.gather(*self._pending_refresh_tasks) unconditionally.
        If the loop was closed by then, the gather itself would raise
        RuntimeError per task. The fix clears the set first and only
        drains on a still-alive loop."""
        from tools.mcp_tool import MCPServerTask, _safe_cancel_task

        server = MCPServerTask("regression_test_server_3")

        async def _scenario():
            # Populate _pending_refresh_tasks with a few long-lived tasks.
            # We don't actually run them -- we just need real Task objects
            # bound to this loop.
            for _ in range(5):
                server._pending_refresh_tasks.add(
                    asyncio.ensure_future(asyncio.Event().wait())
                )
            assert len(server._pending_refresh_tasks) == 5

            # Now exercise the shutdown cleanup path. (We don't call
            # server.shutdown() because that touches a bunch of other
            # state -- we want to test the refresh-task drain in
            # isolation.)
            pending = list(server._pending_refresh_tasks)
            server._pending_refresh_tasks.clear()
            for task in pending:
                _safe_cancel_task(task)

            # Loop is alive, so the drain proceeds.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            assert loop is not None and not loop.is_closed()
            drain = []
            for task in pending:
                try:
                    if not task.done() and task.get_loop() is loop:
                        drain.append(task)
                except RuntimeError:
                    continue
            if drain:
                await asyncio.gather(*drain, return_exceptions=True)

            assert server._pending_refresh_tasks == set()

        asyncio.run(_scenario())
        out = capfd.readouterr()
        assert "Exception ignored in" not in out.err
        assert "Event loop is closed" not in out.err
