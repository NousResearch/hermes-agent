"""Regression tests for ``MCPServerTask.run`` + ``asyncio.CancelledError``.

Background
==========
On Python 3.11+, ``asyncio.CancelledError`` inherits from ``BaseException``
rather than ``Exception``, so a bare ``except Exception`` does NOT catch it.
``MCPServerTask.run`` had a broad ``except Exception`` around the transport
loop which meant a task cancellation (gateway restart, explicit
``task.cancel()``) caused the reconnect loop to exit silently — the MCP
server stayed dead until Hermes was restarted. See #9930.

The fix adds an explicit ``except asyncio.CancelledError: raise`` BEFORE
the broad catch so cancellation propagates cleanly to asyncio's task
machinery and ``MCPServerTask.shutdown()``'s ``await self._task`` completes
without hanging the reconnect loop.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest



async def _hanging_run(self, cfg):
    """Stand-in transport that hangs forever so we can cancel it."""
    await asyncio.sleep(3600)


class TestCancelledErrorPropagation:
    def test_cancel_task_if_loop_open_skips_closed_owning_loop(self):
        """Closed-loop cleanup must not call ``Task.cancel()``.

        ``Task.cancel()`` schedules cancellation with ``loop.call_soon()``; if
        coroutine finalization runs after the MCP loop is closed, that raises
        ``RuntimeError("Event loop is closed")``. The helper should detect the
        closed owning loop and return False without raising.
        """
        from tools.mcp_tool import MCPServerTask

        loop = asyncio.new_event_loop()
        task = loop.create_task(asyncio.sleep(3600))
        task._log_destroy_pending = False

        try:
            loop.close()

            assert task.done() is False
            assert task.get_loop().is_closed() is True
            assert MCPServerTask._cancel_task_if_loop_open(task) is False
        finally:
            task.get_coro().close()

    def test_cancel_task_if_loop_open_preserves_live_loop_cancellation(self):
        """Open-loop tasks should still receive ordinary cancellation."""
        from tools.mcp_tool import MCPServerTask

        loop = asyncio.new_event_loop()
        task = loop.create_task(asyncio.sleep(3600))
        task._log_destroy_pending = False

        try:
            assert MCPServerTask._cancel_task_if_loop_open(task) is True
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(task)
            assert task.cancelled() is True
        finally:
            if not task.done():
                task.get_coro().close()
            loop.close()

    def test_cancelled_error_is_not_swallowed_by_except_exception(self):
        """CancelledError raised inside the transport call must re-raise
        so the reconnect loop terminates cleanly on cancel — not stay wedged."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("cancel-test")

        async def drive():
            with patch.object(MCPServerTask, "_run_stdio", _hanging_run), \
                 patch.object(MCPServerTask, "_is_http", lambda self: False):
                task = asyncio.create_task(server.run({"command": "fake"}))
                # Let the run loop enter the try/except and start awaiting.
                await asyncio.sleep(0.05)
                task.cancel()
                # The fix guarantees the task completes (either via
                # CancelledError propagation or clean exit) rather than
                # hanging forever.
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except asyncio.CancelledError:
                    return "cancelled_cleanly"
                except asyncio.TimeoutError:
                    # If we hit this, the reconnect loop swallowed the cancel
                    # and stayed wedged — the exact #9930 bug.
                    task.cancel()
                    try:
                        await task
                    except Exception:
                        pass
                    return "wedged"
                return "clean_return"

        outcome = asyncio.run(drive())
        assert outcome in {"cancelled_cleanly", "clean_return"}, (
            f"MCPServerTask.run wedged on cancel (outcome={outcome}) — "
            f"#9930 regression"
        )

    def test_shutdown_completes_promptly_when_task_is_cancelled(self):
        """``shutdown()`` falls through to ``task.cancel()`` + ``await self._task``
        after a grace period. That cancel must unwedge the reconnect loop —
        otherwise ``await self._task`` hangs indefinitely."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("shutdown-cancel-test")

        async def drive():
            with patch.object(MCPServerTask, "_run_stdio", _hanging_run), \
                 patch.object(MCPServerTask, "_is_http", lambda self: False):
                server._task = asyncio.ensure_future(server.run({"command": "fake"}))
                await asyncio.sleep(0.05)
                server._shutdown_event.set()
                server._task.cancel()
                try:
                    await asyncio.wait_for(server._task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                return server._task.done()

        done = asyncio.run(drive())
        assert done, "MCPServerTask did not finish after cancel — #9930 regression"
