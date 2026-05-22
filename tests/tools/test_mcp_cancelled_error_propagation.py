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
        assert outcome in ("cancelled_cleanly", "clean_return"), (
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
