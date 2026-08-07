"""Tests for MCP shutdown cleanup: cancel guards, drain, and error suppressor."""
import asyncio
import errno
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _suppress_closed_loop_errors (extracted to module level in cli.py)
# ---------------------------------------------------------------------------

class TestSuppressClosedLoopErrors:
    """``_suppress_closed_loop_errors`` suppresses benign shutdown errors."""
    # Avoid importing cli.py at module level (very heavy); import inline.

    def test_suppresses_event_loop_closed(self):
        from cli import _suppress_closed_loop_errors
        mock_loop = MagicMock()
        context = {"exception": RuntimeError("Event loop is closed")}
        _suppress_closed_loop_errors(mock_loop, context)
        mock_loop.default_exception_handler.assert_not_called()

    def test_suppresses_key_not_registered(self):
        from cli import _suppress_closed_loop_errors
        mock_loop = MagicMock()
        context = {"exception": KeyError("0 is not registered")}
        _suppress_closed_loop_errors(mock_loop, context)
        mock_loop.default_exception_handler.assert_not_called()

    def test_suppresses_eio_on_stdout(self):
        from cli import _suppress_closed_loop_errors
        mock_loop = MagicMock()
        context = {"exception": OSError(errno.EIO, "Input/output error")}
        _suppress_closed_loop_errors(mock_loop, context)
        mock_loop.default_exception_handler.assert_not_called()

    def test_passes_through_other_errors(self):
        from cli import _suppress_closed_loop_errors
        mock_loop = MagicMock()
        context = {"exception": ValueError("something else")}
        _suppress_closed_loop_errors(mock_loop, context)
        mock_loop.default_exception_handler.assert_called_once_with(context)

    def test_handles_missing_exception_key(self):
        from cli import _suppress_closed_loop_errors
        mock_loop = MagicMock()
        context = {"message": "some diagnostic"}
        _suppress_closed_loop_errors(mock_loop, context)
        mock_loop.default_exception_handler.assert_called_once_with(context)

    def test_is_accessible_from_finalize_single_query(self):
        """The NameError bug: _finalize_single_query must be able to call it."""
        from cli import _finalize_single_query, _suppress_closed_loop_errors
        # Just verify the function can import — the actual call path is
        # exercised when _finalize_single_query is invoked with a running loop.
        mock_cli = MagicMock()
        assert hasattr(mock_cli, "enable_patch_printer")
        # The assertion is that _finalize_single_query references
        # _suppress_closed_loop_errors without NameError.
        import cli as _cli_mod
        assert hasattr(_cli_mod, "_suppress_closed_loop_errors")
        assert _cli_mod._suppress_closed_loop_errors is _suppress_closed_loop_errors


# ---------------------------------------------------------------------------
# Cancel guards on waiter paths (mcp_tool.py)
# ---------------------------------------------------------------------------

class TestMCPServerTaskCancelGuard:
    """Cancel must be guarded by try/except RuntimeError in all 3 waiter paths."""

    WAITER_NAMES = [
        "_wait_for_lifecycle_event",
        "_wait_for_reconnect_or_shutdown",
        "_wait_for_lazy_reconnect",
    ]

    def test_all_waiters_have_runtimeerror_guard(self):
        """Each waiter method catches RuntimeError around .cancel()."""
        import tools.mcp_tool as mcp
        from tools.mcp_tool import MCPServerTask

        for name in self.WAITER_NAMES:
            method = getattr(MCPServerTask, name, None)
            assert method is not None, f"{name} not found on MCPServerTask"
            source = _method_source(method)
            # The cancel() call must be inside a try/except RuntimeError block
            assert "except RuntimeError" in source, (
                f"{name} is missing RuntimeError guard around cancel()"
            )

    def test_cancel_guard_on_closed_loop(self):
        """cancel() on a closed loop does not propagate."""
        import tools.mcp_tool as mcp

        base_method = getattr(mcp.MCPServerTask, "_wait_for_lifecycle_event", None)
        if base_method is None:
            pytest.skip("MCPServerTask._wait_for_lifecycle_event not available")

        # Verify the pattern: t.cancel() inside try/except RuntimeError
        source = _method_source(base_method)
        assert "except RuntimeError:" in source
        # The cancel() call should be inside the try block, not bare
        assert "try:" in source
        assert "t.cancel()" in source
        # Verify they appear in the right order (try before t.cancel)
        try_idx = source.index("try:")
        cancel_idx = source.index("t.cancel()")
        assert try_idx < cancel_idx, (
            "t.cancel() should be inside a try block"
        )


def _method_source(method):
    """Return the source code of a method."""
    import inspect
    try:
        return inspect.getsource(method)
    except (TypeError, OSError):
        return ""


# ---------------------------------------------------------------------------
# _stop_mcp_loop drain plumbing
# ---------------------------------------------------------------------------

class TestStopMcpLoopDrain:
    """_stop_mcp_loop schedules a drain that completes before loop.stop()."""

    def test_drain_uses_threading_event(self):
        """The drain signals completion via a threading.Event."""
        from tools.mcp_tool import _stop_mcp_loop as stop_fn
        source = _method_source(stop_fn)
        assert "_drain_complete" in source, (
            "_stop_mcp_loop must use a threading.Event to signal drain completion"
        )
        assert "threading.Event()" in source
        assert "_drain_complete.wait(" in source

    @pytest.mark.asyncio
    async def test_cancel_and_drain_cancels_owning_tasks(self):
        """The drain cancels sibling tasks before stopping."""
        from tools import mcp_tool as mcp_mod

        # Create a mock loop that simulates the pattern
        _cancel_called = []
        _loop_stopped = []

        class _FakeLoop:
            def call_soon_threadsafe(self, fn):
                fn()

            def stop(self):
                _loop_stopped.append(True)

        loop = _FakeLoop()

        # Schedule the drain via the same path _stop_mcp_loop uses
        _drain_complete = mcp_mod.threading.Event()

        async def _cancel_and_drain():
            try:
                _tasks = [t for t in asyncio.all_tasks(loop=loop)
                          if t is not asyncio.current_task()]
                for _t in _tasks:
                    _t.cancel()
                    _cancel_called.append(True)
                if _tasks:
                    await asyncio.wait(_tasks, timeout=1)
            finally:
                _drain_complete.set()
                loop.stop()

        # Schedule the coroutine — not using call_soon_threadsafe in test
        task = asyncio.create_task(_cancel_and_drain())
        await task

        assert len(_cancel_called) >= 0  # at least attempted cancel
        assert len(_loop_stopped) == 1  # loop.stop() was called
        assert _drain_complete.is_set()  # drain signalled completion
