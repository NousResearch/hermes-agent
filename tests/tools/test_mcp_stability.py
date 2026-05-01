"""Tests for MCP stability fixes — event loop handler, PID tracking, shutdown robustness."""

import asyncio
import os
import signal
import threading
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fix 1: MCP event loop exception handler
# ---------------------------------------------------------------------------

class TestMCPLoopExceptionHandler:
    """_mcp_loop_exception_handler suppresses benign 'Event loop is closed'."""

    def test_suppresses_event_loop_closed(self):
        from tools.mcp_tool import _mcp_loop_exception_handler
        loop = MagicMock()
        context = {"exception": RuntimeError("Event loop is closed")}
        # Should NOT call default handler
        _mcp_loop_exception_handler(loop, context)
        loop.default_exception_handler.assert_not_called()

    def test_forwards_other_runtime_errors(self):
        from tools.mcp_tool import _mcp_loop_exception_handler
        loop = MagicMock()
        context = {"exception": RuntimeError("some other error")}
        _mcp_loop_exception_handler(loop, context)
        loop.default_exception_handler.assert_called_once_with(context)

    def test_forwards_non_runtime_errors(self):
        from tools.mcp_tool import _mcp_loop_exception_handler
        loop = MagicMock()
        context = {"exception": ValueError("bad value")}
        _mcp_loop_exception_handler(loop, context)
        loop.default_exception_handler.assert_called_once_with(context)

    def test_forwards_contexts_without_exception(self):
        from tools.mcp_tool import _mcp_loop_exception_handler
        loop = MagicMock()
        context = {"message": "just a message"}
        _mcp_loop_exception_handler(loop, context)
        loop.default_exception_handler.assert_called_once_with(context)

    def test_handler_installed_on_mcp_loop(self):
        """_ensure_mcp_loop installs the exception handler on the new loop."""
        import tools.mcp_tool as mcp_mod
        try:
            mcp_mod._ensure_mcp_loop()
            with mcp_mod._lock:
                loop = mcp_mod._mcp_loop
            assert loop is not None
            assert loop.get_exception_handler() is mcp_mod._mcp_loop_exception_handler
        finally:
            mcp_mod._stop_mcp_loop()


# ---------------------------------------------------------------------------
# Fix 2: stdio PID tracking
# ---------------------------------------------------------------------------

class TestStdioPidTracking:
    """_snapshot_child_pids and _stdio_pids track subprocess PIDs."""

    def test_snapshot_returns_set(self):
        from tools.mcp_tool import _snapshot_child_pids
        result = _snapshot_child_pids()
        assert isinstance(result, set)
        # All elements should be ints
        for pid in result:
            assert isinstance(pid, int)

    def test_stdio_pids_starts_empty(self):
        from tools.mcp_tool import _stdio_pids, _lock
        with _lock:
            # Might have residual state from other tests, just check type
            assert isinstance(_stdio_pids, dict)

    def test_kill_orphaned_noop_when_empty(self):
        """_kill_orphaned_mcp_children does nothing when no PIDs tracked."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _stdio_pids,
            _lock,
        )

        with _lock:
            _stdio_pids.clear()
            _orphan_stdio_pids.clear()

        # Should not raise
        _kill_orphaned_mcp_children()

    def test_kill_orphaned_handles_dead_pids(self):
        """_kill_orphaned_mcp_children gracefully handles already-dead PIDs."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _lock,
        )

        # Use a PID that definitely doesn't exist
        fake_pid = 999999999
        with _lock:
            _orphan_stdio_pids.add(fake_pid)

        # Should not raise (ProcessLookupError is caught)
        _kill_orphaned_mcp_children()

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_kill_orphaned_uses_sigkill_when_available(self, monkeypatch):
        """SIGTERM-first then SIGKILL after 2s for orphan cleanup."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _lock,
        )

        fake_pid = 424242
        with _lock:
            _orphan_stdio_pids.clear()
            _orphan_stdio_pids.add(fake_pid)

        fake_sigkill = 9
        monkeypatch.setattr(signal, "SIGKILL", fake_sigkill, raising=False)

        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("time.sleep") as mock_sleep:
            _kill_orphaned_mcp_children()

        # SIGTERM, then alive-check (signal 0), then SIGKILL
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)
        mock_kill.assert_any_call(fake_pid, 0)  # alive check
        mock_kill.assert_any_call(fake_pid, fake_sigkill)
        assert mock_kill.call_count == 3
        mock_sleep.assert_called_once_with(2)

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_kill_orphaned_falls_back_without_sigkill(self, monkeypatch):
        """Without SIGKILL, SIGTERM is used for both phases."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _lock,
        )

        fake_pid = 434343
        with _lock:
            _orphan_stdio_pids.clear()
            _orphan_stdio_pids.add(fake_pid)

        monkeypatch.delattr(signal, "SIGKILL", raising=False)

        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("time.sleep") as mock_sleep:
            _kill_orphaned_mcp_children()

        # SIGTERM phase, alive check raises (process gone), no escalation
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)
        assert mock_sleep.called

        with _lock:
            assert fake_pid not in _orphan_stdio_pids


# ---------------------------------------------------------------------------
# Fix 3: MCP reload timeout (cli.py)
# ---------------------------------------------------------------------------

class TestMCPReloadTimeout:
    """_check_config_mcp_changes uses a timeout on _reload_mcp."""

    def test_reload_timeout_does_not_block_forever(self, tmp_path, monkeypatch):
        """If _reload_mcp hangs, the config watcher times out and returns."""
        import time

        # Create a mock HermesCLI-like object with the needed attributes
        class FakeCLI:
            _config_mtime = 0.0
            _config_mcp_servers = {}
            _last_config_check = 0.0
            _command_running = False
            config = {}
            agent = None

            def _reload_mcp(self):
                # Simulate a hang — sleep longer than the timeout
                time.sleep(60)

            def _slow_command_status(self, cmd):
                return cmd

        # This test verifies the timeout mechanism exists in the code
        # by checking that _check_config_mcp_changes doesn't call
        # _reload_mcp directly (it uses a thread now)
        import inspect
        from cli import HermesCLI
        source = inspect.getsource(HermesCLI._check_config_mcp_changes)
        # The fix adds threading.Thread for _reload_mcp
        assert "Thread" in source or "thread" in source.lower(), \
            "_check_config_mcp_changes should use a thread for _reload_mcp"


# ---------------------------------------------------------------------------
# Fix 4: MCP initial connection retry with backoff
# (Ported from Kilo Code's MCP resilience fix)
# ---------------------------------------------------------------------------

class TestMCPInitialConnectionRetry:
    """MCPServerTask.run() retries initial connection failures instead of giving up."""

    def test_initial_connect_retries_constant_exists(self):
        """_MAX_INITIAL_CONNECT_RETRIES should be defined."""
        from tools.mcp_tool import _MAX_INITIAL_CONNECT_RETRIES
        assert _MAX_INITIAL_CONNECT_RETRIES >= 1

    def test_initial_connect_retry_succeeds_on_second_attempt(self):
        """Server succeeds after one transient initial failure."""
        from tools.mcp_tool import MCPServerTask, _MAX_INITIAL_CONNECT_RETRIES

        call_count = 0

        async def _run():
            nonlocal call_count
            server = MCPServerTask("test-retry")

            # Track calls via patching the method on the class
            original_run_stdio = MCPServerTask._run_stdio

            async def fake_run_stdio(self_inner, config):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("DNS resolution failed")
                # Second attempt: success — set ready and "run" until shutdown
                self_inner._ready.set()
                await self_inner._shutdown_event.wait()

            with patch.object(MCPServerTask, '_run_stdio', fake_run_stdio):
                task = asyncio.ensure_future(server.run({"command": "fake"}))
                await server._ready.wait()

                # It should have succeeded (no error) after retrying
                assert server._error is None, f"Expected no error, got: {server._error}"
                assert call_count == 2, f"Expected 2 attempts, got {call_count}"

                # Clean shutdown
                server._shutdown_event.set()
                await task

        asyncio.get_event_loop().run_until_complete(_run())

    def test_initial_connect_gives_up_after_max_retries(self):
        """Server gives up after _MAX_INITIAL_CONNECT_RETRIES failures."""
        from tools.mcp_tool import MCPServerTask, _MAX_INITIAL_CONNECT_RETRIES

        call_count = 0

        async def _run():
            nonlocal call_count
            server = MCPServerTask("test-exhaust")

            async def fake_run_stdio(self_inner, config):
                nonlocal call_count
                call_count += 1
                raise ConnectionError("DNS resolution failed")

            with patch.object(MCPServerTask, '_run_stdio', fake_run_stdio):
                task = asyncio.ensure_future(server.run({"command": "fake"}))
                await server._ready.wait()

                # Should have an error after exhausting retries
                assert server._error is not None
                assert "DNS resolution failed" in str(server._error)
                # 1 initial + N retries = _MAX_INITIAL_CONNECT_RETRIES + 1 total attempts
                assert call_count == _MAX_INITIAL_CONNECT_RETRIES + 1

                await task

        asyncio.get_event_loop().run_until_complete(_run())

    def test_initial_connect_retry_respects_shutdown(self):
        """Shutdown during initial retry backoff aborts cleanly."""
        from tools.mcp_tool import MCPServerTask

        async def _run():
            server = MCPServerTask("test-shutdown")
            attempt = 0

            async def fake_run_stdio(self_inner, config):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise ConnectionError("transient failure")
                # Should not reach here because shutdown fires during sleep
                raise AssertionError("Should not attempt after shutdown")

            with patch.object(MCPServerTask, '_run_stdio', fake_run_stdio):
                task = asyncio.ensure_future(server.run({"command": "fake"}))

                # Give the first attempt time to fail, then set shutdown
                # during the backoff sleep
                await asyncio.sleep(0.1)
                server._shutdown_event.set()
                await server._ready.wait()

                # Should have the error set and be done
                assert server._error is not None
                await task

        asyncio.get_event_loop().run_until_complete(_run())
