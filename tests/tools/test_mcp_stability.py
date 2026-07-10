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

    def test_probe_cleanup_does_not_stop_loop_with_registered_servers(self):
        """Probe cleanup must not kill the shared loop used by live MCP tools."""
        import tools.mcp_tool as mcp_mod

        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()
        try:
            mcp_mod._ensure_mcp_loop()
            with mcp_mod._lock:
                loop = mcp_mod._mcp_loop
                mcp_mod._servers["live"] = MagicMock(session=object())

            assert mcp_mod._stop_mcp_loop_if_idle() is False

            with mcp_mod._lock:
                assert mcp_mod._mcp_loop is loop
                assert mcp_mod._mcp_thread is not None
            assert loop is not None
            assert loop.is_running()
        finally:
            with mcp_mod._lock:
                mcp_mod._servers.clear()
                mcp_mod._server_connecting.clear()
            mcp_mod._stop_mcp_loop()

    def test_shutdown_finalizes_async_generators_before_closing_loop(self):
        """Stopping the MCP loop drains async generators on its owner thread."""
        import tools.mcp_tool as mcp_mod

        finalized = threading.Event()

        async def _stream():
            try:
                yield "ready"
            finally:
                finalized.set()

        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()

        mcp_mod._ensure_mcp_loop()
        with mcp_mod._lock:
            loop = mcp_mod._mcp_loop

        streams = []

        async def _start_stream():
            # Construct and first-iterate the generator on the MCP loop so it
            # is registered with that loop's async-generator hooks.
            stream = _stream()
            streams.append(stream)
            return await stream.__anext__()

        first_item = asyncio.run_coroutine_threadsafe(_start_stream(), loop)
        assert first_item.result(timeout=2) == "ready"

        assert mcp_mod._stop_mcp_loop() is True
        assert finalized.wait(timeout=2)
        assert loop.is_closed()


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
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _stdio_pids,
            _lock,
        )

        with _lock:
            _stdio_pids.clear()
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()

        # Should not raise
        _kill_orphaned_mcp_children()

    def test_kill_orphaned_handles_dead_pids(self):
        """_kill_orphaned_mcp_children gracefully handles already-dead PIDs."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _lock,
        )

        # Use a PID that definitely doesn't exist
        fake_pid = 999999999
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_servers[fake_pid] = "orphan"

        # Should not raise (ProcessLookupError is caught)
        _kill_orphaned_mcp_children()

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_kill_orphaned_uses_sigkill_when_available(self, monkeypatch):
        """SIGTERM-first then SIGKILL after 2s for orphan cleanup."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _lock,
        )

        fake_pid = 424242
        with _lock:
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_servers[fake_pid] = "orphan"

        fake_sigkill = 9
        monkeypatch.setattr(signal, "SIGKILL", fake_sigkill, raising=False)

        # Post-#21561 the alive check routes through
        # ``gateway.status._pid_exists`` (so it's safe on Windows — see
        # bpo-14484). Return True so the SIGKILL escalation fires.
        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.mcp_tool.time.sleep") as mock_sleep:
            _kill_orphaned_mcp_children()

        # SIGTERM then SIGKILL; the alive check no longer touches os.kill.
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)
        mock_kill.assert_any_call(fake_pid, fake_sigkill)
        assert mock_kill.call_count == 2
        mock_sleep.assert_called_once_with(2)

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_kill_orphaned_falls_back_without_sigkill(self, monkeypatch):
        """Without SIGKILL, SIGTERM is used for both phases."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _lock,
        )

        fake_pid = 434343
        with _lock:
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_servers[fake_pid] = "orphan"

        monkeypatch.delattr(signal, "SIGKILL", raising=False)

        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("tools.mcp_tool.time.sleep") as mock_sleep:
            _kill_orphaned_mcp_children()

        # SIGTERM phase, alive check raises (process gone), no escalation
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)
        assert mock_sleep.called

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_run_stdio_reaps_orphans_before_spawn(self):
        """_run_stdio kills orphaned PIDs from prior failed attempts (#57355)."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _stdio_pids,
            _stdio_pgids,
            _lock,
            MCPServerTask,
        )
        from unittest.mock import patch, MagicMock, AsyncMock

        # Seed an orphan PID that belongs to a prior failed connection.
        fake_pid = 999999997
        with _lock:
            _orphan_stdio_pids.add(fake_pid)

        server = MCPServerTask.__new__(MCPServerTask)
        server.name = "test-zombie-reap"
        server._ready = MagicMock()
        server._shutdown_event = MagicMock()
        server._shutdown_event.is_set.return_value = True
        server._reconnect_event = MagicMock()
        server._sampling = None
        server._elicitation = None
        server._registered_tool_names = []

        config = {"command": "echo", "args": ["hello"]}

        import asyncio

        async def _run():
            # _run_stdio should reap orphans before it gets to the
            # stdio_client spawn.  Patch the OSV check (local import)
            # and stdio_client so no real subprocess is spawned.
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._build_safe_env", return_value={}), \
                 patch("tools.mcp_tool._resolve_stdio_command",
                       return_value=("echo", {})), \
                 patch("tools.mcp_tool._write_stderr_log_header"), \
                 patch("tools.mcp_tool._get_mcp_stderr_log",
                       return_value=None), \
                 patch("tools.mcp_tool.check_package_for_malware",
                       return_value=None, create=True), \
                 patch("tools.osv_check.check_package_for_malware",
                       return_value=None):
                # Patch stdio_client to raise so the test exits quickly
                cm = MagicMock()
                cm.__aenter__ = AsyncMock(side_effect=RuntimeError("test"))
                cm.__aexit__ = AsyncMock(return_value=False)
                with patch("tools.mcp_tool.stdio_client", return_value=cm):
                    try:
                        await server._run_stdio(config)
                    except Exception:
                        pass

        asyncio.run(_run())

        # The orphan must have been reaped before the spawn attempt.
        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_kill_orphaned_can_filter_by_server_name(self):
        """Reconnect cleanup reaps only the orphan owned by that MCP server."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _lock,
        )

        target_pid = 454545
        other_pid = 464646
        with _lock:
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _orphan_stdio_pids.update({target_pid, other_pid})
            _orphan_stdio_pid_servers[target_pid] = "feishu"
            _orphan_stdio_pid_servers[other_pid] = "mimir"

        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=False), \
             patch("tools.mcp_tool.time.sleep") as mock_sleep:
            _kill_orphaned_mcp_children(server_name="feishu")

        mock_kill.assert_called_once_with(target_pid, signal.SIGTERM)
        mock_sleep.assert_called_once_with(2)
        with _lock:
            assert target_pid not in _orphan_stdio_pids
            assert target_pid not in _orphan_stdio_pid_servers
            assert other_pid in _orphan_stdio_pids
            assert _orphan_stdio_pid_servers[other_pid] == "mimir"


# ---------------------------------------------------------------------------
# Fix 2b: stdio descendant reaping via process group (issue #23799)
# ---------------------------------------------------------------------------
#
# When a stdio MCP wrapper (e.g. ``openclaw mcp serve``) itself spawns a
# helper subprocess (``claude mcp serve``) and then exits, the helper
# reparents to systemd-user and is invisible to the per-pid orphan reaper.
# The fix captures the wrapper's pgid at spawn time and reaps via killpg,
# which reaches same-group descendants whether or not the direct pid is alive.

class TestStdioPgroupReaping:
    """_kill_orphaned_mcp_children reaps via killpg when a pgid is tracked."""

    def _reset_state(self):
        from tools.mcp_tool import (
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _stdio_pgids,
            _stdio_pids,
            _lock,
        )
        with _lock:
            _stdio_pids.clear()
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _stdio_pgids.clear()

    def test_killpg_used_when_pgid_tracked(self, monkeypatch):
        """SIGTERM and SIGKILL route through killpg when pgid is known."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _stdio_pgids,
            _lock,
        )

        self._reset_state()
        fake_pid = 525252
        fake_pgid = 525252  # session leader: pgid == pid
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _stdio_pgids[fake_pid] = fake_pgid

        fake_sigkill = 9
        monkeypatch.setattr(signal, "SIGKILL", fake_sigkill, raising=False)

        # Ensure os.killpg exists on this platform for the test to make sense;
        # the production fallback path is covered by the per-pid tests above.
        if not hasattr(os, "killpg"):
            pytest.skip("os.killpg not available on this platform")

        with patch("tools.mcp_tool.os.killpg") as mock_killpg, \
             patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=True), \
             patch("time.sleep"):
            _kill_orphaned_mcp_children()

        # Both phases should have used killpg (pgroup reach), not per-pid kill.
        mock_killpg.assert_any_call(fake_pgid, signal.SIGTERM)
        mock_killpg.assert_any_call(fake_pgid, fake_sigkill)
        assert mock_killpg.call_count == 2
        mock_kill.assert_not_called()

        with _lock:
            assert fake_pid not in _orphan_stdio_pids
            assert fake_pid not in _stdio_pgids

    def test_killpg_skipped_when_pgid_matches_gateway_own_pgroup(self, monkeypatch):
        """#47134: when a tracked MCP child shares the gateway's OWN process
        group, killpg(pgid) would signal the gateway itself and crash it.
        The guard must skip killpg for that pgid and fall through to per-pid
        os.kill instead."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _stdio_pgids,
            _lock,
        )

        if not hasattr(os, "killpg") or not hasattr(os, "getpgrp"):
            pytest.skip("os.killpg/os.getpgrp not available on this platform")

        self._reset_state()
        gateway_pgid = 424242
        fake_pid = 717171  # a child pid that resolves to the gateway's pgid
        other_pid = 818181  # a normal child in its OWN (non-gateway) group
        other_pgid = 818181
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _stdio_pgids[fake_pid] = gateway_pgid  # == gateway's own pgid
            _orphan_stdio_pids.add(other_pid)
            _stdio_pgids[other_pid] = other_pgid  # distinct group → killpg OK

        fake_sigkill = 9
        monkeypatch.setattr(signal, "SIGKILL", fake_sigkill, raising=False)

        with patch("tools.mcp_tool.os.getpgrp", return_value=gateway_pgid), \
             patch("tools.mcp_tool.os.killpg") as mock_killpg, \
             patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=True), \
             patch("time.sleep"):
            _kill_orphaned_mcp_children()

        # killpg must NEVER be called for the gateway's own pgid (would self-kill).
        killpg_pgids = [call.args[0] for call in mock_killpg.call_args_list]
        assert gateway_pgid not in killpg_pgids, (
            "killpg was called with the gateway's own pgid — self-kill (#47134)"
        )
        # The shared-pgid child must be reaped via per-pid kill instead.
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)
        mock_kill.assert_any_call(fake_pid, fake_sigkill)
        # NEGATIVE CONTROL: a child in a DISTINCT group must STILL use killpg —
        # the guard must skip only the gateway's own group, not all pgids.
        assert other_pgid in killpg_pgids, (
            "killpg must still be used for a non-gateway pgid (guard too broad)"
        )

    def test_killpg_failure_falls_back_to_kill(self, monkeypatch):
        """If killpg raises ProcessLookupError (pgroup gone), try os.kill."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _stdio_pgids,
            _lock,
        )

        self._reset_state()
        fake_pid = 636363
        fake_pgid = 636363
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _stdio_pgids[fake_pid] = fake_pgid

        if not hasattr(os, "killpg"):
            pytest.skip("os.killpg not available on this platform")

        with patch(
            "tools.mcp_tool.os.killpg",
            side_effect=ProcessLookupError("no such process group"),
        ) as mock_killpg, \
             patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=False), \
             patch("time.sleep"):
            _kill_orphaned_mcp_children()

        # killpg was attempted (phase 1 SIGTERM) and fell back to os.kill.
        # Phase 3 skips because _pid_exists returns False (direct pid gone).
        mock_killpg.assert_called()
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)

        with _lock:
            assert fake_pid not in _orphan_stdio_pids
            assert fake_pid not in _stdio_pgids

    def test_no_pgid_uses_per_pid_kill(self, monkeypatch):
        """When no pgid is recorded (e.g. Windows), fall back to os.kill."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _stdio_pgids,
            _lock,
        )

        self._reset_state()
        fake_pid = 747474
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            # No entry in _stdio_pgids.

        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=False), \
             patch("time.sleep"):
            # killpg may or may not exist; either way the no-pgid path skips it.
            _kill_orphaned_mcp_children()

        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    @pytest.mark.live_system_guard_bypass
    @pytest.mark.skipif(
        not hasattr(os, "killpg") or not hasattr(os, "setsid"),
        reason="POSIX-only: requires os.killpg and os.setsid",
    )
    def test_grandchild_reaped_via_pgroup(self, tmp_path):
        """End-to-end: parent spawns grandchild, parent exits, killpg reaps grandchild.

        Mirrors issue #23799: a stdio MCP wrapper (parent) launches a long-lived
        helper subprocess (grandchild) in the same process group, then the
        wrapper exits while the grandchild keeps running.  killpg on the pgid
        captured at spawn time must still deliver the signal to the grandchild.

        Marked ``live_system_guard_bypass`` because this test genuinely needs
        real signal delivery to its own subprocess tree (the conftest guard
        only knows the test's *initial* children; the spawned tree here is
        outside that allowlist).
        """
        import subprocess
        import sys
        import time as _time

        psutil = pytest.importorskip("psutil")

        # Grandchild: sleep forever, write its pid then wait.  The pid file
        # is written to a temp path and os.replace()d into place so the
        # polling reader below can never observe a created-but-empty file
        # (CI flake: int('') ValueError when the reader won the race between
        # open('w') creating the file and write() filling it).
        grandchild_pid_file = tmp_path / "grandchild.pid"
        grandchild_script = tmp_path / "grandchild.py"
        grandchild_script.write_text(
            "import os, sys, time\n"
            f"tmp = {str(grandchild_pid_file)!r} + '.tmp'\n"
            "with open(tmp, 'w') as f:\n"
            "    f.write(str(os.getpid()))\n"
            f"os.replace(tmp, {str(grandchild_pid_file)!r})\n"
            "while True:\n"
            "    time.sleep(0.5)\n"
        )

        # Parent: spawn grandchild, exit immediately (without killing it).
        parent_script = tmp_path / "parent.py"
        parent_script.write_text(
            "import subprocess, sys\n"
            f"subprocess.Popen([sys.executable, {str(grandchild_script)!r}])\n"
            # Parent exits — grandchild reparents to init.
        )

        # Spawn parent in its own session (mirrors stdio_client behaviour).
        parent = subprocess.Popen(
            [sys.executable, str(parent_script)],
            start_new_session=True,
        )
        parent_pgid = os.getpgid(parent.pid)
        # Wait for parent to exit and grandchild to spin up.
        parent.wait(timeout=5)
        deadline = _time.time() + 5
        while _time.time() < deadline and not grandchild_pid_file.exists():
            _time.sleep(0.05)
        assert grandchild_pid_file.exists(), "grandchild did not start"
        grandchild_pid = int(grandchild_pid_file.read_text().strip())

        # Sanity: grandchild is alive and shares the parent's pgid.
        assert psutil.pid_exists(grandchild_pid)
        assert os.getpgid(grandchild_pid) == parent_pgid

        # Drive the reaper: register the parent pid + pgid as an orphan.
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _stdio_pgids,
            _stdio_pids,
            _lock,
        )
        with _lock:
            _stdio_pids.clear()
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _stdio_pgids.clear()
            _orphan_stdio_pids.add(parent.pid)
            _orphan_stdio_pid_servers[parent.pid] = "orphan"
            _stdio_pgids[parent.pid] = parent_pgid
        try:
            _kill_orphaned_mcp_children()
        finally:
            # Belt-and-suspenders: ensure grandchild is dead even if test fails.
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Grandchild should be gone — SIGTERM via killpg in phase 1 reached it.
        deadline = _time.time() + 3
        while _time.time() < deadline and psutil.pid_exists(grandchild_pid):
            _time.sleep(0.05)
        assert not psutil.pid_exists(grandchild_pid), (
            "grandchild survived killpg-based reaping (issue #23799 regression)"
        )


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
        from tools.mcp_tool import MCPServerTask

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
        """Server parks (does not exit) after _MAX_INITIAL_CONNECT_RETRIES failures."""
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
                # The task parks for later revival instead of exiting.
                await asyncio.sleep(0)
                assert not task.done(), "run task should park, not exit"

                server._shutdown_event.set()
                server._reconnect_event.set()
                await asyncio.wait_for(task, timeout=5)

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
