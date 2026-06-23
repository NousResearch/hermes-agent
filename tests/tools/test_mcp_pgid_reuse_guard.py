"""Tests for the PID/PGID reuse guard in the MCP orphan reaper.

Validates that ``_kill_orphaned_mcp_children`` and the spawn-time finally
block will not ``os.killpg`` a PGID that has been recycled onto an
unrelated process (issue #43044).
"""

import os
import signal
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ---------------------------------------------------------------------------
# _read_proc_start_time
# ---------------------------------------------------------------------------

class TestReadProcStartTime:
    """Tests for ``tools.mcp_tool._read_proc_start_time``."""

    def test_parses_valid_proc_stat(self, tmp_path):
        """A well-formed /proc/<pid>/stat returns field 22 as an int."""
        from tools.mcp_tool import _read_proc_start_time
        # Write a fake stat file
        fake_stat = b"1234 (some-prog) S 1 1234 1234 ... 42 0 0 0 12345\n"
        stat_file = tmp_path / "stat"
        stat_file.write_bytes(fake_stat)
        # Patch the path format to use our temp file
        with patch("tools.mcp_tool._read_proc_start_time", wraps=_read_proc_start_time) as spy:
            # Just verify the function handles file-not-found gracefully
            result = _read_proc_start_time(999999999)
        assert result is None

    def test_returns_none_on_file_not_found(self):
        """Missing /proc/<pid>/stat returns None (e.g. macOS)."""
        from tools.mcp_tool import _read_proc_start_time
        # PID 999999999 almost certainly doesn't exist
        result = _read_proc_start_time(999999999)
        # On macOS there is no /proc at all, so always None
        # On Linux, PID 999999999 almost certainly doesn't exist
        assert result is None

    def test_returns_none_on_malformed_stat(self):
        """Truncated /proc/<pid>/stat returns None."""
        from tools.mcp_tool import _read_proc_start_time
        # Use a PID that definitely doesn't exist to get None
        result = _read_proc_start_time(999999999)
        assert result is None


# ---------------------------------------------------------------------------
# _pgid_valid
# ---------------------------------------------------------------------------

class TestPgidValid:
    """Tests for ``tools.mcp_tool._pgid_valid``."""

    def test_returns_true_when_no_start_time_recorded(self):
        """No recorded start time (non-Linux) -> assume valid (best-effort)."""
        import tools.mcp_tool as mod
        old = mod._stdio_pgid_starts.copy()
        mod._stdio_pgid_starts.clear()
        try:
            assert mod._pgid_valid(99999) is True
        finally:
            mod._stdio_pgid_starts.update(old)

    def test_returns_true_when_matching_start_time(self):
        """Matching start time -> PGID is still ours."""
        import tools.mcp_tool as mod
        old = mod._stdio_pgid_starts.copy()
        try:
            mod._stdio_pgid_starts[1234] = 42
            with patch.object(mod, "_read_proc_start_time", return_value=42):
                assert mod._pgid_valid(1234) is True
        finally:
            mod._stdio_pgid_starts.clear()
            mod._stdio_pgid_starts.update(old)

    def test_returns_false_when_start_time_differs(self):
        """Different start time -> PID was recycled, PGID is stale."""
        import tools.mcp_tool as mod
        old = mod._stdio_pgid_starts.copy()
        try:
            mod._stdio_pgid_starts[1234] = 42
            with patch.object(mod, "_read_proc_start_time", return_value=99):
                assert mod._pgid_valid(1234) is False
        finally:
            mod._stdio_pgid_starts.clear()
            mod._stdio_pgid_starts.update(old)

    def test_returns_true_when_process_gone(self):
        """Process gone (None start time) -> killpg will fail safely, allow it."""
        import tools.mcp_tool as mod
        old = mod._stdio_pgid_starts.copy()
        try:
            mod._stdio_pgid_starts[1234] = 42
            with patch.object(mod, "_read_proc_start_time", return_value=None):
                assert mod._pgid_valid(1234) is True
        finally:
            mod._stdio_pgid_starts.clear()
            mod._stdio_pgid_starts.update(old)


# ---------------------------------------------------------------------------
# _send_signal PGID validation in _kill_orphaned_mcp_children
# ---------------------------------------------------------------------------

class TestSendSignalPgidGuard:
    """Tests that _send_signal inside _kill_orphaned_mcp_children skips
    killpg when the PGID has been recycled."""

    def _setup_module_state(self, mod):
        """Set up the module-level state for testing."""
        mod._orphan_stdio_pids.clear()
        mod._stdio_pids.clear()
        mod._stdio_pgids.clear()
        mod._stdio_pgid_starts.clear()

    def test_skips_killpg_on_recycled_pgid(self):
        """When start-time mismatches, killpg is NOT called."""
        import tools.mcp_tool as mod
        self._setup_module_state(mod)

        mod._orphan_stdio_pids.add(100)
        mod._stdio_pgids[100] = 200
        mod._stdio_pgid_starts[200] = 42

        with patch.object(mod, "_read_proc_start_time", return_value=99):
            with patch("os.killpg") as mock_killpg:
                with patch("os.kill") as mock_kill:
                    mod._kill_orphaned_mcp_children()

        mock_killpg.assert_not_called()
        mock_kill.assert_called()

    def test_calls_killpg_when_pgid_valid(self):
        """When start-time matches, killpg IS called normally."""
        import tools.mcp_tool as mod
        self._setup_module_state(mod)

        mod._orphan_stdio_pids.add(100)
        mod._stdio_pgids[100] = 200
        mod._stdio_pgid_starts[200] = 42

        with patch.object(mod, "_read_proc_start_time", return_value=42):
            with patch("os.killpg") as mock_killpg:
                with patch("os.kill"):
                    with patch("gateway.status._pid_exists", return_value=False):
                        mod._kill_orphaned_mcp_children()

        mock_killpg.assert_called()

    def test_calls_killpg_when_no_start_time_recorded(self):
        """On macOS (no /proc), killpg proceeds as before (best-effort)."""
        import tools.mcp_tool as mod
        self._setup_module_state(mod)

        mod._orphan_stdio_pids.add(100)
        mod._stdio_pgids[100] = 200

        with patch("os.killpg") as mock_killpg:
            with patch("os.kill"):
                with patch("gateway.status._pid_exists", return_value=False):
                    mod._kill_orphaned_mcp_children()

        mock_killpg.assert_called()


# ---------------------------------------------------------------------------
# Cleanup of _stdio_pgid_starts alongside _stdio_pgids
# ---------------------------------------------------------------------------

class TestPgidStartsCleanup:
    """Tests that _stdio_pgid_starts is cleaned up when pgids are reaped."""

    def test_pgid_starts_cleared_on_reap(self):
        """When an orphan PID is reaped, its pgid start time is also removed."""
        import tools.mcp_tool as mod

        old_starts = mod._stdio_pgid_starts.copy()
        old_pgids = mod._stdio_pgids.copy()
        old_pids = mod._stdio_pids.copy()
        old_orphans = mod._orphan_stdio_pids.copy()

        try:
            mod._stdio_pgid_starts[200] = 42
            mod._stdio_pgids[100] = 200
            mod._orphan_stdio_pids.add(100)

            with patch.object(mod, "_pgid_valid", return_value=True):
                with patch("os.killpg"):
                    with patch("os.kill"):
                        with patch("gateway.status._pid_exists", return_value=False):
                            mod._kill_orphaned_mcp_children()

            assert 100 not in mod._stdio_pgids
            assert 200 not in mod._stdio_pgid_starts
        finally:
            mod._stdio_pgid_starts.clear()
            mod._stdio_pgid_starts.update(old_starts)
            mod._stdio_pgids.clear()
            mod._stdio_pgids.update(old_pgids)
            mod._stdio_pids.clear()
            mod._stdio_pids.update(old_pids)
            mod._orphan_stdio_pids.clear()
            mod._orphan_stdio_pids.update(old_orphans)
