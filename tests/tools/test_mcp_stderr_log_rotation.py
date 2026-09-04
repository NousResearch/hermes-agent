"""Tests for the mcp-stderr.log copy-rotate bound (open-time rotation).

A misbehaving stdio server (fast crash-loop, e.g. launched without its
credentials) can push the shared stderr log to tens of MB with no upper
bound. Rotation must happen exactly once per process, inside the open
lock, before any fd is handed to a subprocess — and must copy+truncate
(not rename) so append-mode writers in other processes keep working.
"""

import threading
from unittest.mock import patch

import pytest

import tools.mcp_tool as mcp_tool


def _reset_module_log_state():
    mcp_tool._mcp_stderr_log_fh = None


class TestRotateMcpStderrLog:
    """_rotate_mcp_stderr_log behaviour on disk."""

    def setup_method(self):
        _reset_module_log_state()

    def teardown_method(self):
        _reset_module_log_state()

    def test_no_rotation_under_threshold(self, tmp_path):
        log = tmp_path / "mcp-stderr.log"
        log.write_text("small", encoding="utf-8")
        mcp_tool._rotate_mcp_stderr_log(log)
        assert log.read_text(encoding="utf-8") == "small"
        assert not (tmp_path / "mcp-stderr.log.1").exists()

    def test_rotation_moves_old_backups_and_truncates(self, tmp_path):
        log = tmp_path / "mcp-stderr.log"
        log.write_text("x" * (mcp_tool._MCP_STDERR_LOG_MAX_BYTES + 1),
                       encoding="utf-8")
        stale2 = tmp_path / "mcp-stderr.log.2"
        stale2.write_text("old-two", encoding="utf-8")
        mcp_tool._rotate_mcp_stderr_log(log)
        assert log.read_text(encoding="utf-8") == ""
        assert (tmp_path / "mcp-stderr.log.1").read_text(encoding="utf-8").startswith("x")
        # .2 (oldest) was pushed out by the backup-count bound of 2
        assert not (tmp_path / "mcp-stderr.log.3").exists()

    def test_rotation_failure_is_silent(self, tmp_path):
        log = tmp_path / "does-not-exist-dir" / "mcp-stderr.log"
        # stat() on a missing parent raises inside the try — must not
        # propagate (best-effort contract).
        mcp_tool._rotate_mcp_stderr_log(log)


class TestGetMcpStderrLogRotationHook:
    """_get_mcp_stderr_log invokes rotation exactly once per process."""

    def setup_method(self):
        _reset_module_log_state()

    def teardown_method(self):
        _reset_module_log_state()

    def test_rotation_called_once_per_process(self, tmp_path):
        calls = []
        real_rotate = mcp_tool._rotate_mcp_stderr_log

        def counting_rotate(path):
            calls.append(path)
            real_rotate(path)

        with patch.object(mcp_tool, "_rotate_mcp_stderr_log", side_effect=counting_rotate), \
             patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            fh1 = mcp_tool._get_mcp_stderr_log()
            fh2 = mcp_tool._get_mcp_stderr_log()
            assert fh1 is fh2
        assert len(calls) == 1
        assert calls[0].name == "mcp-stderr.log"
        fh1.close()

    def test_shared_handle_across_threads(self, tmp_path):
        """Concurrent first-open from several threads yields one handle."""
        results = []
        barrier = threading.Barrier(4)

        def opener():
            barrier.wait()
            results.append(mcp_tool._get_mcp_stderr_log())

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            threads = [threading.Thread(target=opener) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert len(results) == 4
        assert all(fh is results[0] for fh in results)
        results[0].close()
