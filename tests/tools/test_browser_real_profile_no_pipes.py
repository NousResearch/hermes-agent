"""Regression tests for the #106244 pipe-EOF wedge in the real-profile path.

The two ``agent-browser`` CLI call sites in ``browser_tool_real_profile.py``
must redirect stdout/stderr to files in the session socket dir — never
``capture_output=True`` (pipes). The CLI spawns the browser-use daemon, the
daemon inherits the pipe write end, and after ``timeout`` kills the CLI the
no-deadline post-kill ``communicate()`` wait blocks on pipe EOF forever while
the daemon lives: the caller (holding ``_real_profile_cdp_lock``) wedges with
no deadline, and every queued ``browser_exec`` call times out. Same pattern
and same fix as ``browser_tool_session`` (#106244, fixed there in #106300).
"""
import os
from unittest.mock import patch

import pytest
from tools import browser_tool_install as bt_install
from tools import browser_tool_real_profile as bt_real_profile
from tools import browser_tool_session as bt_session




def _completed(argv):
    import subprocess
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


class TestRealProfileSessionCmdNoPipes:

    def test_session_cmd_redirects_output_to_files_not_pipes(self, tmp_path, monkeypatch):
        """stdout/stderr must be file handles in the socket dir, never capture_output pipes."""
        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            captured["kw"] = kw
            return _completed(argv)

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
        monkeypatch.setattr(bt_session, "_prepare_session_socket_dir", lambda name: str(tmp_path))
        with patch.object(bt_real_profile.subprocess, "run", side_effect=fake_run):
            proc = bt_real_profile._agent_browser_session_cmd("s1", "get", "cdp-url", log_label="t")
        kw = captured["kw"]
        # The invariant: real file objects (fd-backed), not pipes.
        assert not kw.get("capture_output"), "capture_output pipes wedge on daemon-held write ends (#106244)"
        assert hasattr(kw["stdout"], "fileno") and hasattr(kw["stderr"], "fileno")
        assert str(tmp_path) in getattr(kw["stdout"], "name", "")
        assert str(tmp_path) in getattr(kw["stderr"], "name", "")
        assert kw.get("timeout") == 15, "the 15s deadline must stay bounded"
        assert proc is not None

    def test_session_cmd_reads_back_and_unlinks_output_files(self, tmp_path, monkeypatch):
        """After a successful run the redirected output is read back and the files removed."""

        def fake_run(argv, **kw):
            kw["stdout"].write(b"ws://127.0.0.1:41000/devtools/browser/x\n")
            kw["stderr"].write(b"")
            return _completed(argv)

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
        monkeypatch.setattr(bt_session, "_prepare_session_socket_dir", lambda name: str(tmp_path))
        with patch.object(bt_real_profile.subprocess, "run", side_effect=fake_run):
            proc = bt_real_profile._agent_browser_session_cmd("s1", "get", "cdp-url", log_label="t")
        assert "ws://127.0.0.1:41000" in (proc.stdout or "")
        leftovers = [p for p in os.listdir(tmp_path) if p.startswith("_stdout_rp") or p.startswith("_stderr_rp")]
        assert leftovers == [], f"output files not cleaned up: {leftovers}"

    def test_session_cmd_timeout_returns_none_instead_of_wedging(self, tmp_path, monkeypatch):
        """TimeoutExpired must surface as None promptly — the caller never blocks past it."""
        import subprocess as real_subprocess

        def fake_run(argv, **kw):
            raise real_subprocess.TimeoutExpired(cmd="agent-browser", timeout=15)

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
        monkeypatch.setattr(bt_session, "_prepare_session_socket_dir", lambda name: str(tmp_path))
        with patch.object(bt_real_profile.subprocess, "run", side_effect=fake_run):
            proc = bt_real_profile._agent_browser_session_cmd("s1", "get", "cdp-url", log_label="t")
        assert proc is None

    def test_session_cmd_timeout_cleans_up_output_files(self, tmp_path, monkeypatch):
        """TimeoutExpired must read back + unlink the redirect files before returning —
        fixed names plus a surviving daemon writer would otherwise leave the next call
        reading the killed run's output (stale CDP ports parsed by _agent_browser_get_cdp)."""
        import subprocess as real_subprocess

        def fake_run(argv, **kw):
            # Simulate the killed CLI's partial output sitting in the redirect file.
            with open(kw["stdout"].name, "wb") as f:
                f.write(b"ws://127.0.0.1:9999/partial-from-killed-run\n")
            raise real_subprocess.TimeoutExpired(cmd="agent-browser", timeout=15)

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
        monkeypatch.setattr(bt_session, "_prepare_session_socket_dir", lambda name: str(tmp_path))
        with patch.object(bt_real_profile.subprocess, "run", side_effect=fake_run):
            proc = bt_real_profile._agent_browser_session_cmd("s1", "get", "cdp-url", log_label="t")
        assert proc is None
        leftovers = [p for p in os.listdir(tmp_path) if p.startswith("_stdout_rp") or p.startswith("_stderr_rp")]
        assert leftovers == [], f"redirect files leaked on the timeout path: {leftovers}"

    def test_session_cmd_error_cleans_up_output_files(self, tmp_path, monkeypatch):
        """SubprocessError/OSError must unlink the redirect files too — every exit path cleans up."""
        import subprocess as real_subprocess

        def fake_run(argv, **kw):
            raise real_subprocess.SubprocessError("spawn failed")

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
        monkeypatch.setattr(bt_session, "_prepare_session_socket_dir", lambda name: str(tmp_path))
        with patch.object(bt_real_profile.subprocess, "run", side_effect=fake_run):
            proc = bt_real_profile._agent_browser_session_cmd("s1", "get", "cdp-url", log_label="t")
        assert proc is None
        leftovers = [p for p in os.listdir(tmp_path) if p.startswith("_stdout_rp") or p.startswith("_stderr_rp")]
        assert leftovers == [], f"redirect files leaked on the error path: {leftovers}"
