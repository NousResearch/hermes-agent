"""Tests for ProcessRegistry.kill_process (split from test_process_registry.py)."""

import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.process_registry import ProcessRegistry
from tests.tools.test_process_registry import _make_session, _reset_systemd_scope_cache  # noqa: F401  (autouse fixture)


@pytest.fixture()
def registry():
    return ProcessRegistry()


# =========================================================================
# Kill process
# =========================================================================

class TestKillProcess:
    def test_kill_already_exited(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        result = registry.kill_process(s.id)
        assert result["status"] == "already_exited"


    def test_kill_detached_session_uses_host_pid(self, registry):
        s = _make_session(sid="proc_detached", command="sleep 999")
        s.pid = 424242
        s.detached = True
        registry._running[s.id] = s

        # PIDs the platform's kill primitive was actually aimed at. The
        # guarantee — a detached session is killed by its HOST pid — is
        # platform-neutral; only the primitive differs, so mock whichever one
        # ``_terminate_host_pid`` really reaches rather than asserting the
        # POSIX seam everywhere (that seam is never constructed on Windows,
        # which made this assertion unfalsifiable there).
        killed_pids = []
        # Post-#115490 kill_process verifies tree death after signalling: the
        # fake kill must actually kill, or the live fake reads as a survivor
        # and the kill correctly reports incomplete.
        kill_state = {"alive": True}

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
            def children(self, recursive=False):
                return []
            def terminate(self):
                killed_pids.append(self.pid)
                kill_state["alive"] = False

        def _fake_taskkill(argv, **_kwargs):
            # ["taskkill", "/PID", "<pid>", "/T", "/F"]
            killed_pids.append(int(argv[argv.index("/PID") + 1]))
            kill_state["alive"] = False
            return subprocess.CompletedProcess(argv, 0, "", "")

        import psutil as _psutil

        if sys.platform == "win32":
            # Windows tree-kill shells out to ``taskkill /PID <pid> /T /F``
            # and returns before psutil is imported (_terminate_host_pid).
            kill_seam = patch("tools.process_registry.subprocess.run", _fake_taskkill)
        else:
            kill_seam = patch.object(
                _psutil, "Process", side_effect=lambda pid: FakeProcess(pid)
            )

        try:
            # Post-#21561: liveness probe routes through
            # ``ProcessRegistry._is_host_pid_alive`` (→
            # ``gateway.status._pid_exists``), and the actual kill on POSIX
            # routes through ``psutil.Process(pid).terminate()``. Neither
            # touches ``os.kill`` directly. Mock both seams.  Disable the
            # SIGKILL-escalation step (grace=0) so it doesn't call
            # ``psutil.wait_procs`` on the FakeProcess.
            with patch("gateway.status._pid_exists", side_effect=lambda pid: kill_state["alive"]), \
                 patch.object(ProcessRegistry, "_daemon_term_grace_seconds",
                              staticmethod(lambda: 0.0)), \
                 kill_seam:
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert killed_pids == [424242]
        finally:
            registry._running.pop(s.id, None)

    def test_kill_receipt_rewritten_when_reader_finalises_first(self, registry):
        """A kill racing the reader thread must not persist as a plain exit.

        The signal path blocks for the SIGKILL grace window, during which the
        reader thread can observe the exit and finalise the session first. The
        durable receipt from that first save says ``exited``; the kill result
        returned to the caller says ``killed``. The second save must rewrite
        the receipt so the persisted record matches what the caller was told.
        """
        s = _make_session(sid="proc_kill_race", command="sleep 999")
        s.pid = 424243
        s.detached = True
        registry._running[s.id] = s

        def reader_wins_during_signal(pid, start=None):
            # The reader thread observes the SIGTERMed exit while the signal
            # path is still inside its grace window and finalises first.
            registry._finish_exited(s, 0)

        saved = []

        def record_save(session):
            saved.append(
                (session.completion_reason, session.termination_source, session.exit_code)
            )

        try:
            host_guard = patch.object(
                ProcessRegistry, "_host_pid_is_ours", return_value=True
            )
            term_patch = patch.object(
                ProcessRegistry,
                "_terminate_host_pid",
                side_effect=reader_wins_during_signal,
            )
            saver = patch(
                "tools.process_registry.save_completed_result", side_effect=record_save
            )
            with host_guard, term_patch, saver:
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert result["completion_reason"] == "killed"
            assert result["termination_source"] == "process.kill"
            # First save: the reader won the race and persisted a plain exit.
            assert saved[0] == ("exited", "", 0)
            # Second save: the receipt rewritten with the kill outcome.
            assert saved[-1] == ("killed", "process.kill", -15)
        finally:
            registry._running.pop(s.id, None)
            registry._finished.pop(s.id, None)
