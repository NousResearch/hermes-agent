"""Tests for #115184: a protocol-level agent-browser failure — exit 101 against a
stale local session, or empty/non-JSON output from a dead daemon — must recycle the
cached session record and retry once, instead of handing the same poisoned record to
every later call until manual recycling.

Page-level JSON errors (the backend answered, the page said no) and non-local sessions
(cloud/CDP/real-profile/Lightpanda) never recycle: cloud recovery has its own owner and
the Lightpanda engine has its Chrome fallback.

All tests use a fake daemon layer (scripted Popen + monkeypatched probes) — no real
agent-browser or Chromium is spawned.
"""

import json
import os
import subprocess

import pytest

import tools.browser_tool as bt
from tools import browser_tool_session as bt_session
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_install as bt_install
from tools import browser_tool_lifecycle as bt_lifecycle

TASK = "poison-task"


@pytest.fixture(autouse=True)
def _reset_browser_state():
    def _clear():
        bt._active_sessions.clear()
        bt._session_last_activity.clear()
        bt._last_active_session_key.clear()
        bt._suspect_browser_sessions.clear()

    _clear()
    yield
    _clear()


def _local_session(name="h_stale"):
    return {"session_name": name, "bb_session_id": None, "cdp_url": None,
            "features": {"local": True}}


def _scripted_popen(script):
    """Popen fake playing one script entry per spawned command: {rc, stdout, stderr}."""
    argv_log = []

    class FakePopen:
        def __init__(self, argv, stdout=None, stderr=None, **_kwargs):
            self.argv = list(argv)
            argv_log.append(self.argv)
            self._entry = script[min(len(argv_log), len(script)) - 1]
            if self._entry.get("stdout"):
                os.write(stdout, self._entry["stdout"])
            if self._entry.get("stderr"):
                os.write(stderr, self._entry["stderr"])

        def wait(self, timeout=None):
            self.returncode = self._entry["rc"]
            return self.returncode

    return FakePopen, argv_log


def _install_command_stubs(monkeypatch, tmp_path, popen_cls):
    """Common _run_browser_command environment with a fake daemon layer."""
    monkeypatch.setattr(bt_install, "_find_agent_browser", lambda: "agent-browser")
    monkeypatch.setattr("tools.browser_tool_install._requires_real_termux_browser_install", lambda _cmd: False)
    monkeypatch.setattr("tools.browser_tool_install._chromium_installed", lambda: True)
    monkeypatch.setattr("tools.browser_tool_lifecycle._start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", lambda _tid: None)
    monkeypatch.setattr("tools.browser_tool_cdp._stop_cdp_supervisor", lambda _tid: None)
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr("tools.browser_tool_lifecycle._write_owner_pid", lambda *_args: None)
    monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
    monkeypatch.setattr("tools.browser_tool_install._merge_browser_path", lambda value: value)
    monkeypatch.setattr("tools.browser_tool_cloud._get_browser_engine", lambda: "auto")
    monkeypatch.setattr("tools.browser_tool_cloud._is_headed_mode", lambda: False)
    monkeypatch.setattr(subprocess, "Popen", popen_cls)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    # Fresh-session plumbing for the retry attempt (pure local Chromium).
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr("tools.browser_tool_real_profile._real_profile_cdp", lambda: (None, None))
    monkeypatch.setattr(bt, "_is_browser_use_cli_mode", lambda: False)
    monkeypatch.setattr("tools.browser_tool_lightpanda_fallback._using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(bt_lifecycle, "_session_has_expired", lambda *_a, **_k: False)


_SUCCESS = json.dumps({"success": True, "data": {"ok": 1}}).encode()


class TestDeadDaemonRecyclesAndRetriesOnce:
    def test_exit_101_dead_daemon_retries_on_fresh_session(self, monkeypatch, tmp_path):
        """No pid file → daemon treated as dead → record evicted, one retry on a fresh session."""
        bt._active_sessions[TASK] = _local_session("h_stale")
        popen, argv_log = _scripted_popen([
            {"rc": 101, "stdout": b"", "stderr": b"daemon session error"},
            {"rc": 0, "stdout": _SUCCESS},
        ])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "click", ["@e1"], timeout=5)

        assert result["success"] is True
        assert len(argv_log) == 2  # exactly one retry
        assert "h_stale" in argv_log[0]
        retry_session = argv_log[1][argv_log[1].index("--session") + 1]
        assert retry_session != "h_stale" and retry_session.startswith("h_")
        # The poisoned record was evicted: the cached session is the retry's fresh one.
        assert bt._active_sessions[TASK]["session_name"] == retry_session
        assert TASK not in bt._suspect_browser_sessions  # flag must not poison the fresh session

    def test_empty_output_rc0_dead_daemon_retries_once(self, monkeypatch, tmp_path):
        """Empty stdout with rc=0 on a non-EMPTY_OK idempotent command (snapshot) = stale
        daemon signature: recycle and retry once."""
        bt._active_sessions[TASK] = _local_session("h_stale2")
        popen, argv_log = _scripted_popen([
            {"rc": 0, "stdout": b"", "stderr": b""},
            {"rc": 0, "stdout": _SUCCESS},
        ])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "snapshot", ["-c"], timeout=5)

        assert result["success"] is True
        assert len(argv_log) == 2

    def test_empty_output_rc0_mutating_command_not_retried(self, monkeypatch, tmp_path):
        """REQUIRED negative probe: rc=0 with no output can also mean the command already
        ran and the daemon died before flushing stdout — re-issuing a mutating command
        (click on a submit control, fill, press) would apply it twice. Non-idempotent
        commands keep main's single-attempt behaviour."""
        session_info = _local_session("h_maybe")
        bt._active_sessions[TASK] = session_info
        popen, argv_log = _scripted_popen([{"rc": 0, "stdout": b"", "stderr": b""}])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "click", ["@e1"], timeout=5)

        assert result["success"] is False
        assert "returned no output" in result["error"]
        assert len(argv_log) == 1  # never re-run a possibly-landed mutating command
        assert bt._active_sessions[TASK] is session_info
        assert TASK not in bt._suspect_browser_sessions

    def test_non_json_output_rc0_mutating_command_not_retried(self, monkeypatch, tmp_path):
        """Same double-execution guard for the non-JSON arm: rc=0 means the CLI considers
        the command finished, so a mutating command is never replayed on that evidence."""
        session_info = _local_session("h_maybe2")
        bt._active_sessions[TASK] = session_info
        popen, argv_log = _scripted_popen([{"rc": 0, "stdout": b"daemon: panic", "stderr": b""}])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "click", ["@e1"], timeout=5)

        assert result["success"] is False
        assert "Non-JSON output" in result["error"]
        assert len(argv_log) == 1
        assert bt._active_sessions[TASK] is session_info
        assert TASK not in bt._suspect_browser_sessions

    def test_second_failure_is_returned_not_looped(self, monkeypatch, tmp_path):
        """Retry budget is one: a second backend failure is returned to the caller."""
        bt._active_sessions[TASK] = _local_session("h_stale3")
        popen, argv_log = _scripted_popen([
            {"rc": 101, "stdout": b"", "stderr": b"first"},
            {"rc": 101, "stdout": b"", "stderr": b"second"},
        ])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "click", ["@e1"], timeout=5)

        assert result["success"] is False
        assert result["returncode"] == 101
        assert len(argv_log) == 2


class TestAliveDaemonMarksSuspect:
    def test_exit_101_alive_daemon_recycles_at_next_use(self, monkeypatch, tmp_path):
        """Alive/responsive daemon: only the command died — suspect flag set, the retry's
        _get_session_info consumes it (clean teardown) and runs on a fresh session."""
        bt._active_sessions[TASK] = _local_session("h_alive")
        popen, argv_log = _scripted_popen([
            {"rc": 101, "stdout": b"", "stderr": b"session error"},
            {"rc": 0, "stdout": _SUCCESS},
        ])
        _install_command_stubs(monkeypatch, tmp_path, popen)
        monkeypatch.setattr(bt_session, "_read_browser_daemon_pid", lambda *_a: 4321)
        monkeypatch.setattr(bt_lifecycle, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(bt_lifecycle, "_verify_reapable_browser_daemon", lambda *_a: True)
        monkeypatch.setattr(bt_session, "_browser_daemon_responsive", lambda *_a, **_k: True)
        marks = []
        original_mark = bt._BrowserSessionBackend.mark_suspect

        def counting_mark(self, reason):
            marks.append(reason)
            original_mark(self, reason)

        monkeypatch.setattr(bt._BrowserSessionBackend, "mark_suspect", counting_mark)
        cleanups = []

        def fake_cleanup(task_id):
            cleanups.append(task_id)
            with bt._cleanup_lock:
                bt._active_sessions.pop(task_id, None)
                bt._session_last_activity.pop(task_id, None)

        monkeypatch.setattr(bt_lifecycle, "_cleanup_single_browser_session", fake_cleanup)

        result = bt_session._run_browser_command(TASK, "click", ["@e1"], timeout=5)

        assert result["success"] is True
        assert len(argv_log) == 2
        # The suspect flag was set before the retry and consumed by it (exactly one recycle).
        assert len(marks) == 1 and "poisoned" in marks[0]
        assert cleanups == [TASK]
        assert TASK not in bt._suspect_browser_sessions


class TestNoRecycleGuards:
    def test_page_level_json_error_never_retries(self, monkeypatch, tmp_path):
        """REQUIRED negative probe: a parsed page-level failure (no returncode) must not
        recycle or retry — the backend is healthy, the page said no."""
        session_info = _local_session("h_healthy")
        bt._active_sessions[TASK] = session_info
        popen, argv_log = _scripted_popen([
            {"rc": 0, "stdout": json.dumps({"success": False, "error": "element @e99 not found"}).encode()},
        ])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "click", ["@e99"], timeout=5)

        assert result["success"] is False
        assert len(argv_log) == 1
        assert bt._active_sessions[TASK] is session_info  # cached session untouched
        assert TASK not in bt._suspect_browser_sessions

    def test_cloud_session_exit_101_keeps_single_attempt(self, monkeypatch, tmp_path):
        """Cloud/CDP sessions (bb_session_id/cdp_url) are outside the local-backend contract:
        behavior on main is preserved (no retry, no recycle)."""
        cloud_session = {"session_name": "bb_1", "bb_session_id": "s-1",
                         "cdp_url": "ws://cdp.example", "features": {}}
        bt._active_sessions[TASK] = cloud_session
        popen, argv_log = _scripted_popen([
            {"rc": 101, "stdout": b"", "stderr": b"remote failure"},
        ])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "click", ["@e1"], timeout=5)

        assert result["success"] is False
        assert len(argv_log) == 1
        assert bt._active_sessions[TASK] is cloud_session
        assert TASK not in bt._suspect_browser_sessions

    def test_close_empty_rc0_is_success_not_recoverable(self, monkeypatch, tmp_path):
        """close is in _EMPTY_OK_COMMANDS: empty rc=0 output is success, never a retry."""
        session_info = _local_session("h_closing")
        bt._active_sessions[TASK] = session_info
        popen, argv_log = _scripted_popen([{"rc": 0, "stdout": b"", "stderr": b""}])
        _install_command_stubs(monkeypatch, tmp_path, popen)

        result = bt_session._run_browser_command(TASK, "close", timeout=5)

        assert result["success"] is True
        assert len(argv_log) == 1
        assert bt._active_sessions[TASK] is session_info
