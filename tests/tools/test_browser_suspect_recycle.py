"""Tests for #85125 Phase 3b (browser) + 4c: suspect-session recycle after a
command timeout (#72205, salvaging #72206) and daemon tree-kill on the wedged
path (#68139).

All tests use a fake daemon layer (mocked Popen + monkeypatched probes) —
no real agent-browser or Chromium is spawned.
"""

import json
import os
import subprocess
from unittest.mock import Mock

import pytest

import tools.browser_tool as bt

TASK = "suspect-task"


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


def _local_session(name="stuck-session"):
    return {"session_name": name, "bb_session_id": None, "cdp_url": None}


def _install_command_stubs(monkeypatch, tmp_path, process):
    """Common _run_browser_command environment with a fake daemon layer."""
    monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
    monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
    monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
    monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt, "_ensure_cdp_supervisor", lambda _tid: None)
    monkeypatch.setattr(bt, "_stop_cdp_supervisor", lambda _tid: None)
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
    monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
    monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
    monkeypatch.setattr(bt, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(bt, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)


class TestTimeoutMarksSuspect:
    def test_timeout_with_alive_daemon_marks_suspect_exactly_once(
        self, monkeypatch, tmp_path
    ):
        """Alive-daemon branch: session stays cached, flagged suspect once."""
        session_info = _local_session()
        bt._active_sessions[TASK] = session_info

        process = Mock()
        process.returncode = -9
        process.wait.side_effect = [subprocess.TimeoutExpired("agent-browser", 1), -9]
        _install_command_stubs(monkeypatch, tmp_path, process)

        # Daemon alive + responsive → recycle-at-next-use branch, no kill.
        monkeypatch.setattr(bt, "_read_browser_daemon_pid", lambda *_a: 4321)
        monkeypatch.setattr(bt, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(bt, "_verify_reapable_browser_daemon", lambda *_a: True)
        monkeypatch.setattr(bt, "_browser_daemon_responsive", lambda *_a, **_k: True)
        kills = []
        monkeypatch.setattr("agent.deadline.kill_process_tree", lambda pid, **_k: kills.append(pid))

        marks = []
        original_mark = bt._BrowserSessionBackend.mark_suspect

        def counting_mark(self, reason):
            marks.append(reason)
            original_mark(self, reason)

        monkeypatch.setattr(bt._BrowserSessionBackend, "mark_suspect", counting_mark)

        result = bt._run_browser_command(TASK, "click", ["@e1"], timeout=1)

        assert result["success"] is False
        assert len(marks) == 1  # marked suspect exactly once
        assert bt._suspect_browser_sessions == {
            TASK: "browser command timed out; session may be poisoned"
        }
        # Alive branch: session stays cached for the next-use recycle...
        assert bt._active_sessions[TASK] is session_info
        # ...and the daemon is NOT tree-killed.
        assert kills == []


class TestNextUseRecycles:
    def test_next_call_after_suspect_recycles_then_succeeds(self, monkeypatch):
        stale = _local_session("stale-session")
        bt._active_sessions[TASK] = stale
        bt._suspect_browser_sessions[TASK] = "browser command timed out"

        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(bt, "_ensure_cdp_supervisor", lambda _tid: None)

        cleanups = []

        def fake_cleanup(task_id):
            cleanups.append(task_id)
            with bt._cleanup_lock:
                bt._active_sessions.pop(task_id, None)
                bt._session_last_activity.pop(task_id, None)

        monkeypatch.setattr(bt, "_cleanup_single_browser_session", fake_cleanup)
        fresh = {"session_name": "fresh-session"}
        monkeypatch.setattr(bt, "_create_local_session", lambda _tid: dict(fresh))

        session = bt._get_session_info(TASK)

        assert cleanups == [TASK]  # suspect session recycled exactly once
        assert session["session_name"] == "fresh-session"
        assert bt._active_sessions[TASK] is session
        assert TASK not in bt._suspect_browser_sessions  # flag consumed

        # A second call reuses the fresh session without another recycle.
        again = bt._get_session_info(TASK)
        assert again is session
        assert cleanups == [TASK]

    def test_ensure_healthy_true_without_suspect_flag(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            bt, "_cleanup_single_browser_session", lambda t: called.append(t)
        )
        assert bt._browser_session_backend(TASK).ensure_healthy() is True
        assert called == []


class TestSuccessfulCallNeverRecycles:
    def test_successful_command_does_not_recycle_or_mark(self, monkeypatch, tmp_path):
        """REQUIRED negative probe: success must not touch the cached session."""
        session_info = _local_session("healthy-session")
        bt._active_sessions[TASK] = session_info

        payload = json.dumps({"success": True, "data": {"ok": 1}}).encode()

        class FakePopen:
            returncode = 0

            def __init__(self, *_args, **kwargs):
                os.write(kwargs["stdout"], payload)

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        process = None  # FakePopen installed above; stub the rest.
        _install_command_stubs(monkeypatch, tmp_path, process)
        monkeypatch.setattr(subprocess, "Popen", FakePopen)  # re-assert after stubs

        recycle_calls = []
        monkeypatch.setattr(
            bt, "_cleanup_single_browser_session",
            lambda t: recycle_calls.append(t),
        )
        discard_calls = []
        monkeypatch.setattr(
            bt, "_discard_timed_out_browser_session",
            lambda *a: discard_calls.append(a),
        )
        kills = []
        monkeypatch.setattr("agent.deadline.kill_process_tree", lambda pid, **_k: kills.append(pid))

        result = bt._run_browser_command(TASK, "click", ["@e1"], timeout=5)

        assert result == {"success": True, "data": {"ok": 1}}
        assert bt._active_sessions[TASK] is session_info  # cache untouched
        assert bt._suspect_browser_sessions == {}  # never marked suspect
        assert recycle_calls == []  # never recycled
        assert discard_calls == []
        assert kills == []


class TestWedgedDaemonTreeKill:
    def test_wedged_daemon_is_tree_killed_and_session_evicted(
        self, monkeypatch, tmp_path
    ):
        session_info = _local_session("wedged-session")
        bt._active_sessions[TASK] = session_info
        bt._session_last_activity[TASK] = 1.0
        bt._last_active_session_key[TASK] = TASK

        daemon_pid = 5150
        socket_dir = tmp_path / "agent-browser-wedged-session"
        socket_dir.mkdir()
        (socket_dir / "wedged-session.pid").write_text(str(daemon_pid))

        process = Mock()
        process.returncode = -9
        process.wait.side_effect = [subprocess.TimeoutExpired("agent-browser", 1), -9]
        _install_command_stubs(monkeypatch, tmp_path, process)

        # Wedged: daemon PID exists but the control socket is unresponsive.
        monkeypatch.setattr(bt, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(bt, "_verify_reapable_browser_daemon", lambda *_a: True)
        monkeypatch.setattr(bt, "_browser_daemon_responsive", lambda *_a, **_k: False)

        kills = []
        monkeypatch.setattr(
            "agent.deadline.kill_process_tree",
            lambda pid, **_k: kills.append(pid) or True,
        )

        result = bt._run_browser_command(TASK, "click", ["@e1"], timeout=1)

        assert result["success"] is False
        assert kills == [daemon_pid]  # tree-kill hit the daemon PID
        assert TASK not in bt._active_sessions  # evicted now, not at next use
        assert TASK not in bt._session_last_activity
        assert TASK not in bt._last_active_session_key
        assert not socket_dir.exists()  # socket dir reclaimed

    def test_dead_daemon_skips_kill_but_still_evicts(self, monkeypatch, tmp_path):
        """No PID file → nothing to kill, but the session is still discarded."""
        session_info = _local_session("dead-session")
        bt._active_sessions[TASK] = session_info
        socket_dir = str(tmp_path / "agent-browser-dead-session")
        os.makedirs(socket_dir)

        kills = []
        monkeypatch.setattr(
            "agent.deadline.kill_process_tree",
            lambda pid, **_k: kills.append(pid) or True,
        )
        monkeypatch.setattr(bt, "_stop_cdp_supervisor", lambda _tid: None)

        bt._handle_browser_command_timeout(TASK, session_info, socket_dir)

        assert kills == []
        assert TASK not in bt._active_sessions
        # The eviction already removed the poisoned entry, so the flag is
        # dropped too — it must not poison a later session under this key.
        assert TASK not in bt._suspect_browser_sessions


class TestFreshSessionClearsStaleFlag:
    def test_new_session_creation_drops_stale_suspect_flag(self, monkeypatch):
        """Wedged path evicts + flags; the fresh session must not inherit it."""
        bt._suspect_browser_sessions[TASK] = "stale reason"

        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr(bt, "_ensure_cdp_supervisor", lambda _tid: None)
        monkeypatch.setattr(
            bt, "_create_local_session", lambda _tid: {"session_name": "fresh"}
        )

        session = bt._get_session_info(TASK)

        assert session["session_name"] == "fresh"
        assert TASK not in bt._suspect_browser_sessions
