"""Post-kill tree-death verification (#115490).

A kill that leaves survivors must NOT write a killed receipt or prune the
session: the live tree would become unmanageable (missing from the
background list while still holding resources).
"""

import sys
import time
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


def _live_session(sid="proc_killverify1", pid=424242, start=777001):
    s = ProcessSession(
        id=sid, command="sleep 999", task_id="t1", started_at=time.time(),
        pid=pid, pid_scope="host", host_start_time=start,
    )
    proc = MagicMock()
    proc.pid = pid
    s.process = proc
    return s


def _paused_registry_calls():
    """Neutralize tree-kill + checkpoint side effects; return the mocks."""
    t = patch.object(ProcessRegistry, "_terminate_host_pid", return_value=None).start()
    c = patch.object(ProcessRegistry, "_write_checkpoint", return_value=None).start()
    s = patch("tools.process_registry.save_completed_result").start()
    return t, c, s


def test_kill_with_surviving_root_keeps_session_running():
    """Root ignores the kill -> no killed receipt, session stays running."""
    reg = ProcessRegistry()
    s = _live_session()
    s.process.poll.return_value = None  # Popen child still alive
    reg._running[s.id] = s
    try:
        _, _, save = _paused_registry_calls()
        with patch.object(ProcessRegistry, "_host_pid_is_ours", return_value=True), \
             patch("psutil.Process", side_effect=Exception("gone")):
            result = reg.kill_process(s.id)
    finally:
        patch.stopall()
    assert result["status"] != "killed", result
    assert result.get("process_running") is True
    assert result.get("survivors") == [424242], result
    assert s.exited is False
    assert s.completion_reason != "killed"
    assert s.id in reg._running
    assert s.id not in reg._finished
    save.assert_not_called()


def test_kill_with_surviving_descendant_keeps_session_running():
    """Live root with a live owned descendant -> session stays running."""
    reg = ProcessRegistry()
    s = _live_session(sid="proc_killverify2")
    s.process.poll.return_value = None  # root still alive
    reg._running[s.id] = s
    kid = MagicMock()
    kid.pid = 424243
    kid.is_running.return_value = True
    kid.status.return_value = "running"
    parent = MagicMock()
    parent.children.return_value = [kid]
    try:
        _, _, save = _paused_registry_calls()
        with patch.object(ProcessRegistry, "_host_pid_is_ours", return_value=True), \
             patch("psutil.Process", return_value=parent):
            result = reg.kill_process(s.id)
    finally:
        patch.stopall()
    assert result["status"] != "killed", result
    assert set(result.get("survivors", [])) == {424242, 424243}, result
    assert s.exited is False
    assert s.id in reg._running
    assert s.id not in reg._finished
    save.assert_not_called()


def test_kill_with_dead_tree_still_reports_killed():
    """Full tree death preserves the existing killed contract."""
    reg = ProcessRegistry()
    s = _live_session(sid="proc_killverify3")
    s.process.poll.return_value = -15
    reg._running[s.id] = s
    try:
        _, _, save = _paused_registry_calls()
        with patch.object(ProcessRegistry, "_host_pid_is_ours", return_value=False), \
             patch("psutil.Process", side_effect=Exception("gone")):
            result = reg.kill_process(s.id)
    finally:
        patch.stopall()
    assert result["status"] == "killed", result
    assert s.exited is True
    assert s.completion_reason == "killed"
    assert s.id not in reg._running
    assert s.id in reg._finished


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal escalation; Windows uses taskkill")
def test_escalated_kill_of_sigterm_ignoring_child_reports_killed(tmp_path, monkeypatch):
    """The #115490 scenario itself: a child that ignores SIGTERM is SIGKILLed after the grace
    window, and the verification must give the kernel a moment to reap it — poll() right
    after kill() still says alive, which turned every escalated kill into 'Kill incomplete'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds", staticmethod(lambda: 0.2))
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self: None)
    reg = ProcessRegistry()
    s = reg.spawn_local("trap '' TERM; sleep 30", cwd=str(tmp_path))
    try:
        time.sleep(0.3)  # let bash install the trap
        result = reg.kill_process(s.id)
    finally:
        with suppress(Exception):
            s.process.kill()
            s.process.wait(timeout=2)
    assert result["status"] == "killed", result
    assert s.completion_reason == "killed"
    assert s.id in reg._finished
