"""Post-kill tree-death verification (#115490).

A kill that leaves survivors must NOT write a killed receipt or prune the
session: the live tree would become unmanageable (missing from the
background list while still holding resources).
"""

import time
from unittest.mock import MagicMock, patch

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
