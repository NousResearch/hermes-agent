"""Hard runtime deadlines for tracked background terminal processes."""
import json
import os
import queue
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from tools.process_registry import ProcessRegistry


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _pid_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid_alive(pid):
    if not _pid_exists(pid):
        return False
    try:
        state = subprocess.run(
            ["/bin/ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True, check=True,
        ).stdout.strip()
        return not state.startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def test_runtime_deadline_is_enforced_without_wait():
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 30", runtime_deadline=time.time() + 0.5)
    session.notify_on_complete = True

    assert _wait_until(lambda: session._completion_event.is_set()), "deadline was not enforced"
    assert session.exited is True
    assert session.completion_reason == "timed_out"
    assert session.termination_source == "terminal.timeout"
    assert session.exit_code == -15
    time.sleep(0.2)
    completions = []
    while not registry.completion_queue.empty():
        event = registry.completion_queue.get_nowait()
        if event.get("type") == "completion":
            completions.append(event)
    assert len(completions) == 1
    assert completions[0]["completion_reason"] == "timed_out"
    assert completions[0]["termination_source"] == "terminal.timeout"
    assert completions[0]["exit_code"] == -15


def test_deadline_without_timeout_leaves_daemon_running():
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 0.1", runtime_deadline=0.0)

    assert not session.runtime_deadline
    time.sleep(0.25)
    assert registry.poll(session.id)["status"] == "exited"


def test_expired_recovered_deadline_is_handled_immediately(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    checkpoint = tmp_path / "processes.json"
    checkpoint.write_text(json.dumps([{
        "session_id": "proc_expired", "command": "sleep 30", "pid": child.pid,
        "pid_scope": "host", "started_at": time.time() - 10,
        "runtime_deadline": time.time() - 1, "notify_on_complete": True,
    }]))
    registry = ProcessRegistry()
    try:
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 1
        session = registry.get("proc_expired")
        assert session is not None
        assert _wait_until(lambda: session._completion_event.is_set())
        assert session.completion_reason == "timed_out"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_deadline_survivor_failure_stays_running_and_emits_actionable_event():
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 30", runtime_deadline=time.time() + 0.1)
    session.notify_on_complete = True
    survivor = {"pid": 987654, "cmdline": "sleep 30", "status": "alive"}

    try:
        with patch.object(registry, "_terminate_host_pid", return_value=survivor), \
             patch.object(registry, "_post_kill_survivors", return_value=[survivor]):
            assert _wait_until(lambda: not registry.completion_queue.empty())

        event = registry.completion_queue.get_nowait()
        assert event["type"] == "timeout_error"
        assert event["survivors"] == [survivor]
        assert "process(action='kill', session_id=" in event["message"]
        assert session.id in registry._running
        assert session.exited is False
    finally:
        if session.pid is not None:
            registry._terminate_host_pid(session.pid, expected_start=session.host_start_time)


def test_deadline_kills_root_and_descendant_tree():
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 30 & child=$!; echo $child; wait $child", runtime_deadline=time.time() + 0.3)
    assert _wait_until(lambda: bool(session.output_buffer.strip()))
    child_pid = int(session.output_buffer.strip())

    assert _wait_until(lambda: session._completion_event.is_set())
    assert _wait_until(lambda: not _pid_alive(session.pid) and not _pid_alive(child_pid))


def test_normal_exit_before_deadline_cancels_timeout_handling():
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 0.05", runtime_deadline=time.time() + 2)

    assert _wait_until(lambda: session._completion_event.is_set())
    assert session.completion_reason == "exited"
    assert session.termination_source == ""


def test_runtime_deadline_checkpoint_recovers_and_rearms(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    checkpoint = tmp_path / "processes.json"
    checkpoint.write_text(json.dumps([{
        "session_id": "proc_recovered", "command": "sleep 30", "pid": child.pid,
        "pid_scope": "host", "started_at": time.time(), "runtime_deadline": time.time() + 0.15,
        "notify_on_complete": True,
    }]))
    registry = ProcessRegistry()
    try:
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 1

        session = registry.get("proc_recovered")
        assert session is not None
        assert _wait_until(lambda: session._completion_event.is_set())
        assert session.completion_reason == "timed_out"
        assert session.termination_source == "terminal.timeout"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
