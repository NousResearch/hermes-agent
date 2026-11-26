"""A checkpointed heartbeat must keep beating after a gateway restart.

The heartbeat emitter thread only starts at spawn time (arm_heartbeat is its sole
call site). Checkpoint recovery re-arms completion watchers but restarted the
emitter for nobody, so an adopted long-running job's periodic heartbeats stayed
silently dead for the rest of its life.
"""

import json
import subprocess
import sys

from tools.process_registry import ProcessRegistry


def _sleep_proc() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
    )


def test_recovery_restarts_the_heartbeat_emitter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    proc = _sleep_proc()
    try:
        start = ProcessRegistry._safe_host_start_time(proc.pid)
        assert start is not None
        (tmp_path / "processes.json").write_text(json.dumps([{
            "session_id": "proc_hb", "command": "long-job", "pid": proc.pid,
            "pid_scope": "host", "host_start_time": start,
            "notify_on_complete": True, "watcher_interval": 5,
            "watcher_platform": "telegram", "watcher_chat_id": "123",
            "heartbeat_seconds": 5, "session_key": "agent:main:telegram:dm:1",
        }]), encoding="utf-8")

        registry = ProcessRegistry()
        assert registry.recover_from_checkpoint() == 1

        session = registry._running.get("proc_hb")
        assert session is not None and session.heartbeat_seconds == 5
        thread = registry._heartbeat_thread
        assert thread is not None and thread.is_alive(), (
            "recovered session carries heartbeat_seconds>0 but the emitter "
            f"thread never started: {thread!r}"
        )
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_recovery_without_heartbeat_starts_no_emitter(tmp_path, monkeypatch):
    """A checkpoint with no heartbeat intervals must not spin up the thread."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    proc = _sleep_proc()
    try:
        start = ProcessRegistry._safe_host_start_time(proc.pid)
        assert start is not None
        (tmp_path / "processes.json").write_text(json.dumps([{
            "session_id": "proc_plain", "command": "plain-job", "pid": proc.pid,
            "pid_scope": "host", "host_start_time": start,
        }]), encoding="utf-8")

        registry = ProcessRegistry()
        assert registry.recover_from_checkpoint() == 1

        thread = registry._heartbeat_thread
        assert thread is None or not thread.is_alive(), (
            "recovery of a heartbeat-less checkpoint must not start the emitter thread"
        )
    finally:
        proc.terminate()
        proc.wait(timeout=10)
