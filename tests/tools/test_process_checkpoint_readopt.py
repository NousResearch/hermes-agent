"""Checkpoint recovery must not invent an exit for a live child.

A still-live host PID is re-attached and stays running. A confirmed start-time
mismatch means the PID was reused: the old entry is closed and that PID is not
signalled. A gone PID is pruned. A completion is queued only when a real exit
status was collected.
"""

import json
import subprocess
import sys
import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _sleep_proc() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
    )


def _detach(registry, proc, *, sid, start, notify=True) -> ProcessSession:
    session = ProcessSession(
        id=sid,
        command="sleep-for-readopt",
        task_id="t1",
        started_at=time.time(),
        pid=proc.pid,
        pid_scope="host",
        detached=True,
        host_start_time=start,
        notify_on_complete=notify,
    )
    registry._running[session.id] = session
    return session


def _completions(registry) -> list:
    found = []
    while not registry.completion_queue.empty():
        found.append(registry.completion_queue.get_nowait())
    return found


class TestCheckpointReadopt:
    def test_live_matching_pid_stays_running_when_later_start_probe_fails(
        self, registry, monkeypatch
    ):
        """A recovered child that is still alive must not be reported exited
        just because the start-time probe cannot be read."""
        proc = _sleep_proc()
        try:
            start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert start is not None
            session = _detach(registry, proc, sid="proc_live_probe", start=start)
            monkeypatch.setattr(ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid: None))

            polled = registry.poll(session.id)

            assert proc.poll() is None
            assert polled["status"] == "running"
            assert "exit_code" not in polled
            assert session.exited is False
            assert session.id in registry._running
            assert _completions(registry) == []
        finally:
            proc.kill()
            proc.wait()

    def test_reused_pid_is_closed_without_kill_or_completion(self, registry):
        """A live PID whose start time does not match was reused. Close the
        old entry, do not signal that PID, and do not emit a completion."""
        proc = _sleep_proc()
        try:
            real = ProcessRegistry._safe_host_start_time(proc.pid)
            assert real is not None
            session = _detach(registry, proc, sid="proc_reused", start=real + 1)

            polled = registry.poll(session.id)

            assert proc.poll() is None, "reused PID must not be killed"
            assert session.exited is True
            assert session.exit_code is None
            assert session.id in registry._finished
            assert session.id not in registry._running
            assert _completions(registry) == []
            assert polled["status"] != "running"
            assert polled.get("exit_code") is None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_kill_terminates_readopted_pid_when_start_probe_fails(
        self, registry, monkeypatch
    ):
        """kill must reach a re-adopted live child with an unreadable start
        time, not report already_exited and leave it running unsupervised."""
        proc = _sleep_proc()
        try:
            start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert start is not None
            session = _detach(registry, proc, sid="proc_kill_probe", start=start)
            monkeypatch.setattr(ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid: None))

            result = registry.kill_process(session.id)

            assert result["status"] == "killed", result
            proc.wait(timeout=10)
            assert session.exited is True
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_gone_pid_is_pruned_without_a_false_completion(self, registry):
        session = ProcessSession(
            id="proc_gone",
            command="already-dead",
            task_id="t1",
            started_at=time.time(),
            pid=2**31 - 1,
            pid_scope="host",
            detached=True,
            host_start_time=1,
            notify_on_complete=True,
        )
        registry._running[session.id] = session

        listed = registry.list_sessions()
        polled = registry.poll(session.id)

        assert session.id not in registry._running
        assert all(row["session_id"] != session.id for row in listed)
        assert polled["status"] == "not_found"
        assert _completions(registry) == []

    def test_recover_reattaches_live_pid_when_start_probe_fails(
        self, registry, tmp_path, monkeypatch
    ):
        proc = _sleep_proc()
        try:
            start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert start is not None
            checkpoint = tmp_path / "procs.json"
            checkpoint.write_text(json.dumps([{
                "session_id": "proc_recover_live",
                "command": "sleep-for-readopt",
                "pid": proc.pid,
                "pid_scope": "host",
                "host_start_time": start,
                "task_id": "t1",
                "notify_on_complete": True,
            }]))
            monkeypatch.setattr(
                "tools.process_registry.CHECKPOINT_PATH", checkpoint
            )
            monkeypatch.setattr(
                ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid: None)
            )

            recovered = registry.recover_from_checkpoint()
            polled = registry.poll("proc_recover_live")

            assert recovered == 1
            assert proc.poll() is None
            assert polled["status"] == "running"
            assert "exit_code" not in polled
            assert _completions(registry) == []
        finally:
            proc.kill()
            proc.wait()

    def test_recover_does_not_kill_a_reused_pid(self, registry, tmp_path, monkeypatch):
        proc = _sleep_proc()
        try:
            real = ProcessRegistry._safe_host_start_time(proc.pid)
            assert real is not None
            checkpoint = tmp_path / "procs.json"
            checkpoint.write_text(json.dumps([{
                "session_id": "proc_recover_reused",
                "command": "sleep-for-readopt",
                "pid": proc.pid,
                "pid_scope": "host",
                "host_start_time": real + 1,
                "task_id": "t1",
                "notify_on_complete": True,
                "systemd_unit": "hermes-worker-proc_recover_reused.scope",
            }]))
            stopped = []
            monkeypatch.setattr(
                "tools.process_registry.CHECKPOINT_PATH", checkpoint
            )
            monkeypatch.setattr(
                "tools.process_registry._stop_systemd_unit",
                lambda unit: stopped.append(unit) or True,
            )

            recovered = registry.recover_from_checkpoint()

            assert recovered == 0
            assert proc.poll() is None
            assert registry.get("proc_recover_reused") is None
            assert stopped == []
            assert _completions(registry) == []
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
