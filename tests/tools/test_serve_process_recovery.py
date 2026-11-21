"""Serve-backend process-checkpoint recovery (upstream port).

``hermes serve`` (the Desktop backend) never ran
``process_registry.recover_from_checkpoint()`` at startup — only ``hermes
gateway run`` did. Any backend restart (supervisor respawn, crash, update)
orphaned every checkpointed background child: alive children lost their exit
observer (no reader thread, no gateway watcher loop), and dead children were
pruned silently — no process-results receipt, no completion wake. The owning
session never learns its background process finished.

These tests pin the three halves of the fix:
- lifespan recovery: the serve backend adopts checkpointed children at startup
  (launch profile; multiplexed secondaries under their own scope);
- detached sweeper: a recovered-alive child's exit is observed (receipt +
  completion wake) without any gateway watcher loop;
- unobserved exit: a checkpointed ``notify_on_complete`` child whose PID is
  already dead publishes a ``reason="lost"`` wake instead of silence — an
  existing receipt wins (the exit WAS observed by someone).
"""

import json
import subprocess
import sys
import time

import pytest

from tools.process_registry import CHECKPOINT_PATH, ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _sleep_proc() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
    )


def _write_checkpoint(tmp_path, entries) -> None:
    path = tmp_path / "procs.json"
    path.write_text(json.dumps(entries))
    import tools.process_registry as pr

    pr.CHECKPOINT_PATH = path
    return path


def _completions(registry) -> list:
    found = []
    while not registry.completion_queue.empty():
        found.append(registry.completion_queue.get_nowait())
    return found


@pytest.fixture(autouse=True)
def _restore_checkpoint_path():
    import tools.process_registry as pr

    yield
    pr.CHECKPOINT_PATH = CHECKPOINT_PATH


class TestServeLifespanRecovery:
    def test_recover_serve_processes_recovers_launch_profile(self, registry, tmp_path, monkeypatch):
        """The helper the serve lifespan calls adopts checkpointed children."""
        proc = _sleep_proc()
        try:
            start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert start is not None
            _write_checkpoint(tmp_path, [{
                "session_id": "proc_serve_live",
                "command": "sleep-for-serve-recovery",
                "pid": proc.pid,
                "pid_scope": "host",
                "host_start_time": start,
                "task_id": "t1",
                "notify_on_complete": True,
            }])
            from hermes_cli.web_server_app import _recover_serve_process_checkpoint

            recovered = _recover_serve_process_checkpoint()

            assert recovered == 1
            from tools.process_registry import process_registry
            polled = process_registry.poll("proc_serve_live")
            assert polled["status"] == "running"
        finally:
            proc.kill()
            proc.wait()

    def test_recovery_failure_never_breaks_backend_startup(self, tmp_path, monkeypatch):
        """The lifespan path is best-effort: a broken checkpoint logs and moves on."""
        _write_checkpoint(tmp_path, [{"session_id": "x", "pid": None}])
        monkeypatch.setattr(
            ProcessRegistry, "recover_from_checkpoint",
            staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        from hermes_cli.web_server_app import _recover_serve_process_checkpoint

        assert _recover_serve_process_checkpoint() == 0  # no raise


class TestDetachedSweeperObservesExit:
    def test_dead_checkpointed_child_publishes_lost_wake_and_receipt(self, registry, tmp_path, monkeypatch):
        """A notify child that died while the backend was down wakes its session."""
        # A real, already-dead PID with its real start time: fate == "gone".
        dead = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.05)"],
            stdin=subprocess.DEVNULL,
        )
        dead.wait()
        _write_checkpoint(tmp_path, [{
            "session_id": "proc_died_unobserved",
            "command": "sleep-for-lost-wake",
            "pid": dead.pid,
            "pid_scope": "host",
            "host_start_time": ProcessRegistry._safe_host_start_time(dead.pid),
            "task_id": "t1",
            "notify_on_complete": True,
        }])

        registry.recover_from_checkpoint()

        events = _completions(registry)
        assert len(events) == 1
        evt = events[0]
        assert evt["type"] == "completion"
        assert evt["session_id"] == "proc_died_unobserved"
        assert evt["completion_reason"] == "lost"
        assert evt["exit_code"] is None

    def test_recovered_alive_child_exit_is_observed_by_sweeper(self, registry, tmp_path, monkeypatch):
        """The sweeper thread publishes the exit of a recovered-alive child (no reader)."""
        proc = _sleep_proc()
        try:
            start = ProcessRegistry._safe_host_start_time(proc.pid)
            _write_checkpoint(tmp_path, [{
                "session_id": "proc_sweep_me",
                "command": "sleep-for-sweep",
                "pid": proc.pid,
                "pid_scope": "host",
                "host_start_time": start,
                "task_id": "t1",
                "notify_on_complete": True,
            }])
            registry.recover_from_checkpoint()
            assert _completions(registry) == []  # alive: no premature wake

            proc.kill()
            proc.wait()

            registry._sweep_detached_once()

            events = _completions(registry)
            assert len(events) == 1
            assert events[0]["completion_reason"] == "lost"
            # poll() must NOT report an invented exit status (upstream policy:
            # _uncollected_gone) — the wake, not a fabricated status, is the fix.
            assert registry.poll("proc_sweep_me")["status"] == "not_found"
            # A receipt now exists: a second sweep/recovery must not re-publish.
            registry._sweep_detached_once()
            assert _completions(registry) == []
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_existing_receipt_wins_over_lost_wake(self, registry, tmp_path, monkeypatch):
        """A receipt on disk proves someone observed the exit: no duplicate wake."""
        dead = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.05)"],
            stdin=subprocess.DEVNULL,
        )
        dead.wait()
        _write_checkpoint(tmp_path, [{
            "session_id": "proc_already_receipted",
            "command": "sleep-for-receipt",
            "pid": dead.pid,
            "pid_scope": "host",
            "host_start_time": ProcessRegistry._safe_host_start_time(dead.pid),
            "task_id": "t1",
            "notify_on_complete": True,
        }])
        # Simulate the observed receipt: save_completed_result already ran.
        from tools.process_registry_results import _result_paths, save_completed_result

        session = ProcessSession(
            id="proc_already_receipted", command="sleep-for-receipt", task_id="t1",
            started_at=time.time(), pid=dead.pid, pid_scope="host", detached=True,
            host_start_time=None, notify_on_complete=True,
        )
        session.exited = True
        session.exit_code = 0
        save_completed_result(session)

        registry.recover_from_checkpoint()

        assert _completions(registry) == []  # observed exit: no lost wake


class TestSweeperLifecycle:
    def test_sweeper_starts_once_and_is_daemon(self, registry):
        registry._ensure_detached_sweeper()
        first = registry._detached_sweeper_thread
        assert first is not None and first.daemon is True
        registry._ensure_detached_sweeper()
        assert registry._detached_sweeper_thread is first
