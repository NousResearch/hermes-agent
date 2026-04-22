"""Tests for notify_on_complete background process feature.

Covers:
  - ProcessSession.notify_on_complete field
  - ProcessRegistry.completion_queue population on _move_to_finished()
  - Checkpoint persistence of notify_on_complete
  - Terminal tool schema includes notify_on_complete
  - Terminal tool handler passes notify_on_complete through
"""

import json
import os
import queue
import shlex
import sys
import threading
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(
    sid="proc_test_notify",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    notify_on_complete=False,
) -> ProcessSession:
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
        notify_on_complete=notify_on_complete,
    )
    return s


def _wait_until(predicate, timeout=5, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _drain_completion_queue(registry):
    items = []
    while True:
        try:
            items.append(registry.completion_queue.get_nowait())
        except queue.Empty:
            return items


# =========================================================================
# ProcessSession field
# =========================================================================

class TestProcessSessionField:
    def test_default_false(self):
        s = ProcessSession(id="proc_1", command="echo hi")
        assert s.notify_on_complete is False

    def test_set_true(self):
        s = ProcessSession(id="proc_1", command="echo hi", notify_on_complete=True)
        assert s.notify_on_complete is True


# =========================================================================
# Completion queue
# =========================================================================

class TestCompletionQueue:
    def test_queue_exists(self, registry):
        assert hasattr(registry, "completion_queue")
        assert registry.completion_queue.empty()

    def test_move_to_finished_no_notify(self, registry):
        """Processes without notify_on_complete don't enqueue."""
        s = _make_session(notify_on_complete=False, output="done")
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)
        assert registry.completion_queue.empty()

    def test_move_to_finished_with_notify(self, registry):
        """Processes with notify_on_complete push to queue."""
        s = _make_session(
            notify_on_complete=True,
            output="build succeeded",
            exit_code=0,
        )
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        assert not registry.completion_queue.empty()
        completion = registry.completion_queue.get_nowait()
        assert completion["session_id"] == s.id
        assert completion["command"] == "echo hello"
        assert completion["exit_code"] == 0
        assert "build succeeded" in completion["output"]

    def test_move_to_finished_nonzero_exit(self, registry):
        """Nonzero exit codes are captured correctly."""
        s = _make_session(
            notify_on_complete=True,
            output="FAILED",
            exit_code=1,
        )
        s.exited = True
        s.exit_code = 1
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        completion = registry.completion_queue.get_nowait()
        assert completion["exit_code"] == 1
        assert "FAILED" in completion["output"]

    def test_output_truncated_to_2000(self, registry):
        """Long output is truncated to last 2000 chars."""
        long_output = "x" * 5000
        s = _make_session(
            notify_on_complete=True,
            output=long_output,
        )
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        completion = registry.completion_queue.get_nowait()
        assert len(completion["output"]) == 2000

    def test_multiple_completions_queued(self, registry):
        """Multiple notify processes all push to the same queue."""
        for i in range(3):
            s = _make_session(
                sid=f"proc_{i}",
                notify_on_complete=True,
                output=f"output_{i}",
            )
            s.exited = True
            s.exit_code = 0
            registry._running[s.id] = s
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)

        completions = []
        while not registry.completion_queue.empty():
            completions.append(registry.completion_queue.get_nowait())
        assert len(completions) == 3
        ids = {c["session_id"] for c in completions}
        assert ids == {"proc_0", "proc_1", "proc_2"}


# =========================================================================
# Checkpoint persistence
# =========================================================================

class TestEndToEndNotifyFlow:
    def test_long_running_poll_log_then_wait_still_notifies(self, registry, tmp_path):
        session = ProcessSession(
            id="proc_long_running",
            command="fake long task",
            task_id="t1",
            started_at=time.time(),
            cwd=str(tmp_path),
            notify_on_complete=True,
        )
        registry._running[session.id] = session

        def finish_in_background():
            with session._lock:
                session.output_buffer += "start\n"
            time.sleep(0.4)
            with session._lock:
                session.output_buffer += "middle\n"
            time.sleep(0.8)
            with session._lock:
                session.output_buffer += "done\n"
                session.exited = True
                session.exit_code = 0
            registry._move_to_finished(session)

        worker = threading.Thread(target=finish_in_background, daemon=True)
        worker.start()

        assert registry.completion_queue.empty()
        assert _wait_until(
            lambda: registry.poll(session.id)["status"] == "running"
            and "start" in registry.poll(session.id)["output_preview"],
            timeout=2,
        )

        poll_result = registry.poll(session.id)
        assert poll_result["status"] == "running"
        assert "start" in poll_result["output_preview"]
        assert registry.completion_queue.empty()

        log_result = registry.read_log(session.id)
        assert log_result["status"] == "running"
        assert "start" in log_result["output"]
        assert log_result["total_lines"] >= 1
        assert registry.completion_queue.empty()

        wait_result = registry.wait(session.id, timeout=5)
        worker.join(timeout=1)
        assert wait_result["status"] == "exited"
        assert wait_result["exit_code"] == 0
        assert "done" in wait_result["output"]

        assert _wait_until(lambda: not registry.completion_queue.empty(), timeout=2)
        completion = registry.completion_queue.get_nowait()
        assert completion["session_id"] == session.id
        assert completion["exit_code"] == 0
        assert "start" in completion["output"]
        assert "middle" in completion["output"]
        assert "done" in completion["output"]


class TestTerminalAndProcessIntegration:
    def test_terminal_background_poll_log_wait_still_enqueues_completion(self, monkeypatch, tmp_path):
        from tools.process_registry import _handle_process, process_registry
        from tools.terminal_tool import cleanup_vm, terminal_tool

        monkeypatch.setenv("TERMINAL_ENV", "local")
        task_id = f"notify_flow_{time.time_ns()}"
        _drain_completion_queue(process_registry)

        python_code = (
            "import time; "
            "print('start', flush=True); "
            "time.sleep(0.4); "
            "print('middle', flush=True); "
            "time.sleep(0.8); "
            "print('done', flush=True)"
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(python_code)}"

        try:
            start_result = json.loads(
                terminal_tool(
                    command=command,
                    background=True,
                    notify_on_complete=True,
                    task_id=task_id,
                    workdir=str(tmp_path),
                )
            )
            session_id = start_result["session_id"]
            assert start_result["notify_on_complete"] is True
            assert start_result["exit_code"] == 0
            assert process_registry.completion_queue.empty()

            assert _wait_until(
                lambda: (
                    poll := json.loads(_handle_process({"action": "poll", "session_id": session_id}))
                )["status"] == "running"
                and "start" in poll["output_preview"],
                timeout=3,
            )

            poll_result = json.loads(_handle_process({"action": "poll", "session_id": session_id}))
            assert poll_result["status"] == "running"
            assert "start" in poll_result["output_preview"]
            assert process_registry.completion_queue.empty()

            log_result = json.loads(_handle_process({"action": "log", "session_id": session_id}))
            assert log_result["status"] == "running"
            assert "start" in log_result["output"]
            assert log_result["total_lines"] >= 1
            assert process_registry.completion_queue.empty()

            wait_result = json.loads(
                _handle_process({"action": "wait", "session_id": session_id, "timeout": 5})
            )
            assert wait_result["status"] == "exited"
            assert wait_result["exit_code"] == 0
            assert "done" in wait_result["output"]

            assert _wait_until(lambda: not process_registry.completion_queue.empty(), timeout=2)
            completion = process_registry.completion_queue.get_nowait()
            assert completion["session_id"] == session_id
            assert completion["exit_code"] == 0
            assert "start" in completion["output"]
            assert "middle" in completion["output"]
            assert "done" in completion["output"]
        finally:
            session = process_registry.get(start_result["session_id"]) if 'start_result' in locals() else None
            if session and not session.exited:
                process_registry.kill_process(start_result["session_id"])
            cleanup_vm(task_id)
            _drain_completion_queue(process_registry)


class TestCheckpointNotify:
    def test_checkpoint_includes_notify(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session(notify_on_complete=True)
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["notify_on_complete"] is True

    def test_checkpoint_without_notify(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session(notify_on_complete=False)
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert data[0]["notify_on_complete"] is False

    def test_recover_preserves_notify(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "notify_on_complete": True,
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            s = registry.get("proc_live")
            assert s.notify_on_complete is True

    def test_recover_requeues_notify_watchers(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "session_key": "sk1",
            "watcher_platform": "telegram",
            "watcher_chat_id": "123",
            "watcher_thread_id": "42",
            "watcher_interval": 5,
            "notify_on_complete": True,
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert len(registry.pending_watchers) == 1
            assert registry.pending_watchers[0]["notify_on_complete"] is True

    def test_recover_defaults_false(self, registry, tmp_path):
        """Old checkpoint entries without the field default to False."""
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            s = registry.get("proc_live")
            assert s.notify_on_complete is False


# =========================================================================
# Terminal tool schema
# =========================================================================

class TestTerminalSchema:
    def test_schema_has_notify_on_complete(self):
        from tools.terminal_tool import TERMINAL_SCHEMA
        props = TERMINAL_SCHEMA["parameters"]["properties"]
        assert "notify_on_complete" in props
        assert props["notify_on_complete"]["type"] == "boolean"
        assert props["notify_on_complete"]["default"] is False

    def test_handler_passes_notify(self):
        """_handle_terminal passes notify_on_complete to terminal_tool."""
        from tools.terminal_tool import _handle_terminal
        with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
            _handle_terminal(
                {"command": "echo hi", "background": True, "notify_on_complete": True},
                task_id="t1",
            )
            _, kwargs = mock_tt.call_args
            assert kwargs["notify_on_complete"] is True


# =========================================================================
# Code execution blocked params
# =========================================================================

class TestCodeExecutionBlocked:
    def test_notify_on_complete_blocked_in_sandbox(self):
        from tools.code_execution_tool import _TERMINAL_BLOCKED_PARAMS
        assert "notify_on_complete" in _TERMINAL_BLOCKED_PARAMS
