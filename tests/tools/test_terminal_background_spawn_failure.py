"""Background spawn failure reporting and failed-session observability.

``terminal(background=true)`` used to return a fake ``Background process
started`` payload when ``spawn_via_env`` produced a dead-on-arrival session
(no PID, ``completion_reason == "failed_start"``). The session was also
never registered, so ``process(action='list'/'poll')`` answered empty /
``not_found``.

These tests drive the public ``terminal_tool`` + ``process`` surfaces against
a fake non-local environment so they exercise ``spawn_via_env``, not the
local ``Popen`` path. They do not claim that a long-running command that
fails to launch inside a real sandbox is made to start.
"""

from __future__ import annotations

import json
import time

import pytest

import tools.terminal_tool as terminal_tool
from tools.process_registry import (
    MAX_PROCESSES,
    ProcessSession,
    _handle_process,
    process_registry,
)


def _non_local_config(**overrides):
    config = {
        "env_type": "ssh",
        "timeout": 60,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "lifetime_seconds": 3600,
    }
    config.update(overrides)
    return config


class _BrokenSpawnEnvironment:
    """Non-local backend whose background wrapper never emits a PID."""

    env: dict = {}

    def execute(self, command, timeout=None, rewrite_compound_background=False, **kwargs):
        return {
            "output": "bash: /sandbox/tmp/hermes_bg.pid: Read-only file system",
            "returncode": 1,
        }


class _HealthySpawnEnvironment:
    """Non-local backend whose wrapper prints a PID — a real launch."""

    env: dict = {}

    def execute(self, command, timeout=None, rewrite_compound_background=False, **kwargs):
        return {"output": "4242", "returncode": 0}


@pytest.fixture
def background_env(monkeypatch):
    """Route terminal_tool at a caller-supplied fake non-local environment."""

    def _install(environment, task_id="default"):
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _non_local_config())
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            terminal_tool,
            "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        monkeypatch.setitem(terminal_tool._active_environments, task_id, environment)
        monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
        return task_id

    with process_registry._lock:
        pre_running = set(process_registry._running)
        pre_finished = set(process_registry._finished)

    yield _install

    with process_registry._lock:
        leaked_running = [
            process_registry._running.pop(session_id)
            for session_id in list(process_registry._running)
            if session_id not in pre_running
        ]
        for session_id in list(process_registry._finished):
            if session_id not in pre_finished:
                process_registry._finished.pop(session_id, None)
    for session in leaked_running:
        session.exited = True
        thread = getattr(session, "_reader_thread", None)
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive(), f"poller thread {thread.name} outlived the test"


def test_non_local_no_pid_failed_start_returns_failure(background_env):
    task_id = background_env(_BrokenSpawnEnvironment())

    result = json.loads(
        terminal_tool.terminal_tool(
            command="azcopy sync foo bar", background=True, task_id=task_id
        )
    )

    assert result.get("error")
    assert result.get("exit_code") not in (0, None)
    assert result.get("output") != "Background process started"
    assert result.get("status") == "failed_start"


def test_failure_output_and_exit_status_are_preserved(background_env):
    task_id = background_env(_BrokenSpawnEnvironment())

    result = json.loads(
        terminal_tool.terminal_tool(
            command="azcopy sync foo bar", background=True, task_id=task_id
        )
    )

    assert result["exit_code"] == 1
    combined = f"{result.get('output') or ''} {result.get('error') or ''}"
    assert "Read-only file system" in combined


def test_failed_session_visible_in_process_list(background_env):
    task_id = background_env(_BrokenSpawnEnvironment())

    result = json.loads(
        terminal_tool.terminal_tool(
            command="azcopy sync foo bar", background=True, task_id=task_id
        )
    )
    session_id = result["session_id"]

    listed = json.loads(_handle_process({"action": "list"}, task_id=task_id))
    by_id = {row["session_id"]: row for row in listed["processes"]}
    assert session_id in by_id
    assert by_id[session_id]["status"] == "exited"


def test_failed_session_is_pollable(background_env):
    task_id = background_env(_BrokenSpawnEnvironment())

    result = json.loads(
        terminal_tool.terminal_tool(
            command="azcopy sync foo bar", background=True, task_id=task_id
        )
    )
    session_id = result["session_id"]

    polled = json.loads(_handle_process({"action": "poll", "session_id": session_id}))
    assert polled.get("status") != "not_found"
    assert polled["status"] == "exited"
    assert polled["completion_reason"] == "failed_start"
    assert polled["exit_code"] == 1


def test_successful_background_launch_unchanged(background_env):
    task_id = background_env(_HealthySpawnEnvironment())

    result = json.loads(
        terminal_tool.terminal_tool(
            command="./run-server.sh", background=True, task_id=task_id
        )
    )

    assert result.get("error") is None
    assert result.get("exit_code") == 0
    assert result["output"] == "Background process started"
    assert result["session_id"]
    assert result["pid"] == 4242

    polled = process_registry.poll(result["session_id"])
    assert polled.get("status") != "not_found"
    assert polled["status"] == "running"


def test_immediately_completed_success_is_not_failed_start(background_env, monkeypatch):
    """A spawn that already finished with exit 0 is not a launch failure."""
    task_id = background_env(_HealthySpawnEnvironment())
    completed = ProcessSession(
        id="proc_fast_ok",
        command="true",
        task_id=task_id,
        started_at=0.0,
        pid=4242,
        exited=True,
        exit_code=0,
        completion_reason="exited",
        output_buffer="ok\n",
    )
    process_registry._finished[completed.id] = completed

    monkeypatch.setattr(
        process_registry,
        "spawn_via_env",
        lambda **_kwargs: completed,
    )

    result = json.loads(
        terminal_tool.terminal_tool(command="true", background=True, task_id=task_id)
    )

    assert result.get("status") != "failed_start"
    assert result.get("error") is None
    assert result.get("exit_code") == 0
    assert result.get("session_id") == "proc_fast_ok"
    assert result.get("output") != "Background process started"


def test_registry_lifecycle_stays_bounded_after_failed_starts():
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    now = time.time()
    for i in range(MAX_PROCESSES):
        session = ProcessSession(
            id=f"proc_old_{i}",
            command="sleep 1",
            started_at=now - i,
            exited=True,
            exit_code=0,
        )
        registry._finished[session.id] = session

    class RaisingEnv:
        def execute(self, command, **kwargs):
            raise RuntimeError("sandbox unreachable")

    failed = registry.spawn_via_env(RaisingEnv(), "azcopy sync foo bar")
    total = len(registry._running) + len(registry._finished)
    assert total <= MAX_PROCESSES
    assert failed.id in registry._finished
    assert failed.id not in registry._running
    assert len(registry._finished) == MAX_PROCESSES
