"""Regression tests for session-owned background process cleanup.

Top-level terminal processes share ``task_id='default'`` but retain the
originating agent session in ``session_key``.  Kanban one-shot workers must be
able to reap those processes when their agent closes.
"""

from __future__ import annotations

import os
import shlex
import sys
import threading
import time
import uuid
from unittest.mock import MagicMock

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


def _session(sid: str, *, task_id: str, session_key: str) -> ProcessSession:
    return ProcessSession(
        id=sid,
        command="sleep 60",
        task_id=task_id,
        session_key=session_key,
        started_at=time.time(),
        process=MagicMock(),
        pid=4242,
    )


def test_kill_all_can_select_shared_sandbox_process_by_session_key(monkeypatch):
    registry = ProcessRegistry()
    owned = _session("proc_owned", task_id="default", session_key="kanban-session")
    foreign = _session("proc_foreign", task_id="default", session_key="other-session")
    registry._running = {owned.id: owned, foreign.id: foreign}

    killed: list[str] = []

    def fake_kill(
        session_id: str,
        *,
        source: str,
        consume_output: bool = False,
    ):
        assert consume_output is False
        killed.append(session_id)
        return {"status": "killed"}

    monkeypatch.setattr(registry, "kill_process", fake_kill)

    assert registry.kill_all(session_key="kanban-session") == 1
    assert killed == ["proc_owned"]


def test_kill_all_combines_task_and_session_ownership_without_regression(monkeypatch):
    registry = ProcessRegistry()
    by_task = _session("proc_task", task_id="isolated-task", session_key="other")
    by_session = _session("proc_session", task_id="default", session_key="kanban-session")
    unrelated = _session("proc_unrelated", task_id="default", session_key="other")
    registry._running = {
        by_task.id: by_task,
        by_session.id: by_session,
        unrelated.id: unrelated,
    }

    killed: list[str] = []

    def fake_kill(
        session_id: str,
        *,
        source: str,
        consume_output: bool = False,
    ):
        assert consume_output is False
        killed.append(session_id)
        return {"status": "killed"}

    monkeypatch.setattr(registry, "kill_process", fake_kill)

    assert registry.kill_all(
        task_id="isolated-task", session_key="kanban-session"
    ) == 2
    assert killed == ["proc_task", "proc_session"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process liveness assertion")
def test_session_scoped_cleanup_terminates_real_background_process(tmp_path):
    psutil = pytest.importorskip("psutil")
    registry = ProcessRegistry()
    session = registry.spawn_local(
        command="sleep 60",
        cwd=str(tmp_path),
        task_id="default",
        session_key="kanban-session",
    )
    try:
        assert session.pid is not None
        assert psutil.pid_exists(session.pid)

        assert registry.kill_all(session_key="kanban-session") == 1

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and psutil.pid_exists(session.pid):
            time.sleep(0.05)
        assert not psutil.pid_exists(session.pid)
    finally:
        registry.kill_all()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifecycle E2E")
def test_one_shot_cli_cleanup_reaps_session_owned_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Exercise CLI -> AIAgent.close -> ProcessRegistry with a real process."""
    import cli as cli_mod
    from run_agent import AIAgent
    from tools.process_registry import process_registry

    session_key = f"kanban-e2e-{uuid.uuid4()}"
    command = (
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote('import time; time.sleep(60)')}"
    )
    process_session = process_registry.spawn_local(
        command,
        task_id="default",
        session_key=session_key,
    )
    process = process_session.process
    assert process is not None
    assert process.poll() is None

    agent = object.__new__(AIAgent)
    setattr(agent, "session_id", session_key)
    setattr(agent, "_active_children", set())
    setattr(agent, "_active_children_lock", threading.Lock())
    setattr(agent, "client", None)
    setattr(agent, "_end_session_on_close", False)
    setattr(agent, "_session_db", None)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli_mod, "_cleanup_done", False)
    monkeypatch.setattr(cli_mod, "_active_agent_ref", agent)

    try:
        cli_mod._run_cleanup(notify_session_finalize=False)
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process_registry.kill_all(session_key=session_key)
            process.wait(timeout=5)
