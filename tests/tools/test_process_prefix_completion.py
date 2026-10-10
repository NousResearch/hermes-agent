"""A prefix passed to process_manage must mark the resolved session, not the raw text."""

import time

from tools.process_registry import ProcessRegistry, ProcessSession


def _exited_session(registry: ProcessRegistry) -> ProcessSession:
    session = ProcessSession(
        id="proc_4dae56ca81f6",
        command="echo done",
        task_id="t1",
        owner_task_id="t1",
        session_key="s1",
        started_at=time.time(),
        notify_on_complete=True,
    )
    session.exited = True
    session.exit_code = 0
    session.output_buffer = "done\n"
    registry._running[session.id] = session
    registry.completion_queue.put({
        "type": "completion",
        "session_id": session.id,
        "task_id": "t1",
        "owner_task_id": "t1",
        "command": session.command,
        "exit_code": 0,
        "output": "done",
    })
    return session


def test_prefix_poll_suppresses_the_resolved_completion():
    registry = ProcessRegistry()
    session = _exited_session(registry)

    registry.poll("4dae")

    assert session.id in registry._poll_observed
    assert "4dae" not in registry._poll_observed
    assert registry.drain_notifications() == []


def test_prefix_wait_marks_the_resolved_session_consumed():
    registry = ProcessRegistry()
    session = _exited_session(registry)

    result = registry.wait("4dae", timeout=1)

    assert result["status"] == "exited"
    assert session.id in registry._completion_consumed
    assert "4dae" not in registry._completion_consumed
    assert registry.drain_notifications(skip_poll_observed=False) == []
