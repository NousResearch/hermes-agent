"""Tests for spawn-time completion-notification intent in tools/process_registry.py.

The intent has to be seeded while the session is created: a process that exits
before ``spawn_local`` returns would otherwise never queue a completion event.
"""

import pytest

from tools.process_registry import ProcessRegistry


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def test_fast_process_keeps_spawn_time_notification_intent(registry, tmp_path):
    session = registry.spawn_local(
        "printf fast-done",
        cwd=str(tmp_path),
        session_key="acp-session",
        notify_on_complete=True,
    )
    assert session._completion_event.wait(timeout=5)
    event = registry.completion_queue.get(timeout=1)
    assert event["session_id"] == session.id
    assert event["session_key"] == "acp-session"
    assert event["exit_code"] == 0
