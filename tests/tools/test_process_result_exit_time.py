"""Regression for #155114: completed process exit times survive receipt restoration."""

import json
import os
import time

import pytest

from hermes_constants import get_hermes_home
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_results import RESULT_RETENTION_SECONDS


def _finish_session(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    session = ProcessSession(
        id="proc_exit_time", command="echo done", task_id="task",
        parent_session_id="owner-session", exited=True, exit_code=0,
    )
    registry = ProcessRegistry()
    registry._running[session.id] = session
    registry._move_to_finished(session)
    return session


def test_exit_time_round_trips(monkeypatch):
    session = _finish_session(monkeypatch)

    restored = ProcessRegistry().get(session.id)

    assert session.exited_at > 0
    assert restored is not None
    assert restored.exited_at == session.exited_at


@pytest.mark.parametrize("value", [None, 0])
def test_legacy_receipt_exit_time_uses_receipt_mtime(monkeypatch, value):
    session = _finish_session(monkeypatch)
    path = get_hermes_home() / "logs" / "process-results" / f"{session.id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if value is None:
        del record["exited_at"]
    else:
        record["exited_at"] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    modified = int(time.time() - RESULT_RETENTION_SECONDS / 2)
    os.utime(path, (modified, modified))

    restored = ProcessRegistry().get(session.id)

    assert restored is not None
    assert restored.exited_at == modified
