"""A notify_on_complete spawn tells the agent how to describe the job if its turn ends first.

The user reads a turn's last line as the job's status. The spawn result says the completion
arrives as its own turn and asks for a plain "Still running: ..." line rather than an implied
finish; a session that cannot deliver a completion gets the existing poll-only note instead.
"""
import json
import shlex
import sys

import pytest

from tools.process_registry import ProcessRegistry
from tools.terminal_tool import terminal_tool


@pytest.fixture()
def registry(monkeypatch):
    import tools.process_registry as processes
    reg = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", reg)
    yield reg
    reg.kill_all()


def _spawn(tmp_path, *, notify):
    command = shlex.quote(sys.executable) + " -c " + shlex.quote("import time; time.sleep(30)")
    return json.loads(terminal_tool(command, background=True, task_id="note-owner",
                                    workdir=str(tmp_path), notify_on_complete=notify))


def test_notify_spawn_carries_the_status_line_promise(registry, tmp_path):
    result = _spawn(tmp_path, notify=True)

    assert result["notify_on_complete"] is True
    assert "Still running:" in result["notify_note"]


def test_silent_spawn_and_undeliverable_session_get_no_promise(registry, tmp_path, monkeypatch):
    assert "notify_note" not in _spawn(tmp_path, notify=False)

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t-1")  # one-shot worker: nothing can deliver later
    result = _spawn(tmp_path, notify=True)

    assert result["notify_on_complete"] is False and "notify_note" not in result
