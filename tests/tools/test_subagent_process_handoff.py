"""Subagent → parent background-process handoff (process_manage action='handoff').

Ownership is the process's ``owner_task_id``: completion notices are stamped from it at exit and the parent's
``drain_notifications`` suppresses ``sa-`` owners, so a child's watcher never reaches the parent and is killed at child
teardown. A validated handoff flips the owner so the completion lands in the parent's chat; anything not handed off is
named on the child's result as orphaned before teardown kills it.
"""

import json
import time
import weakref

import pytest

from tools.delegate_tool import _register_subagent, _unregister_subagent
from tools.process_registry import ProcessRegistry, process_registry, _handle_process
from tools.process_registry_notifications import _process_accounting_lines, format_process_notification


class _Parent:
    def __init__(self):
        self.session_id = "sess-handoff"
        self._current_task_id = "parent-turn-1"


class _Child:
    def __init__(self, parent):
        self._delegate_parent_ref = weakref.ref(parent)


def _register(sid, child):
    _register_subagent({"subagent_id": sid, "parent_id": None, "depth": 0, "goal": "g", "model": "m",
                        "started_at": time.time(), "status": "running", "tool_count": 0, "agent": child})


@pytest.fixture(autouse=True)
def _plain_spawn(monkeypatch):
    """Spawn plain children: the systemd-run --user --scope wrapper is irrelevant here and stalls under pytest."""
    import tools.process_registry as _pr
    monkeypatch.setattr(_pr, "_SYSTEMD_SCOPE_AVAILABLE", False)


@pytest.fixture
def clean_queue():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def test_handed_off_process_completion_reaches_parent_and_leftover_is_reported(clean_queue):
    """A real child-owned process handed off carries the parent's owner id (so the parent's drain accepts it, with the
    handoff purpose), while a sibling the child did not hand off is still owned by the child and is listed as orphaned
    on the child's result."""
    sid = "sa-0-handoff01"
    parent = _Parent()
    child = _Child(parent)
    _register(sid, child)
    try:
        handed = process_registry.spawn_local("sleep 0.4; echo ci-green", task_id=sid, owner_task_id=sid)
        handed.notify_on_complete = True
        leftover = process_registry.spawn_local("sleep 30", task_id=sid, owner_task_id=sid)

        out = json.loads(_handle_process(
            {"action": "handoff", "session_id": handed.id, "data": "CI watcher for PR 1"}, task_id=sid))
        assert out["status"] == "handed_off"
        assert handed.owner_task_id == "parent-turn-1" and handed.session_key == "sess-handoff"
        assert child._handed_off_processes[0]["session_id"] == handed.id

        # Only the un-handed sibling is left in the child's name — this is what the result entry reports as orphaned.
        assert [s.id for s in process_registry.running_owned_by(sid)] == [leftover.id]

        # Let it exit on its own — a registry wait() would mark the completion consumed (that is the parent-observed path).
        deadline = time.time() + 10
        while process_registry.completion_queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        assert handed.exited
        # The parent drains with the default suppression of sa- owners: the handed-off completion passes it.
        events = process_registry.drain_notifications(owns_event=lambda e: True)
        mine = [(e, text) for e, text in events if e.get("session_id") == handed.id]
        assert len(mine) == 1
        evt, text = mine[0]
        assert evt["owner_task_id"] == "parent-turn-1"
        assert "Handed off to you by a subagent" in text and "CI watcher for PR 1" in text
    finally:
        process_registry.kill_all(source="test")
        _unregister_subagent(sid)


def test_handoff_refuses_exited_foreign_or_non_child_callers(clean_queue):
    """A PID in prose is not a transfer: handoff is an error for a process that already exited, one the caller does
    not own, or a caller that is not a registered subagent — never a silent no-op."""
    sid, other = "sa-0-handoff02", "sa-1-handoff03"
    parent = _Parent()
    _register(sid, _Child(parent))
    try:
        done = process_registry.spawn_local("true", task_id=sid, owner_task_id=sid)
        process_registry.wait(done.id, timeout=10)
        assert "error" in json.loads(_handle_process(
            {"action": "handoff", "session_id": done.id, "data": "x"}, task_id=sid))

        # A notify process that exited while the child was alive and was never read is reported to the parent;
        # the one the child waited on (read) is not.
        unread = process_registry.spawn_local("echo UNREAD_RESULT", task_id=sid, owner_task_id=sid)
        unread.notify_on_complete = True
        deadline = time.time() + 10
        while not unread.exited and time.time() < deadline:
            time.sleep(0.05)
        ids = [s.id for s in process_registry.unread_completions_owned_by(sid)]
        assert ids == [unread.id]
        assert "UNREAD_RESULT" in _process_accounting_lines(
            {"unread_completions": [{"session_id": unread.id, "command": "echo", "exit_code": 0,
                                     "output_tail": unread.output_buffer}]})[0]

        foreign = process_registry.spawn_local("sleep 30", task_id=other, owner_task_id=other)
        assert "error" in json.loads(_handle_process(
            {"action": "handoff", "session_id": foreign.id, "data": "x"}, task_id=sid))
        assert foreign.owner_task_id == other

        assert "error" in json.loads(_handle_process(
            {"action": "handoff", "session_id": foreign.id, "data": "x"}, task_id="not-a-subagent"))

        # A completion for an un-handed child process is still suppressed in the parent.
        reg = ProcessRegistry()
        reg.completion_queue.put({"type": "completion", "session_id": "proc_x", "task_id": other,
                                  "owner_task_id": other, "command": "sleep", "exit_code": 0, "output": ""})
        assert reg.drain_notifications() == []
        assert format_process_notification({"type": "completion", "session_id": "p", "command": "c", "exit_code": 0,
                                            "output": "", "handoff_note": "why"}).count("Handed off to you") == 1
    finally:
        process_registry.kill_all(source="test")
        _unregister_subagent(sid)
