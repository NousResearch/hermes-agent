"""A gateway ``notify_on_complete`` completion survives a gateway restart.

The completion queue and the per-process watcher are in-memory and the checkpoint holds live
PIDs only, so a process that exited while the gateway was down or stopping left a receipt and
no notification, permanently. Startup now re-arms a watcher from the receipt until a watcher
settles it.
"""

import time

from hermes_constants import get_hermes_home
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_results import mark_result_notified


def _finish(registry, session_id, *, watcher_interval=5):
    """Exit a tracked process the way the reader thread does; the gateway armed its watcher."""
    session = ProcessSession(
        id=session_id, command="echo review", task_id="task-1", owner_task_id="task-1",
        session_key="agent:main:telegram:group:-100999:7", parent_session_id="sess-owner",
        started_at=time.time(), notify_on_complete=True, watcher_interval=watcher_interval,
        watcher_platform="telegram", watcher_chat_id="-100999", watcher_thread_id="7",
        output_buffer="REVIEW DONE\n",
    )
    with registry._lock:
        registry._running[session.id] = session
    session.mark_exited(0)
    registry._move_to_finished(session)


def test_completion_finished_while_gateway_down_is_rearmed_until_settled():
    _finish(ProcessRegistry(), "proc_replay000001")

    restarted = ProcessRegistry()  # the queued completion died with the old process
    assert restarted.completion_queue.empty()
    assert restarted.restore_pending_notifications() == 1
    assert restarted.restore_pending_notifications() == 0  # never armed twice in one process
    [watcher] = restarted.pending_watchers
    assert watcher["session_id"] == "proc_replay000001"
    assert watcher["notify_on_complete"] is True
    assert (watcher["session_key"], watcher["platform"], watcher["chat_id"], watcher["thread_id"]) == (
        "agent:main:telegram:group:-100999:7", "telegram", "-100999", "7")
    assert watcher["parent_session_id"] == "sess-owner"
    # An exited snapshot: never re-run, never adopted as a live PID.
    assert "proc_replay000001" not in restarted._running
    restored = restarted.get("proc_replay000001")
    assert restored.exited and restored.pid is None and restored.process is None
    assert "REVIEW DONE" in restored.output_buffer

    mark_result_notified("proc_replay000001")
    assert ProcessRegistry().restore_pending_notifications() == 0


def test_completion_without_a_gateway_watcher_is_not_rearmed():
    """A CLI process has no watcher route; replaying it from a later gateway would invent one."""
    _finish(ProcessRegistry(), "proc_replay000002", watcher_interval=0)
    assert (get_hermes_home() / "logs" / "process-results" / "proc_replay000002.json").exists()

    restarted = ProcessRegistry()
    assert restarted.restore_pending_notifications() == 0
    assert restarted.pending_watchers == []
