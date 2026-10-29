"""A job Hermes stopped (or lost) for a session is reported to that session even across a restart.

The in-memory completion notice dies with the backend; the durable receipt does not. When the
session's notification poller starts again, the notice is replayed unless the transcript already
has it. A job that died while Hermes was down entirely leaves only its checkpoint entry; recovery
turns it into a ``lost`` receipt so it is replayed the same way.
"""
import json
import time

import pytest

OWNER = "20261020_123307_ad1d0c"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import process_registry_stopped as stopped
    monkeypatch.setattr(stopped, "_REPLAYED", set())
    from hermes_state import SessionDB
    db = SessionDB()
    db.create_session(OWNER, "desktop")
    db.close()
    return tmp_path


def _receipt(proc, *, source, reason="killed", notify=True, owner=OWNER):
    from tools.process_registry import ProcessSession
    from tools.process_registry_results import save_completed_result
    save_completed_result(ProcessSession(
        id=proc, command="python e2e_run_strip.py", task_id=owner, owner_task_id=owner,
        session_key=owner, parent_session_id=owner, started_at=time.time() - 60,
        notify_on_complete=notify, exited=True, exit_code=-15, completion_reason=reason,
        termination_source=source, output_buffer="plan A done\n"))


def _say(role, content):
    from hermes_state import SessionDB
    db = SessionDB()
    db.append_message(OWNER, role, content)
    db.close()


def _pending(**kw):
    from tools.process_registry_stopped import pending_stopped_notices
    return pending_stopped_notices({OWNER}, **kw)


def test_stopped_job_is_replayed_once_until_the_transcript_has_it(home):
    from tools.process_registry_notifications import format_process_notification
    _receipt("proc_aaaa00000001", source="agent_close:tui_shutdown")
    _say("tool", '{"session_id": "proc_aaaa00000001", "status": "killed"}')  # a process list is not "told"

    [event] = _pending()
    text = format_process_notification(event)
    assert event["restored"] is True and event["session_key"] == OWNER
    assert "was STOPPED by Hermes before it finished" in text and "Stopped:" in text
    assert _pending() == []  # once per process, even if the session re-opens before delivery


def test_delivered_user_stopped_or_still_tracked_jobs_are_not_replayed(home):
    _receipt("proc_bbbb00000001", source="agent_close:lru_evict")
    _say("user", "[IMPORTANT: Background process proc_bbbb00000001 was STOPPED by Hermes ...]")
    _receipt("proc_bbbb00000002", source="process.kill")       # the user stopped it
    _receipt("proc_bbbb00000003", source="agent_close")         # a delegated child's teardown
    _receipt("proc_bbbb00000004", source="gateway_shutdown", notify=False)  # nothing was promised
    _receipt("proc_bbbb00000005", source="gateway_shutdown")
    _receipt("proc_bbbb00000006", source="gateway_shutdown", owner="someone-else")

    assert _pending(tracked={"proc_bbbb00000005"}) == []  # its live notice is still queued here


def test_job_that_died_while_hermes_was_down_is_recorded_lost_and_replayed(home):
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_notifications import format_process_notification
    (home / "processes.json").write_text(json.dumps([{
        "session_id": "proc_cccc00000001", "command": "python e2e_run_strip.py", "pid": 2 ** 22 + 7,
        "pid_scope": "host", "host_start_time": 1, "started_at": time.time() - 600, "task_id": OWNER,
        "owner_task_id": OWNER, "session_key": OWNER, "parent_session_id": OWNER,
        "notify_on_complete": True}]))

    assert ProcessRegistry().recover_from_checkpoint() == 0
    receipt = json.loads((home / "logs" / "process-results" / "proc_cccc00000001.json").read_text())
    assert (receipt["completion_reason"], receipt["termination_source"]) == ("lost", "hermes_restart")

    [event] = _pending()
    text = format_process_notification(event)
    assert "was LOST while Hermes was down" in text and "Stopped:" in text


def test_poller_replay_queues_a_notice_the_owner_drains(home):
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_stopped import replay_stopped_notices
    _receipt("proc_dddd00000001", source="agent_close:ws_orphan_reap")
    registry = ProcessRegistry()

    assert replay_stopped_notices({OWNER, ""}, registry) == 1

    [(event, text)] = registry.drain_notifications(OWNER)  # routed to the owner by session_key
    assert event["session_id"] == "proc_dddd00000001" and "ws_orphan_reap" in text
