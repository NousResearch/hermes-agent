"""A failed child of a still-running detached fan-out is surfaced to the parent immediately.

Batches join on the slowest sibling before ONE consolidated block re-enters. In a 1,393-agent run every wave-1
child died in a 401 storm at 08:29 and the parent learned of it at 09:36 from the batch's "unknown outcome"
block: 66 minutes of a dead wave with nothing running.
"""
import queue
from unittest.mock import patch

from tools import async_delegation as ad
from tools.process_registry_notifications import format_process_notification
from tui_gateway.session_notifications import _notification_event_dedup_key


def _record(status="running"):
    return {"delegation_id": "deleg_x", "status": status, "is_batch": True, "goals": ["a", "b", "c"], "goal": "a",
            "session_key": "sk", "origin_ui_session_id": "ui", "origin_session_id": "", "parent_session_id": "root",
            "dispatched_at": 1.0, "role": "leaf", "model": "m", "context": None, "toolsets": None}


def test_failure_notice_reaches_the_queue_while_the_batch_keeps_running_and_formats_as_early_warning():
    q = queue.Queue()
    entry = {"task_index": 1, "status": "error", "error": "401 authentication_error: key invalid", "duration_seconds": 12.5,
             "live_transcript": "/tmp/live/task-1.log"}
    with patch.object(ad, "_records", {"deleg_x": _record()}), \
         patch("tools.process_registry.process_registry") as reg:
        reg.completion_queue = q
        ad.push_task_failure_notice("deleg_x", entry, n_tasks=3)
        assert ad._records["deleg_x"]["status"] == "running"  # not finalized
    evt = q.get_nowait()
    assert evt["type"] == "async_delegation" and evt["task_failure_notice"] is True
    assert (evt["session_key"], evt["origin_ui_session_id"], evt["parent_session_id"]) == ("sk", "ui", "root")
    text = format_process_notification(evt)
    assert text.startswith("[ASYNC DELEGATION TASK FAILED — deleg_x, task 2/3]")
    assert "Task: b" in text and "401 authentication_error" in text and "/tmp/live/task-1.log" in text
    assert "consolidated results will still arrive" in text


def test_notice_is_not_sent_for_a_finished_batch_and_does_not_dedup_against_the_final_result():
    q = queue.Queue()
    with patch.object(ad, "_records", {"deleg_x": _record(status="completed")}), \
         patch("tools.process_registry.process_registry") as reg:
        reg.completion_queue = q
        ad.push_task_failure_notice("deleg_x", {"task_index": 0, "status": "error"}, n_tasks=3)
    assert q.empty()
    notice = {"type": "async_delegation", "delegation_id": "deleg_x", "task_failure_notice": True, "results": [{"task_index": 2}]}
    final = {"type": "async_delegation", "delegation_id": "deleg_x", "is_batch": True, "results": []}
    assert _notification_event_dedup_key(notice) != _notification_event_dedup_key(final)


def test_interim_notice_never_claims_or_acknowledges_the_batch_final_row(tmp_path, monkeypatch):
    """Independent-review witness: a busy parent that drained the notice first acknowledged the FINAL
    result's durable row, and the consolidated result was never delivered (nor replayed after restart)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    notice = {"type": "async_delegation", "delegation_id": "deleg_x", "task_failure_notice": True, "results": [{"task_index": 0}]}
    final = {"type": "async_delegation", "delegation_id": "deleg_x", "is_batch": True, "results": []}
    # The notice is a non-durable event: empty token, no row touched.
    assert ad.claim_event_delivery(notice, "tui-poller") == ""
    ad.complete_event_delivery(notice, "")
    assert ad.is_interim_delegation_event(notice) and not ad.is_interim_delegation_event(final)


def test_gateway_dedup_identity_separates_notices_from_the_final_and_from_each_other():
    from gateway.run_notifications import GatewayNotificationsMixin
    ident = GatewayNotificationsMixin._completion_delivery_identity
    n0 = {"type": "async_delegation", "delegation_id": "d", "task_failure_notice": True, "results": [{"task_index": 0}]}
    n1 = {"type": "async_delegation", "delegation_id": "d", "task_failure_notice": True, "results": [{"task_index": 1}]}
    final = {"type": "async_delegation", "delegation_id": "d", "is_batch": True, "results": []}
    assert len({ident(n0), ident(n1), ident(final)}) == 3
