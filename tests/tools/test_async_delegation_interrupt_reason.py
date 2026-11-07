"""Regression for #146807: an interrupted delegation records WHY it stopped.

Without the reason the gateway cannot tell the user's ``/stop`` from any other interrupt, so it
wakes the parent with the batch completion and the parent re-dispatches the cancelled work.
"""

import queue

import pytest

import tools.async_delegation as ad
from tools.process_registry_notifications import USER_STOP_INTERRUPT_REASON, format_process_notification


@pytest.fixture(autouse=True)
def _reset_async_delegation():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _seed(delegation_id, interrupt_fn, session_key="sess_A"):
    with ad._records_lock:
        ad._records[delegation_id] = {
            "delegation_id": delegation_id,
            "status": "running",
            "session_key": session_key,
            "interrupt_fn": interrupt_fn,
        }


def test_interrupt_stamps_the_reason_on_the_record():
    calls = []
    _seed("d1", lambda: calls.append(1))

    assert ad.interrupt_for_session(session_key="sess_A", reason=USER_STOP_INTERRUPT_REASON) == 1

    assert calls == [1]
    with ad._records_lock:
        assert ad._records["d1"]["interrupt_reason"] == USER_STOP_INTERRUPT_REASON


def test_failed_interrupt_does_not_claim_a_reason():
    def boom():
        raise RuntimeError("interrupt unavailable")

    _seed("d1", boom)

    assert ad.interrupt_for_session(session_key="sess_A", reason=USER_STOP_INTERRUPT_REASON) == 0

    with ad._records_lock:
        assert "interrupt_reason" not in ad._records["d1"]


def test_completion_event_carries_the_interrupt_reason(monkeypatch):
    from tools import process_registry as pr

    q = queue.Queue()
    monkeypatch.setattr(pr.process_registry, "completion_queue", q)
    monkeypatch.setattr(ad, "_persist_completion", lambda *args, **kwargs: None)
    record = {
        "delegation_id": "d1",
        "status": "interrupted",
        "is_batch": True,
        "interrupt_reason": USER_STOP_INTERRUPT_REASON,
        "session_key": "sess_A",
        "goals": ["review the archive"],
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }
    result = {"results": [{"task_index": 0, "status": "interrupted", "summary": "partial"}]}

    ad._push_completion_event(record, result, "interrupted")

    evt = q.get_nowait()
    assert evt["interrupt_reason"] == USER_STOP_INTERRUPT_REASON
    assert evt["status"] == "interrupted"


def test_cancelled_text_tells_the_parent_not_to_resume():
    evt = {
        "type": "async_delegation",
        "delegation_id": "d1",
        "status": "interrupted",
        "interrupt_reason": USER_STOP_INTERRUPT_REASON,
        "goals": ["review the archive"],
        "results": [{"task_index": 0, "status": "interrupted", "summary": "partial"}],
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }

    text = format_process_notification(evt)

    assert "The user stopped this" in text
    assert "Do not resume" in text


def test_ordinary_completion_keeps_the_existing_prompt():
    evt = {
        "type": "async_delegation",
        "delegation_id": "d1",
        "status": "completed",
        "goals": ["review the archive"],
        "results": [{"task_index": 0, "status": "completed", "summary": "done"}],
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }

    text = format_process_notification(evt)

    assert "The user stopped this" not in text
    assert "re-dispatch" in text
