"""An aligned stop decision preserves live policy and scrubs the durable payload."""
import json
import time

import pytest

from tools import async_delegation as ad


@pytest.mark.parametrize("stopped", [False, True])
def test_completion_scrubs_after_stop_decision_without_replacing_live_copy(tmp_path, monkeypatch, stopped):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ad, "worker_ledger", lambda: None)
    ad._reset_for_tests()
    record = {
        "delegation_id": "completion-policy", "goal": "fixture", "context": None,
        "toolsets": None, "role": "leaf", "model": "fixture", "session_key": "owner",
        "origin_ui_session_id": "", "origin_session_id": "", "parent_session_id": "parent",
        "status": "running", "dispatched_at": time.time(), "completed_at": None,
    }
    ad._persist_dispatch(record)
    if stopped:
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("UPDATE async_delegations SET stop_state='cancel_requested' WHERE delegation_id=?",
                         (record["delegation_id"],))
    monkeypatch.setattr(ad, "redact_completion_event", lambda value: {**value, "summary": "scrubbed"})
    monkeypatch.setattr(ad, "redact_child_result", lambda value: {**value, "summary": "scrubbed"})
    event = {"delegation_id": record["delegation_id"], "status": "completed", "summary": "live-success"}
    result = {"summary": "live-success"}
    cancelled_event = {**event, "status": "interrupted", "summary": "live-stop"}
    cancelled_result = {"summary": "live-stop"}
    try:
        actual_event, actual_result = ad._persist_completion(
            event, result, on_cancel=lambda: (cancelled_event, cancelled_result))
        expected_event, expected_result = (cancelled_event, cancelled_result) if stopped else (event, result)
        assert actual_event == expected_event
        assert actual_result == expected_result
        with ad._DB_LOCK, ad._transaction() as conn:
            state, event_json, result_json, stop_state = conn.execute(
                "SELECT state,event_json,result_json,stop_state FROM async_delegations WHERE delegation_id=?",
                (record["delegation_id"],)).fetchone()
        assert state == expected_event["status"]
        assert json.loads(event_json)["summary"] == "scrubbed"
        assert json.loads(result_json)["summary"] == "scrubbed"
        assert stop_state == ("acknowledged" if stopped else "")
    finally:
        ad._reset_for_tests()
