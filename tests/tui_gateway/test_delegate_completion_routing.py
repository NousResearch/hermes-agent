"""A delegate child must never claim its parent's completion as a continuation.

Regression for the completion-routing repro on PR #150198: use the real child
reservation, first-turn metadata writers and SessionDB, not a stubbed resolver.
"""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.prompt_refresh import stamp_prompt_epoch
from hermes_state import SessionDB
from tests.tools import test_delegate_transcript_readiness as transcript_readiness
from tools.delegate_tool import _build_child_agent
from tools.process_registry_notifications import format_process_notification
from tui_gateway import server


offline_children = transcript_readiness.offline_children


@pytest.mark.parametrize("event_type", ["completion", "async_delegation"])
@pytest.mark.parametrize("compressed", [False, True], ids=["parent", "compression-tip"])
def test_parent_completion_never_delivered_to_delegate_child(
    tmp_path, monkeypatch, offline_children, event_type, compressed
):
    db = SessionDB(db_path=tmp_path / "state.db")
    child = None
    try:
        db.create_session("parent", source="desktop")
        db.append_message("parent", "user", "Delegate an inspection")
        owner_key = "parent"
        if compressed:
            db.publish_compression_child(
                parent_session_id="parent", child_session_id="tip", source="desktop",
                messages=[{"role": "user", "content": "Continue the inspection"}],
                require_compression_lease=False,
            )
            owner_key = "tip"
        child = _build_child_agent(
            task_index=0, goal="Synthetic inspection", context=None, toolsets=["file"],
            model=None, max_iterations=2, task_count=1,
            parent_agent=transcript_readiness._parent(db, owner_key, lambda *a, **kw: None),
        )
        # On a refreshed profile this precedes _ensure_db_session. Without the
        # writer fix it freezes model_config without _delegate_from.
        stamp_prompt_epoch(child, 17)
        child._ensure_db_session()
        db.append_message(child.session_id, "assistant", "Inspection complete")

        owner = {"session_key": owner_key, "agent": None, "running": False,
                 "history_lock": threading.RLock()}
        child_ui = {"session_key": child.session_id, "agent": None, "running": False,
                    "history_lock": threading.RLock()}
        monkeypatch.setattr(server, "_sessions", {"parent-ui": owner, "child-ui": child_ui})
        # Keep the real profile-aware DB context and resolver; only select the
        # isolated store. No model turn or unrelated delivery ledger is needed.
        monkeypatch.setattr(server, "_get_db", lambda: db)
        submit = Mock(return_value=True)
        emit = Mock()
        monkeypatch.setattr(server, "_run_prompt_submit", submit)
        monkeypatch.setattr(server, "_emit", emit)
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "claim")
        monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: None)
        registry = SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda _sid: False)
        event = {"type": event_type, "session_key": "parent"}
        if event_type == "async_delegation":
            # The original tab is gone, so the durable lineage must decide who
            # receives the result; a live origin would bypass the faulty walk.
            event.update(delegation_id="parent-batch", origin_ui_session_id="closed-ui",
                         results=[{"status": "completed", "summary": "Inspection complete"}])
        else:
            event.update(session_id="proc-parent", command="inspect", exit_code=0,
                         completion_reason="exited", output="Inspection complete")

        # Let the delegate's poller see it first, as in the live incident. It
        # must neither emit a status nor inject a turn, and must retain the event.
        deferred = []
        server._notif_handle_ready(
            "child-ui", child_ui, [event], set(), registry, format_process_notification, deferred)
        submit.assert_not_called()
        emit.assert_not_called()
        assert deferred == [event]
        assert not child_ui["running"]

        # The parent (or its real compression continuation) must still receive
        # the retained event. This prevents a false fix that merely drops it.
        retained = []
        server._notif_handle_ready(
            "parent-ui", owner, deferred, set(), registry, format_process_notification, retained)
        submit.assert_called_once()
        assert submit.call_args.args[1] == "parent-ui"
        assert submit.call_args.args[2] is owner
        assert "Inspection complete" in submit.call_args.args[3]
        assert retained == []
        assert registry.completion_queue.empty()
    finally:
        if child is not None:
            child.close()
        db.close()
