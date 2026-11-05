"""Refused producer turns must not hot-loop against the canonical ledger (#140385)."""

import contextlib
import queue
import threading
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tui_gateway import server


@pytest.mark.parametrize("kind", ["completion", "watch_match"])
def test_canonical_busy_notification_backs_off_without_consuming_work(monkeypatch, tmp_path, kind):
    with contextlib.closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        key = "notification-owner"
        db.create_session(key, "desktop")
        epoch = begin_runtime_epoch(db, instance_id="gateway")
        admit_session_input(db, epoch=epoch, principal_id="owner", session_id=key,
                            request_id="busy", payload={"text": "running"})
        claim_session_input(db, epoch=epoch, session_id=key)
        monkeypatch.setattr(server, "_session_db", lambda session: contextlib.nullcontext(db))
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: True)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
        waits = []
        monkeypatch.setattr(server, "time", SimpleNamespace(sleep=waits.append, time=lambda: 1))
        session = {"history_lock": threading.RLock(), "session_key": key, "running": False}
        registry = SimpleNamespace(completion_queue=queue.Queue())
        monkeypatch.setattr("tools.process_registry.process_registry", registry)
        monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *a: "")
        completed = []
        monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *a: completed.append(a))
        monkeypatch.setattr("tools.process_registry_notifications.ProcessNotificationBatch.render", lambda *a: "finished")
        monkeypatch.setattr("tools.process_registry_notifications.ProcessNotificationBatch.display_text", lambda *a: "finished")
        event = {"type": kind, "session_id": "proc", "session_key": key}

        def dispatch():
            if kind == "completion":
                server._notif_dispatch_completions("ui", session, [(event, "finished")], registry, None)
            else:
                assert server._notif_claim_turn(session)
                server._notif_dispatch_event("ui", session, event, "finished")

        dispatch()
        assert session["running"] is False
        assert db.get_messages_as_conversation(key) == []
        assert waits and waits[0] > 0, "refused notification must yield before its caller requeues it"
        assert not emitted, "a retained notification must not paint a failed user turn"
        assert registry.completion_queue.get_nowait() is event
        assert not completed

        # Still refused (the gateway keeps owning the chat): each retry waits twice as long, to a cap.
        for _ in range(8):
            dispatch()
            assert registry.completion_queue.get_nowait() is event
        assert waits == [0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0, 5.0], waits
        assert waits[-1] == server._NOTIF_REFUSAL_BACKOFF_MAX_S

        # Human submissions still report ownership refusals.
        assert server._run_prompt_submit("human", "ui", session, "hello") is False
        assert emitted[0][0] == "error"

        # A later accepted notification settles once and never takes the refusal delay.
        waits.clear()
        monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: True)
        dispatch()
        assert len(completed) == 1
        assert registry.completion_queue.empty()
        assert not waits
        assert "_notif_refusal_delay" not in session  # the next refusal starts from the floor again


def test_a_refusal_that_is_not_requeued_does_not_delay_the_poller(monkeypatch):
    """Kanban and /loop callers keep their notice themselves; the poller thread must not sleep for them
    (it also drives the kanban and loop polls)."""
    waits = []
    monkeypatch.setattr(server, "time", SimpleNamespace(sleep=waits.append, time=lambda: 1))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: False)
    session = {"history_lock": threading.RLock(), "running": True}
    assert server._notif_submit("r", "ui", session, "tick", "loop wakeup send failed") is False
    assert session["running"] is False and not waits
