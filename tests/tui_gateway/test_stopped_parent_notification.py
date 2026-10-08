"""A user stop holds the session. A later completion must not start a turn or ack delivery."""

import contextlib
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _stopped_session(**extra):
    session = {
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "parent-key",
        "_turn_cancel_requested": True,
        "agent": MagicMock(),
    }
    session.update(extra)
    return session


@pytest.mark.parametrize("rid,display_kind", [
    ("__notif__1", "async_delegation_complete"),
    ("__notif__2", "process_complete"),
    ("__notif__3", None),
    ("user-looking", "hidden"),
])
def test_synthetic_submit_returns_false_without_admit(monkeypatch, rid, display_kind):
    from tui_gateway import server

    session = _stopped_session()
    admitted = []

    def admit(*_args, **_kwargs):
        admitted.append(True)
        raise AssertionError("admission cleared the stop")

    monkeypatch.setattr(server, "_admit_prompt_turn", admit)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: admitted.append("ensure"))
    started = server._run_prompt_submit(rid, "sid", session, "child done", display_kind=display_kind)
    assert started is False
    assert admitted == []
    assert session["_turn_cancel_requested"] is True
    assert session["running"] is False
    session["agent"].clear_interrupt.assert_not_called()


def test_hold_blocks_synthetic_submit_after_cancel_was_cleared(monkeypatch):
    """_lock_in_submit_turn clears cancel before the prompt runs. The hold must still refuse."""
    from tui_gateway import server

    session = _stopped_session(_turn_cancel_requested=False, _delegation_hold=True)

    def admit(*_args, **_kwargs):
        raise AssertionError("admission ran while the hold was set")

    monkeypatch.setattr(server, "_admit_prompt_turn", admit)
    started = server._run_prompt_submit("__notif__hold", "sid", session, "child done", display_kind="async_delegation_complete")
    assert started is False
    assert session["_turn_cancel_requested"] is False
    assert session["_delegation_hold"] is True
    assert session["running"] is False
    session["agent"].clear_interrupt.assert_not_called()


def test_live_thread_keeps_running_when_a_synthetic_turn_is_refused(monkeypatch):
    from tui_gateway import server

    class _Alive:
        def is_alive(self):
            return True

    session = _stopped_session(_run_thread=_Alive())
    monkeypatch.setattr(server, "_admit_prompt_turn", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("admitted")))
    assert server._run_prompt_submit("__heartbeat__1", "sid", session, "tick") is False
    assert session["running"] is True


def test_real_user_message_releases_hold_and_requeues(monkeypatch):
    from tui_gateway import server

    held = {"type": "async_delegation", "delegation_id": "deleg_held"}
    completion_queue = queue.Queue()
    session = {
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "parent-key",
        "_delegation_hold": True,
        "_delegation_held_events": [held],
        "agent": MagicMock(),
    }
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_admit_prompt_turn", lambda *a, **k: ([], session["agent"]))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_sessions_lock", threading.Lock())
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_start_session_work", lambda *a, **k: object())
    monkeypatch.setattr(server, "_routing_provenance_db", lambda session: contextlib.nullcontext(None))
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        SimpleNamespace(completion_queue=completion_queue),
    )
    started = server._run_prompt_submit("user-1", "sid", session, "continue")
    assert started is True
    assert "_delegation_hold" not in session
    assert "_delegation_held_events" not in session
    assert completion_queue.get_nowait() == held


def test_refused_real_turn_keeps_the_hold(monkeypatch):
    from tui_gateway import server

    held = {"type": "async_delegation", "delegation_id": "deleg_kept"}
    session = _stopped_session(_turn_cancel_requested=False, _delegation_hold=True, _delegation_held_events=[held])
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_admit_prompt_turn", lambda *a, **k: None)
    assert server._run_prompt_submit("user-2", "sid", session, "continue") is False
    assert session["_delegation_hold"] is True
    assert session["_delegation_held_events"] == [held]


def test_stopped_session_holds_process_completion_without_a_turn():
    from tui_gateway import server

    session = _stopped_session(running=False)
    event = {"type": "completion", "session_id": "proc_1", "command": "sleep 900"}
    registry = SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda _sid: False)
    formatted = []
    ok = server._notif_handle_event(
        "sid", session, event, set(), registry, lambda item: formatted.append(item) or "text", None, owned=True)
    assert ok is True
    assert formatted == []
    assert session["_delegation_hold"] is True
    assert session["_delegation_held_events"] == [event]
    assert session["_turn_cancel_requested"] is True
    assert registry.completion_queue.empty()


def _delivery_spies(monkeypatch, claim_id):
    calls = {"complete": 0, "release": 0, "defer": 0}
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_a, **_k: claim_id)
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda *_a, **_k: calls.__setitem__("complete", calls["complete"] + 1),
    )
    monkeypatch.setattr(
        "tools.async_delegation.release_event_delivery",
        lambda *_a, **_k: calls.__setitem__("release", calls["release"] + 1),
    )
    monkeypatch.setattr(
        "tools.async_delegation.defer_event_delivery",
        lambda *_a, **_k: calls.__setitem__("defer", calls["defer"] + 1),
    )
    return calls


def test_hold_instead_of_turn_defers_before_submit(monkeypatch):
    from tui_gateway import server

    session = _stopped_session()
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_x",
        "status": "completed",
        "session_key": "parent-key",
    }
    calls = _delivery_spies(monkeypatch, "claim-1")
    submitted = []
    monkeypatch.setattr(server, "_notif_submit", lambda *a, **k: submitted.append(a) or False)
    server._notif_dispatch_event("sid", session, event, "done")
    assert submitted == []
    assert calls == {"complete": 0, "release": 0, "defer": 1}
    assert session["_delegation_hold"] is True
    assert session["_delegation_held_events"] == [event]
    assert session["running"] is False


def test_submit_false_defers_and_releases_the_turn(monkeypatch):
    from tui_gateway import server

    session = _stopped_session(_turn_cancel_requested=False)
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_y",
        "status": "completed",
        "session_key": "parent-key",
    }
    calls = _delivery_spies(monkeypatch, "claim-2")
    monkeypatch.setattr(server, "_notif_submit", lambda *_a, **_k: False)
    server._notif_dispatch_event("sid", session, event, "done")
    assert calls["complete"] == 0 and calls["release"] == 0 and calls["defer"] == 1
    assert session["_delegation_held_events"] == [event]
    assert session["running"] is False


def test_interrupted_completion_holds_without_a_prior_stop(monkeypatch):
    from tui_gateway import server

    session = _stopped_session(_turn_cancel_requested=False, running=True)
    event = {"type": "async_delegation", "delegation_id": "deleg_z", "status": "interrupted"}
    calls = _delivery_spies(monkeypatch, "claim-3")
    monkeypatch.setattr(server, "_notif_submit", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("submitted")))
    server._notif_dispatch_event("sid", session, event, "stopped child")
    assert calls["defer"] == 1 and calls["complete"] == 0
    assert session["_delegation_held_events"] == [event]


def test_interrupted_completion_dispatches_after_admitted_user_turn(monkeypatch):
    from tui_gateway import server
    from tools.delegate_resume_policy import release_delegation_hold

    event = {"type": "async_delegation", "delegation_id": "deleg_resume", "status": "interrupted"}
    session = _stopped_session(_turn_cancel_requested=False, running=True,
                               _delegation_hold=True, _delegation_held_events=[event])
    calls = _delivery_spies(monkeypatch, "claim-resume")
    submitted = []
    monkeypatch.setattr(server, "_notif_submit", lambda *a, **k: submitted.append(a) or True)
    assert release_delegation_hold(session) == [event]
    server._notif_dispatch_event("sid", session, event, "stopped child")
    assert len(submitted) == 1
    assert calls["defer"] == 0
    assert "_delegation_hold" not in session


def test_completion_batch_that_does_not_start_stays_pending(monkeypatch):
    from tui_gateway import server

    event = {"type": "async_delegation", "delegation_id": "deleg_batch", "status": "completed"}
    calls = _delivery_spies(monkeypatch, "claim-batch")
    monkeypatch.setattr(server, "_notif_submit", lambda *_a, **_k: False)
    session = _stopped_session(running=False, _turn_cancel_requested=False)
    server._notif_dispatch_completions(
        "sid", session, [(event, "done")],
        SimpleNamespace(completion_queue=queue.Queue(), is_completion_consumed=lambda _sid: False),
        None,
    )
    assert calls == {"complete": 0, "release": 0, "defer": 1}
    assert session["running"] is False
    assert session["_delegation_held_events"] == [event]


def test_dispatch_exception_still_releases_the_claim(monkeypatch):
    from tui_gateway import server

    session = _stopped_session(_turn_cancel_requested=False)
    event = {"type": "async_delegation", "delegation_id": "deleg_boom", "status": "completed"}
    calls = _delivery_spies(monkeypatch, "claim-boom")

    def _boom(*_a, **_k):
        raise RuntimeError("no free worker")

    monkeypatch.setattr(server, "_notif_submit", _boom)
    server._notif_dispatch_event("sid", session, event, "done")
    assert calls == {"complete": 0, "release": 1, "defer": 0}
