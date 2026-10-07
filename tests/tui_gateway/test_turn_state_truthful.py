"""Running state and start frames must describe an admitted live turn (#89978, #111155)."""

import contextlib
import threading
import types

import pytest

from tui_gateway import server


def _session(**extra):
    return dict(agent=types.SimpleNamespace(clear_interrupt=lambda: None), history=[],
                history_lock=threading.Lock(), running=True, _run_turn=1, **extra)


def test_payload_reconciles_only_the_current_started_dead_worker(monkeypatch):
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    monkeypatch.setattr(server, "_fallback_session_info", lambda _: {})
    release = threading.Event()
    alive = threading.Thread(target=lambda: release.wait(timeout=5))
    dead = threading.Thread(target=lambda: None)
    alive.start()
    dead.start()
    dead.join(timeout=2)
    dead._hermes_run_turn = alive._hermes_run_turn = 1
    unstarted = threading.Thread(target=lambda: None)
    unstarted._hermes_run_turn = 1
    try:
        for worker, turn, host, busy in [(alive, 1, False, True), (dead, 2, False, True),
                                       (unstarted, 1, False, True), (dead, 1, True, True),
                                       (dead, 1, False, False)]:
            session = _session(_run_thread=worker, _compute_host_active=host,
                               inflight_turn={"user": "turn", "assistant": "partial", "status": "streaming"})
            session["_run_turn"] = turn
            payload = server._live_session_payload("truthful", session, omit_messages=True)
            assert payload["running"] is busy
            assert ("inflight" in payload) is busy
            assert bool(session["inflight_turn"]) is busy
    finally:
        release.set()
        alive.join(timeout=2)


def test_admission_counter_and_worker_stamp_preserve_successors_and_unknown_liveness(monkeypatch):
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _: False)
    session = _session()
    session["running"] = False
    assert server._notif_claim_turn(session)
    assert session["_run_turn"] == 2
    ran = threading.Event()
    worker = server._start_session_work(ran.set, name="truthful-stamp", session=session)
    worker.join(timeout=2)
    assert ran.is_set() and worker._hermes_run_turn == 2
    for successor in [False, True]:
        def liveness():
            if successor:
                session["_run_thread"] = worker
                return False
            raise RuntimeError("unknown liveness")
        session["_run_thread"] = types.SimpleNamespace(
            _hermes_run_turn=2, join=lambda **_: None, is_alive=liveness)
        assert server._reconcile_finished_run_thread(session) is False
        assert session["running"] is True


@pytest.mark.parametrize("start_emit_fails", [False, True])
def test_start_frames_follow_admission_once_and_keep_diagnostic_visibility(monkeypatch, start_emit_fails):
    from agent import notification_presentation

    events = []
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _: True)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_: None)
    monkeypatch.setattr(server, "_record_turn_marker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_routing_provenance_db", lambda _: contextlib.nullcontext(None))
    for name in ["_release_turn_scopes", "_post_turn_housekeeping", "_emit_settled_session_info"]:
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    def emit(event, *a, **k):
        events.append(event)
        if start_emit_fails and event == "message.start":
            raise RuntimeError("transport unavailable")

    monkeypatch.setattr(server, "_emit", emit)
    monkeypatch.setattr(server, "_prepare_turn_input", lambda *a: events.append("prepared"))
    monkeypatch.setattr(notification_presentation, "notification_config_snapshot",
                        lambda: {"display": {"suppress_warning_notifications": True}})
    for dispatch in ["direct", "followup", "notification"]:
        for refusal in ["closing", "replaced", "retiring", None]:
            for diagnostic in [False, True]:
                events.clear()
                session = _session(_closing=refusal == "closing")
                if refusal == "replaced":
                    server._sessions["truthful"] = {}
                else:
                    server._sessions.clear()
                metadata = {"notification_category": "diagnostic"} if diagnostic else {}
                with monkeypatch.context() as patch:
                    if refusal == "retiring":
                        patch.setattr(server, "_start_session_work", lambda *a, **k: None)
                    if dispatch == "followup":
                        server._dispatch_followup_turn("rid", "truthful", session, "turn", "test")
                    elif dispatch == "notification":
                        server._notif_submit("rid", "truthful", session, "turn", "test", display_metadata=metadata)
                    else:
                        server._run_prompt_submit("rid", "truthful", session, "turn", display_metadata=metadata)
                    if refusal is None:
                        session["_run_thread"].join(timeout=2)
                visible = refusal is None and (not diagnostic or dispatch == "followup")
                assert events.count("message.start") == int(visible)
                if visible:
                    assert events[:2] == ["message.start", "prepared"]
                assert session["running"] is False


@pytest.mark.parametrize("source", ["queued_drain", "bot_live"])
def test_new_claim_stays_busy_before_successor_publication(monkeypatch, tmp_path, source):
    from tools import bot_live_delivery

    previous = threading.Thread(target=lambda: None)
    previous.start()
    previous.join(timeout=2)
    previous._hermes_run_turn = 1
    session = _session(_run_thread=previous, session_key="truthful-key")
    session["running"] = False
    observed = []
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _: False)

    def submit(_rid, _sid, claimed, _text, **_kwargs):
        with claimed["history_lock"]:
            observed.append((claimed["_run_turn"], server._reconcile_finished_run_thread(claimed),
                             claimed["running"]))
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    if source == "queued_drain":
        session["queued_prompt"] = {"text": "queued turn", "transport": None}
        assert server._drain_queued_prompt("rid", "truthful", session)
        assert session["queued_prompt"] is None
    else:
        session["active_session_lease"] = types.SimpleNamespace(lease_id="test-lease", released=False)
        owner = {"lease_id": "test-lease", "live_session_id": "truthful", "session_id": "truthful-key"}
        monkeypatch.setattr(server, "_session_home", lambda _: tmp_path)
        monkeypatch.setattr(bot_live_delivery, "has_mailbox", lambda _: True)
        monkeypatch.setattr(bot_live_delivery, "find_canonical_live_owner", lambda _: owner)
        monkeypatch.setattr(bot_live_delivery, "claim_pending_delivery",
                            lambda *_: {"id": "test-delivery", "message": "bot turn"})
        assert server._poll_bot_live_delivery_once("truthful", session)
    assert observed == [(2, False, True)]
    assert session["running"] is True
    assert session["_run_thread"] is previous
