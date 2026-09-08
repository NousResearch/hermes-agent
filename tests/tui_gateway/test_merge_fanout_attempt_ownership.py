"""Fanout membership and orphan cancellation share exact native turn ownership."""

import threading
from types import SimpleNamespace

from tui_gateway import server


def test_facade_binds_current_merge_helpers_to_its_own_namespace():
    from tui_gateway import methods_session, session_lifecycle, session_transports

    for module, names in (
        (session_lifecycle, ("_reattach_refusal", "_rebind_live_transport", "_schedule_ws_orphan_reap")),
        (session_transports, ("_attach_session_transport",)),
        (methods_session, ("_resume_reuse_live", "_resume_reuse_live_locked")),
    ):
        for name in names:
            bound = getattr(server, name)
            assert bound.__code__ is getattr(module, name).__code__
            assert bound.__globals__ is vars(server)
    assert not hasattr(server, "_reattach_ws_session")


def test_last_peer_orphan_interrupt_preserves_attempt_and_runs_without_locks(monkeypatch):
    class Peer:
        def write(self, frame):
            return True

    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    first, second = Peer(), Peer()
    session = dict(
        transport=first, history_lock=threading.Lock(), running=True,
        session_key="merge-attempt", history=[], attached_images=[])
    monkeypatch.setattr(server, "_sessions", {"sid": session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 10)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *args: False)
    monkeypatch.setattr(server, "_ws_orphan_turn_activity_is_fresh", lambda *args: False)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *args: False)
    monkeypatch.setattr(server, "_clear_pending", lambda *args: None)
    with session["history_lock"]:
        attempt = server._new_turn_attempt(session, {"task_id": "task", "execution_generation": 7})
        session["_turn_attempt_live"] = attempt
        server._rebind_live_transport("sid", session, second)

    interrupted = []

    def hard_interrupt():
        assert not session["history_lock"].locked()
        assert not server._session_resume_lock.locked()
        unlocked = threading.Event()

        def probe_registry_lock():
            with server._sessions_lock:
                unlocked.set()

        probe = threading.Thread(target=probe_registry_lock, daemon=True)
        probe.start()
        assert unlocked.wait(3)
        probe.join(timeout=3)
        assert session["_turn_interrupt_claims"][0][0] == attempt
        interrupted.append(attempt)

    session["agent"] = SimpleNamespace(hard_interrupt=hard_interrupt)
    assert server._close_sessions_for_transport(first) == (0, 0)
    assert server._session_transport_contains(session, second)
    assert not timers
    assert session["_pending_attempt_outcomes"] == {attempt: None}

    assert server._close_sessions_for_transport(second) == (0, 1)
    timers[0].callback()
    assert interrupted == [attempt]
    assert session["running"] is True  # the accepted runner still owns settlement
    assert session["_turn_cancel_attempt"] == attempt
    assert session["_pending_attempt_outcomes"] == {attempt: "cancelled"}
    assert not session["_turn_interrupt_claims"]
    assert server._pending_ws_reaps["sid"] is timers[-1]
    assert len(timers) == 2
