"""Turn settlement survives registry outages without releasing exclusive ownership."""

import logging
import threading
import types

import pytest

from hermes_cli import active_sessions
from tests.tui_gateway.test_hosted_room_turn_slot_release import _session, _turn_env
from tui_gateway import server


CONFIG = {"max_concurrent_sessions": 1}


@pytest.mark.parametrize("failure_type", [OSError, active_sessions.ActiveSessionRegistryError])
@pytest.mark.parametrize("boundary", ["notification", "prompt-worker"])
def test_settlement_outage_recovers_on_backend_heartbeat(
    tmp_path, monkeypatch, caplog, boundary, failure_type,
):
    home = tmp_path / "owning-profile"
    lease, refusal = active_sessions.try_acquire_active_session(
        session_id="settled", surface="desktop", config=CONFIG, registry_home=home,
        metadata={"live_session_id": "original-writer"}, track_liveness=True,
    )
    assert refusal is None and lease is not None
    session = _session("desktop", lease)
    session["session_key"] = "settled"
    session["_auto_continue_scheduled"] = True
    events = []
    drained = []
    next_lease = None

    def unavailable(*args, **kwargs):
        raise failure_type("sensitive registry payload /private/profile/credential")

    try:
        with monkeypatch.context() as outage:
            if boundary == "notification":
                outage.setattr(active_sessions, "atomic_json_write", unavailable)
                with caplog.at_level(logging.WARNING):
                    server._notif_release_turn(session)
            else:
                # Keep the synchronous worker double local to the gateway namespace.
                outage.setattr(server, "threading", types.SimpleNamespace(**vars(threading)))
                _turn_env(outage, tmp_path)

                def inline_work(target, **kwargs):
                    target()
                    return object()

                outage.setattr(server, "_start_session_work", inline_work)
                outage.setattr(server, "_emit", lambda event, sid, payload=None: events.append(event))
                outage.setattr(server, "_drain_queued_prompt", lambda *args: drained.append(True) or True)

                def finish_turn(*args, **kwargs):
                    # Admission has already persisted; only settlement loses its write.
                    outage.setattr(active_sessions, "atomic_json_write", unavailable)
                    return {"final_response": "done"}

                session["agent"].run_conversation = finish_turn
                with caplog.at_level(logging.WARNING):
                    assert server._run_prompt_submit("rid", "settled-ui", session, "hello") is True
                assert "_auto_continue_scheduled" not in session
                assert "session.info" in events
                assert drained == [True], "settlement must still reach queued-prompt followups"

        assert session["running"] is False
        assert session["active_session_lease"] is lease
        assert not lease.released
        assert lease.capacity_active is False
        before = active_sessions.active_session_registry_snapshot(home)
        assert len(before) == 1 and before[0]["capacity_active"] is True
        assert "capacity" in caplog.text.lower()
        assert failure_type.__name__ in caplog.text
        assert "sensitive registry payload" not in caplog.text
        assert "/private/profile/credential" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

        # Exercise the backend's real heartbeat, not a direct capacity-state retry.
        monkeypatch.setattr(server, "_sessions", {"settled-ui": session})
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "unrelated-profile"))
        server._reclaim_orphaned_leases()
        after = active_sessions.active_session_registry_snapshot(home)
        assert len(after) == 1
        assert after[0]["lease_id"] == lease.lease_id
        assert after[0]["capacity_active"] is False
        next_lease, refusal = active_sessions.try_acquire_active_session(
            session_id="unrelated-turn", surface="cli", config=CONFIG, registry_home=home,
        )
        assert refusal is None and next_lease is not None
        foreign, refusal = active_sessions.try_acquire_active_session(
            session_id="settled", surface="cli", config={}, registry_home=home,
            metadata={"live_session_id": "foreign-writer"}, capacity_active=False,
        )
        assert foreign is None
        assert isinstance(refusal, active_sessions.ActiveSessionRefusal)
        assert refusal.reason == active_sessions.SESSION_NOT_OWNED
    finally:
        if next_lease is not None:
            next_lease.release()
        lease.release()


@pytest.mark.parametrize("failure_type", [TypeError, ValueError, AttributeError])
def test_settlement_does_not_hide_unrelated_lease_defects(failure_type):
    def broken(active):
        raise failure_type("unrelated lease defect")

    session = {"history_lock": threading.RLock(), "running": True, "source": "desktop",
               "active_session_lease": types.SimpleNamespace(set_capacity_active=broken)}
    with pytest.raises(failure_type, match="unrelated lease defect"):
        server._notif_release_turn(session)


def test_bot_room_settlement_still_fully_releases_real_lease(tmp_path, monkeypatch):
    lease, refusal = active_sessions.try_acquire_active_session(
        session_id="room-member", surface="bot_room", config=CONFIG, registry_home=tmp_path,
    )
    assert refusal is None and lease is not None
    session = _session("bot_room", lease)

    def must_not_deactivate(*args, **kwargs):
        pytest.fail("bot_room must release ownership, not merely deactivate capacity")

    monkeypatch.setattr(lease, "set_capacity_active", must_not_deactivate)
    try:
        server._notif_release_turn(session)
        assert session["running"] is False
        assert "active_session_lease" not in session
        assert lease.released
        assert active_sessions.active_session_registry_snapshot(tmp_path) == []
    finally:
        lease.release()
