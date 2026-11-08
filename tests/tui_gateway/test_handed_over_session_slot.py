"""A runtime whose slot was handed to another surface cannot start another turn on that session.

The phone/dashboard backend keeps a session runtime alive for the life of its socket. When a native
desktop resume takes the chat over (the idle holder's registry entry is yielded under the liveness
lock), that runtime must stop admitting turns instead of racing the new owner's transcript.
"""
import threading
import time

from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    YIELDABLE_SURFACES,
    active_session_liveness_guard,
    active_session_registry_snapshot,
    release_active_session,
    try_acquire_active_session,
)
from tui_gateway import server


def _runtime_record(session_key: str, *, profile_home, running: bool = False) -> dict:
    class _Agent:
        interrupted = False

        def request_interrupt(self, *args, **kwargs) -> None:
            self.interrupted = True

        def interrupt(self, *args, **kwargs) -> None:
            self.interrupted = True

        def clear_interrupt(self) -> None:
            pass

    return {
        "agent": _Agent(), "agent_ready": None, "transport": object(), "running": running,
        "last_active": time.time(), "created_at": time.time(), "source": "webui",
        "session_key": session_key, "history": [], "history_lock": threading.Lock(),
        "active_session_lease": None, "profile_home": str(profile_home),
    }


def test_a_handed_over_slot_refuses_every_later_turn(tmp_path, monkeypatch):
    key = "20261107_231452_3effe7"
    lease, refusal = try_acquire_active_session(
        session_id=key, surface="webui", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "phone-socket"})
    assert lease is not None and refusal is None, refusal
    record = _runtime_record(key, profile_home=tmp_path)
    record["active_session_lease"] = lease
    monkeypatch.setattr(server, "_sessions", {"phone": record})

    # The phone's own turn is admitted while it holds the slot.
    assert server._ensure_active_session_slot("phone", record) is None

    # The desktop's native resume yields the idle holder's slot (adopt_legacy_session does this).
    with active_session_liveness_guard(
            key, registry_home=tmp_path, yield_surfaces=YIELDABLE_SURFACES,
            holder_is_idle=lambda entry: True) as active:
        assert active is False

    refused = server._ensure_active_session_slot("phone", record)
    assert getattr(refused, "reason", None) == SESSION_NOT_OWNED
    assert record.get("active_session_lease") is None
    # Sticky for this runtime: no fresh claim, so no second writer on the adopted transcript.
    assert getattr(server._ensure_active_session_slot("phone", record), "reason", None) == SESSION_NOT_OWNED
    assert active_session_registry_snapshot(registry_home=tmp_path) == []


def test_a_released_lease_does_not_refuse_the_next_turn(tmp_path, monkeypatch):
    """Control: only a slot that ANOTHER process retired refuses, not one this runtime released."""
    key = "20261108_012937_4af988"
    lease, refusal = try_acquire_active_session(
        session_id=key, surface="tui", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "tui-runtime"})
    assert lease is not None and refusal is None, refusal
    record = _runtime_record(key, profile_home=tmp_path)
    record["active_session_lease"] = lease
    monkeypatch.setattr(server, "_sessions", {"tui": record})
    assert server._ensure_active_session_slot("tui", record) is None

    record.pop("active_session_lease")
    release_active_session(lease)
    assert lease.released
    assert server._ensure_active_session_slot("tui", record) is None
