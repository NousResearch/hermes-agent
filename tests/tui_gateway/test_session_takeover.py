from __future__ import annotations

from pathlib import Path

from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    SESSION_TAKEN_OVER,
    active_session_lease_is_current,
    active_session_registry_snapshot,
    take_over_active_session,
    try_acquire_active_session,
)
from tui_gateway import server


def _claim(home: Path, live_id: str):
    return try_acquire_active_session(
        session_id="shared", surface="desktop", config={}, registry_home=home,
        metadata={"live_session_id": live_id}, track_liveness=True)


def test_takeover_modes_and_fence_old_writer(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    old, refusal = _claim(home, "old-runtime")
    assert old is not None and refusal is None

    blocked, refusal = take_over_active_session(
        session_id="shared", surface="tui", config={"session": {"takeover": "off"}},
        holder_has_live_turn=False, registry_home=home,
        metadata={"live_session_id": "new-runtime"})
    assert blocked is None and getattr(refusal, "reason", None) == SESSION_NOT_OWNED

    blocked, refusal = take_over_active_session(
        session_id="shared", surface="tui", config={"session": {"takeover": "idle"}},
        holder_has_live_turn=True, registry_home=home,
        metadata={"live_session_id": "new-runtime"})
    assert blocked is None and getattr(refusal, "reason", None) == SESSION_NOT_OWNED

    new, refusal = take_over_active_session(
        session_id="shared", surface="tui", config={"session": {"takeover": "idle"}},
        holder_has_live_turn=False, registry_home=home,
        metadata={"live_session_id": "new-runtime"})
    assert new is not None and refusal is None
    assert active_session_lease_is_current(old) is False
    entry = active_session_registry_snapshot(registry_home=home)[0]
    assert entry["lease_id"] == new.lease_id
    assert entry["metadata"]["taken_over_from"]["lease_id"] == old.lease_id

    session = {"active_session_lease": old, "session_key": "shared"}
    refused = server._ensure_active_session_slot("old-runtime", session)
    assert getattr(refused, "reason", None) == SESSION_TAKEN_OVER
    assert "active_session_lease" not in session
    assert session["_lease_taken_over"] is True
    assert active_session_lease_is_current(new) is True


def test_always_takeover_ignores_live_turn(tmp_path: Path) -> None:
    home = tmp_path / "home"
    old, _ = _claim(home, "old-runtime")
    new, refusal = take_over_active_session(
        session_id="shared", surface="tui", config={"session": {"takeover": "always"}},
        holder_has_live_turn=True, registry_home=home,
        metadata={"live_session_id": "new-runtime"})
    assert new is not None and refusal is None
    assert active_session_lease_is_current(old) is False
