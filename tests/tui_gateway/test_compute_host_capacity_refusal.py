"""Direct compute-host admission refuses safely before dispatch or busy mutation."""

import threading
from unittest.mock import Mock

import pytest

from hermes_cli import active_sessions
from tui_gateway import server


@pytest.fixture
def idle_compute_session(monkeypatch):
    cfg = {"max_concurrent_sessions": 1}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_canonical_ledger_refusal", lambda session: None)
    executor = Mock()
    monkeypatch.setattr(server, "_get_compute_host_supervisor", Mock(
        return_value=Mock(submit_turn=executor)))
    session = dict(
        session_key="direct-compute-owner", source="desktop",
        history_lock=threading.RLock(), running=False,
        _compute_host_active=False, _compute_host_submitting=False,
        _gateway_busy_token=None, inflight_turn=None, history=[],
    )
    assert server._ensure_active_session_slot("direct-compute-ui", session) is None
    try:
        yield cfg, session, executor
    finally:
        server._release_active_session_slot(session)


def test_direct_compute_submit_at_capacity_returns_typed_refusal(idle_compute_session):
    cfg, session, executor = idle_compute_session
    blocker, refusal = active_sessions.try_acquire_active_session(
        session_id="telegram-turn", surface="gateway:telegram", config=cfg)
    assert refusal is None and blocker is not None
    owner = session["active_session_lease"]
    before_session = dict(session)
    before_registry = owner.state_path.read_bytes()
    try:
        response = server._submit_prompt_to_compute_host(
            "direct-rid", "direct-compute-ui", session, "synthetic turn")

        assert response["id"] == "direct-rid"
        assert response["error"]["code"] == 4090
        assert response["error"]["data"] == {
            "reason": active_sessions.MAX_CONCURRENT_SESSIONS}
        assert "gateway:telegram" in response["error"]["message"]
        executor.assert_not_called()
        assert session == before_session
        assert owner.state_path.read_bytes() == before_registry
    finally:
        blocker.release()


def test_direct_compute_submit_registry_error_fails_closed(
    idle_compute_session, monkeypatch,
):
    _, session, executor = idle_compute_session
    owner = session["active_session_lease"]
    before_session = dict(session)
    before_registry = owner.state_path.read_bytes()
    # Ownership uncertainty intentionally sacrifices availability: no turn may
    # run when the existing owner's registry state cannot be proved.
    with monkeypatch.context() as patch:
        patch.setattr(active_sessions, "_read_entries", Mock(
            side_effect=active_sessions.ActiveSessionRegistryError("registry unavailable")))
        response = server._submit_prompt_to_compute_host(
            "unavailable-rid", "direct-compute-ui", session, "synthetic turn")

    assert response["id"] == "unavailable-rid"
    assert response["error"] == {
        "code": 4090,
        "message": "Hermes could not safely reserve this session. Try again.",
    }
    executor.assert_not_called()
    assert session == before_session
    assert owner.state_path.read_bytes() == before_registry
