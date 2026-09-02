"""Production-permit session identity must survive every live-session shape."""

from __future__ import annotations

import threading

import tui_gateway.server as server


def test_deferred_session_record_preserves_runtime_session_id() -> None:
    record = server._deferred_session_record(
        "runtime-approval-1",
        "stored-session-1",
        cols=80,
        cwd="/tmp",
        history=[],
        lease=None,
    )

    assert record["session_id"] == "runtime-approval-1"
    assert record["session_key"] == "stored-session-1"


def test_production_permit_response_accepts_the_live_runtime_identity(monkeypatch) -> None:
    runtime_id = "runtime-approval-response"
    session = {"session_id": runtime_id, "session_key": "stored-session-response"}
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))

    import tools.approval as approval

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        approval,
        "resolve_gateway_approval",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )

    response = server._methods["production_permit.respond"](
        "request-1",
        {
            "choice": "once",
            "request_id": "approval-1",
            "session_id": runtime_id,
            "witness": {"signed": True},
        },
    )

    assert response["result"] == {"resolved": True}
    assert calls == [(("stored-session-response", "once"), {"request_id": "approval-1", "witness": {"signed": True}})]


def test_initialized_session_record_preserves_runtime_session_id(monkeypatch) -> None:
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda _sid, _session: threading.Event())
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    with server._sessions_lock:
        server._sessions.clear()
    try:
        server._init_session("runtime-approval-2", "stored-session-2", object(), [])
        assert server._sessions["runtime-approval-2"]["session_id"] == "runtime-approval-2"
        assert server._sessions["runtime-approval-2"]["session_key"] == "stored-session-2"
    finally:
        with server._sessions_lock:
            server._sessions.clear()
