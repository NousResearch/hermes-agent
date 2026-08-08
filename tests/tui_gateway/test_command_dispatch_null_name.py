"""command.dispatch / slash.exec must tolerate JSON-null string fields.

``dict.get("name", "")`` returns None when the key is present with a null
value. A bare ``.lstrip`` / ``.strip`` then raises AttributeError.

``command.dispatch`` runs inline (not in ``_LONG_HANDLERS``), so that crash
can tear down the TUI gateway stdin/WS reader loop. ``slash.exec`` is
pool-wrapped but still returns a cleaner empty-command error when coerced.
"""

from __future__ import annotations

from tui_gateway import server


def test_command_dispatch_tolerates_null_name(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": {}})
    monkeypatch.setattr(server, "_resolve_name", lambda n: n)

    resp = server.dispatch(
        {
            "id": "1",
            "method": "command.dispatch",
            "params": {"name": None, "arg": None, "session_id": "missing"},
        }
    )
    # Must not raise — returns a structured unknown-command / not-found error.
    assert isinstance(resp, dict)
    assert "error" in resp or "result" in resp


def test_command_dispatch_null_name_does_not_crash_inline():
    """Regression: AttributeError must not escape dispatch() for null name."""
    try:
        resp = server.dispatch(
            {
                "id": "2",
                "method": "command.dispatch",
                "params": {"name": None},
            }
        )
    except AttributeError as exc:
        raise AssertionError(f"null name crashed inline dispatch: {exc}") from exc
    assert isinstance(resp, dict)


def test_slash_exec_null_command_returns_empty_error(monkeypatch):
    sid = "slash-null"
    server._sessions[sid] = {"session_key": sid, "agent": None}
    try:
        resp = server.handle_request(
            {
                "id": "3",
                "method": "slash.exec",
                "params": {"session_id": sid, "command": None},
            }
        )
        assert resp["error"]["code"] == 4004
        assert "empty" in resp["error"]["message"].lower()
    finally:
        server._sessions.pop(sid, None)
