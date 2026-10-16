"""``session.delete``: argument checks, the live-session refusal, the profile's db and sessions dir, and
the onboarding run a deleted chat ends first (moved out of test_tui_gateway_server.py)."""

from pathlib import Path

from tui_gateway import server


class _NoOnboardingRun:
    """A fake state DB whose sessions carry no onboarding run (session.delete reads it first)."""

    def get_session_model_config_value(self, *_args):
        return None

    def resolve_resume_session_id(self, *_args):
        return None


def test_session_delete_requires_session_id(monkeypatch):
    """Empty / missing session_id is a 4006 client error (no DB call)."""
    called: list[tuple] = []

    class _DB(_NoOnboardingRun):
        def delete_session(self, *a, **kw):
            called.append((a, kw))
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request({"id": "1", "method": "session.delete", "params": {}})
    assert "error" in resp
    assert resp["error"]["code"] == 4006
    assert called == []


def test_session_delete_returns_db_unavailable_when_no_db(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_db_error", "locked")

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "abc"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036
    assert resp["error"]["data"]["code"] == "storage_locked"


def test_session_delete_refuses_active_session(monkeypatch):
    """Cannot delete a session currently bound to a live TUI session."""
    called: list[str] = []

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            called.append(sid)
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setitem(server._sessions, "live", {"session_key": "key-live"})
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.delete",
                "params": {"session_id": "key-live"},
            }
        )
    finally:
        server._sessions.pop("live", None)

    assert "error" in resp
    assert resp["error"]["code"] == 4023
    assert called == [], "delete_session must not be called for active sessions"


def test_session_delete_fails_closed_when_active_snapshot_raises(monkeypatch):
    """Concurrent ``_sessions`` mutation from another RPC thread can raise
    ``RuntimeError: dictionary changed size during iteration``.  When the
    handler can't enumerate active sessions safely it must refuse the
    delete (fail closed) rather than fall through and allow it."""

    class _DB(_NoOnboardingRun):
        def delete_session(self, *a, **kw):
            raise AssertionError("delete must not run when active snapshot fails")

    class _ExplodingDict:
        def values(self):
            raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_sessions", _ExplodingDict())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "x"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036


def test_session_delete_returns_4007_when_missing(monkeypatch):
    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "ghost"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 4007


def test_session_delete_propagates_db_exception(monkeypatch):
    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            raise RuntimeError("disk full")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "x"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036
    assert "disk full" in resp["error"]["message"]


def test_session_delete_success_returns_deleted_id(monkeypatch):
    """Happy path — DB delete succeeds, response carries the deleted id
    and the on-disk sessions dir is forwarded so transcript files get
    cleaned up alongside the row."""
    captured: dict = {}

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            captured["sid"] = sid
            captured["sessions_dir"] = sessions_dir
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "old-1"}}
    )

    assert "result" in resp, resp
    assert resp["result"] == {"deleted": "old-1"}
    assert captured["sid"] == "old-1"
    # sessions_dir must be forwarded so transcript files get cleaned up
    # too — not just the SQLite row.  The autouse _isolate_hermes_home
    # fixture pins HERMES_HOME to a temp dir; the handler should append
    # /sessions to it.
    assert captured["sessions_dir"] is not None
    assert str(captured["sessions_dir"]).endswith("sessions")




# --------------------------------------------------------------------------
# session.* profile scoping (app-global remote mode) — #62503
# --------------------------------------------------------------------------


def test_session_delete_honors_params_profile_sessions_dir(monkeypatch, tmp_path, route_profiles):
    """Issue #62503: delete must target the profile state.db + sessions dir."""
    profile_home = tmp_path / "profiles" / "mlperf"
    (profile_home / "sessions").mkdir(parents=True)
    captured: dict = {}

    class ProfileDB(_NoOnboardingRun):
        def __init__(self, db_path=None, **_kwargs):
            captured["db_path"] = db_path

        def delete_session(self, sid, sessions_dir=None, **_kw):
            captured["sid"] = sid
            captured["sessions_dir"] = sessions_dir
            return True

        def close(self):
            captured["closed"] = True

    route_profiles(server, lambda p: profile_home if p == "mlperf" else None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.delete",
            "params": {"session_id": "old-ml", "profile": "mlperf"},
        }
    )
    assert "result" in resp, resp
    assert resp["result"] == {"deleted": "old-ml"}
    assert str(captured["db_path"]).endswith("state.db")
    assert Path(captured["sessions_dir"]) == profile_home / "sessions"
    assert captured.get("closed") is True
