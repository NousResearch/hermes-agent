"""Draft session persistence and legacy live-resume compatibility."""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def live_lazy_session(home):
    """A live registry record shaped like session.create's lazy output."""
    sid = "live-lazy-1"
    record = {
        "history": [],
        "last_active": 0.0,
        "pending_title": "Bot Chat",
        "pending_hidden": True,
        "profile_home": str(home / "profiles" / "ops"),
        "running": False,
        "session_key": "20260823_000000_abc123",
        "source": "desktop",
    }
    srv._sessions[sid] = record
    yield sid, record
    srv._sessions.pop(sid, None)


def _resume(params):
    return srv._methods["session.resume"](1, params)


@pytest.mark.parametrize("source", ("desktop", "tui"))
def test_created_draft_resumes_after_backend_restart(home, monkeypatch, source):
    """A durable id returned by session.create remains resumable with zero messages."""
    db = SessionDB(home / "state.db")
    created_sid = resumed_sid = ""
    try:
        monkeypatch.setattr(srv, "_enable_gateway_prompts", lambda: None)
        monkeypatch.setattr(srv, "_resolve_model", lambda: "test/model")
        monkeypatch.setattr(srv, "_schedule_agent_build", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda: None)
        monkeypatch.setattr(srv, "_get_db", lambda: db)

        created = srv._methods["session.create"](1, {"source": source})
        created_sid = created["result"]["session_id"]
        stored_sid = created["result"]["stored_session_id"]
        row = db.get_session(stored_sid)
        assert row and row["message_count"] == 0

        # Backend restart drops runtime state and reopens state.db from disk.
        srv._sessions.pop(created_sid, None)
        db.close()
        db = SessionDB(home / "state.db")

        resumed = _resume({"session_id": stored_sid, "source": source, "omit_messages": True})
        resumed_sid = resumed["result"]["session_id"]
        assert resumed["result"]["resumed"] == stored_sid
        assert resumed["result"]["message_count"] == 0
    finally:
        srv._sessions.pop(created_sid, None)
        srv._sessions.pop(resumed_sid, None)
        db.close()


def test_resume_by_stored_key_reattaches(live_lazy_session):
    sid, record = live_lazy_session
    out = _resume({"profile": "ops", "session_id": record["session_key"], "omit_messages": True})
    assert "error" not in out, out
    assert out["result"]["session_id"] == sid
    assert out["result"]["stored_session_id"] == record["session_key"]


def test_resume_by_pending_title_reattaches(live_lazy_session):
    sid, _record = live_lazy_session
    out = _resume({"profile": "ops", "session_id": "Bot Chat", "omit_messages": True})
    assert "error" not in out, out
    assert out["result"]["session_id"] == sid


def test_unscoped_resume_of_profile_session_fails_closed(live_lazy_session):
    _sid, record = live_lazy_session
    out = _resume({"session_id": record["session_key"], "omit_messages": True})
    assert out.get("error", {}).get("code") == 4007


def test_unknown_id_still_404s(home):
    out = _resume({"profile": "ops", "session_id": "ghost-9999", "omit_messages": True})
    assert out.get("error", {}).get("code") == 4007
