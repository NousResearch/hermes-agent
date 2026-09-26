"""Regression for #116126: a session born hidden stays un-hidden once the user un-hides it.

``_ensure_session_db_row`` runs from every prompt.submit; the deferred ``pending_hidden`` intent must be
consumed once the row carries it, not re-applied on every message.
"""

import pytest
import tui_gateway.server as srv
from hermes_state import SessionDB


@pytest.mark.parametrize("first_apply_fails", [False, True])
def test_unhide_survives_next_session_row_persist(tmp_path, monkeypatch, first_apply_fails):
    with SessionDB(tmp_path / "state.db") as db:
        monkeypatch.setattr(srv, "_get_db", lambda: db)
        monkeypatch.setattr(srv, "_schedule_agent_build", lambda sid: None)
        monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda: None)
        created = srv._methods["session.create"](1, {"hidden": True})
        assert "error" not in created, created
        sid = created["result"]["session_id"]
        session = srv._sessions[sid]
        key = session["session_key"]
        try:
            if first_apply_fails:
                # A transient write failure keeps the intent pending: the next persist still applies it.
                def fail_hide(*args):
                    raise OSError("temporary write failure")

                with monkeypatch.context() as failure:
                    failure.setattr(db, "set_session_hidden", fail_hide)
                    assert srv._ensure_session_db_row(session)
                assert db.get_session(key)["hidden"] == 0
            assert srv._ensure_session_db_row(session)
            assert db.get_session(key)["hidden"] == 1
            # User un-hides (REST PATCH / session.set_hidden both end in this db write) ...
            assert db.set_session_hidden(key, False)
            # ... and the next message's persist must not re-hide it.
            assert srv._ensure_session_db_row(session)
            assert db.get_session(key)["hidden"] == 0
        finally:
            srv._sessions.pop(sid, None)
