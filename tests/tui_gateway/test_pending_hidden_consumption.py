"""Regression for #116126: a session born hidden stays un-hidden once the user un-hides it.

``_ensure_session_db_row`` runs from every prompt.submit; the deferred ``pending_hidden`` intent must be
consumed once the row carries it, not re-applied on every message.
"""

import pytest
import tui_gateway.server as srv
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import session_row


@pytest.mark.parametrize("first_apply_fails", [False, True])
def test_unhide_survives_next_session_row_persist(tmp_path, monkeypatch, first_apply_fails):
    with open_persistence(profile_scope(home=tmp_path)) as db:
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

                with srv._persistence_for(db) as persistence:
                    sessions_type = type(persistence.sessions)
                with monkeypatch.context() as failure:
                    failure.setattr(sessions_type, "set_hidden", fail_hide)
                    assert srv._ensure_session_db_row(session)
                assert session_row(db, key)["hidden"] == 0
            assert srv._ensure_session_db_row(session)
            assert session_row(db, key)["hidden"] == 1
            # User un-hides (REST PATCH / session.set_hidden both end in this db write) ...
            assert db.sessions.set_hidden(key, False)
            # ... and the next message's persist must not re-hide it.
            assert srv._ensure_session_db_row(session)
            assert session_row(db, key)["hidden"] == 0
        finally:
            srv._sessions.pop(sid, None)
