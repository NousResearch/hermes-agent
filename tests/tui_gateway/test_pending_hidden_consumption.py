"""Regression for #116126: initial hiding must not override later recovery."""

import pytest
import tui_gateway.server as srv
from hermes_state import SessionDB


@pytest.mark.parametrize("first_write", ["success", "error", "missing"])
def test_unhide_survives_repeated_session_row_persistence(tmp_path, monkeypatch, first_write):
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
            if first_write != "success":
                def fail_hide(*args):
                    if first_write == "error":
                        raise OSError("temporary write failure")
                    return False

                with monkeypatch.context() as failure:
                    failure.setattr(db, "set_session_hidden", fail_hide)
                    assert srv._ensure_session_db_row(session)
                assert db.get_session(key)["hidden"] == 0
            assert srv._ensure_session_db_row(session)
            assert db.get_session(key)["hidden"] == 1
            assert db.set_session_hidden(key, False)
            assert srv._ensure_session_db_row(session)
            assert db.get_session(key)["hidden"] == 0
        finally:
            srv._sessions.pop(sid, None)
