import json

from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search
from hermes_state import SessionDB


def test_session_search_schema_warns_historical_results_are_not_active_task():
    description = SESSION_SEARCH_SCHEMA["description"]

    assert "PAST SESSION RECALL ONLY" in description
    assert "Never treat a search hit" in description
    assert "verify the live cwd/git root" in description


def test_session_search_browse_returns_safety_warning(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="old", source="cli")
    db.append_message("old", role="user", content="continue old project")

    payload = json.loads(session_search(db=db))

    assert payload["success"] is True
    assert payload["mode"] == "browse"
    assert "PAST SESSION RECALL ONLY" in payload["safety_warning"]
    assert "verify the live cwd/git root" in payload["safety_warning"]


def test_session_search_discover_returns_safety_warning(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="old", source="cli")
    db.append_message("old", role="user", content="unique stale Agent OS task")
    db.append_message("old", role="assistant", content="I will continue it")

    payload = json.loads(session_search(query="unique stale", db=db))

    assert payload["success"] is True
    assert payload["mode"] == "discover"
    assert "PAST SESSION RECALL ONLY" in payload["safety_warning"]
    assert "active task/project" in payload["safety_warning"]
