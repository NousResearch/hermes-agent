"""Search never becomes another session-database initializer."""

import json
from pathlib import Path
import sqlite3



def test_search_does_not_create_current_storage_for_empty_or_foreign_reads(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB
    import tools.session_search_tool  # noqa: F401
    from tools.registry import registry

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    handler = registry.get_entry("session_search").handler
    try:
        empty = json.loads(handler({}))
        assert not (home / "state.db").exists(), "a reader created the canonical database"
        assert empty["success"] is False
        other = home / "profiles" / "reader-target"
        other.mkdir(parents=True)
        db = SessionDB(other / "state.db")
        try:
            db.create_session("foreign-read", source="cli")
            db.append_message("foreign-read", role="user", content="owned target marker")
        finally:
            db.close()
        for args in [
            {"session_id": "foreign-read", "profile": "reader-target"},
            {"session_id": "reader-target/foreign-read"},
        ]:
            result = json.loads(handler(args))
            assert result["success"] is True, result
            assert result["messages"][0]["content"] == "owned target marker"
            assert not (home / "state.db").exists()
    finally:
        reset_hermes_home_override(token)


def test_search_reads_live_storage_without_a_write_commit(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB
    import tools.session_search_tool  # noqa: F401
    from tools.registry import registry

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    owner = SessionDB(home / "state.db")
    observer = sqlite3.connect(home / "state.db")
    try:
        owner.create_session("live-read", source="cli")
        owner.append_message("live-read", role="user", content="live reader marker")
        before = observer.execute("PRAGMA data_version").fetchone()[0]
        result = json.loads(registry.get_entry("session_search").handler({"session_id": "live-read"}))
        assert result["success"] is True, result
        assert result["messages"][0]["content"] == "live reader marker"
        assert observer.execute("PRAGMA data_version").fetchone()[0] == before
    finally:
        observer.close()
        owner.close()
        reset_hermes_home_override(token)
