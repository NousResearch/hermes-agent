from contextlib import contextmanager
from pathlib import Path

from tui_gateway import server


class _DB:
    def __init__(self):
        self.rows = [{"id": f"s{i}", "title": f"Title {i}", "model": "m", "started_at": i,
                      "last_active": i, "message_count": i, "source": "cli"} for i in range(75)]

    def list_sessions_rich(self, **kwargs):
        rows = sorted(self.rows, key=lambda row: row["last_active"], reverse=True)
        query = kwargs.get("search_query")
        if query:
            rows = [row for row in rows if query.lower() in row["title"].lower()]
        return rows[:kwargs["limit"]]

    def set_session_title(self, session_id, title):
        next(row for row in self.rows if row["id"] == session_id)["title"] = title
        return True

    def get_session(self, session_id):
        return next((row for row in self.rows if row["id"] == session_id), None)

    def export_session(self, session_id):
        row = self.get_session(session_id)
        return {**row, "messages": []} if row else None


def test_session_list_pages_searches_and_keeps_newest_first(monkeypatch):
    db = _DB()
    monkeypatch.setattr(server, "_get_db", lambda: db)
    first = server._methods["session.list"]("a", {"limit": 50})["result"]
    second = server._methods["session.list"]("b", {"limit": 50, "offset": 50})["result"]
    found = server._methods["session.list"]("c", {"query": "Title 7"})["result"]
    assert first["has_more"] is True
    assert [row["id"] for row in first["sessions"][:2]] == ["s74", "s73"]
    assert len(first["sessions"] + second["sessions"]) == 75
    assert all("Title 7" in row["title"] for row in found["sessions"])


def test_session_rename_and_export_use_database_and_shared_export(monkeypatch, tmp_path):
    db = _DB()
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setitem(server._sessions, "runtime-1", {
        "session_key": "s1", "history": [], "created_at": 1, "last_active": 1,
    })
    monkeypatch.setitem(server._sessions, "runtime-draft", {
        "session_key": "draft", "history": [], "created_at": 1, "last_active": 1,
    })

    live = server._methods["session.active_list"]("live", {})["result"]["sessions"]
    renamed = server._methods["session.rename"]("a", {"session_id": "s1", "title": "New"})["result"]
    exported = server._methods["session.export"]("b", {"session_id": "s1"})["result"]

    persisted = next(row for row in live if row["id"] == "runtime-1")
    draft = next(row for row in live if row["id"] == "runtime-draft")
    assert persisted["session_key"] == "s1"
    assert "session_key" not in draft
    assert renamed == {"session_id": "s1", "title": "New"}
    assert Path(exported["file"]).is_file()
    assert Path(exported["file"]).parent == tmp_path / "sessions" / "saved"


def test_session_export_writes_to_requested_profile_home(monkeypatch, tmp_path):
    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "secondary"
    db = _DB()

    @contextmanager
    def profile_db(_params):
        yield db

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(server, "_profile_db", profile_db)
    monkeypatch.setattr(server, "_profile_home", lambda profile: profile_home if profile == "secondary" else None)

    exported = server._methods["session.export"](
        "export", {"session_id": "s1", "profile": "secondary"}
    )["result"]

    exported_path = Path(exported["file"])
    assert exported_path.is_file()
    assert exported_path.parent == profile_home / "sessions" / "saved"
    assert not (default_home / "sessions" / "saved").exists()
