"""Project-scoped persistent terminal workspace identity."""

from pathlib import Path

from hermes_cli import projects_db
from tools import terminal_tool


def _project_db(tmp_path: Path, monkeypatch, *, profile: str = "default"):
    db_path = tmp_path / profile / "projects.db"
    db_path.parent.mkdir()
    monkeypatch.setattr(projects_db, "projects_db_path", lambda: db_path)
    conn = projects_db.connect(db_path)
    projects_db.create_project(conn, name="My App", slug="my-app", primary_path=str(tmp_path / "repo"))
    conn.close()
    return tmp_path / "repo"


def test_project_workspace_key_is_stable_and_sanitized(tmp_path, monkeypatch):
    repo = _project_db(tmp_path, monkeypatch)
    monkeypatch.setattr(terminal_tool, "_current_session_profile", lambda: "work/profile")
    assert terminal_tool._project_workspace_key(str(repo / "src")) == "project:work-profile:my-app"


def test_project_workspace_key_is_absent_for_unresolved_path(tmp_path, monkeypatch):
    _project_db(tmp_path, monkeypatch)
    assert terminal_tool._project_workspace_key(str(tmp_path / "unregistered")) is None


def test_persistent_resolution_scopes_project_slots_by_profile(tmp_path, monkeypatch):
    repo = _project_db(tmp_path, monkeypatch, profile="one")
    monkeypatch.setattr(terminal_tool, "_current_session_profile", lambda: "one")
    monkeypatch.setattr(terminal_tool, "_session_scope", lambda: terminal_tool._SessionScope("docker", True))
    terminal_tool.record_session_cwd("default", str(repo))
    first = terminal_tool._resolve_container_task_id(None)

    other_db = tmp_path / "two" / "projects.db"
    other_db.parent.mkdir()
    monkeypatch.setattr(projects_db, "projects_db_path", lambda: other_db)
    conn = projects_db.connect(other_db)
    projects_db.create_project(conn, name="My App", slug="my-app", primary_path=str(repo))
    conn.close()
    monkeypatch.setattr(terminal_tool, "_current_session_profile", lambda: "two")
    second = terminal_tool._resolve_container_task_id(None)

    assert first == "project:one:my-app"
    assert second == "project:two:my-app"
    assert first != second


def test_unresolved_project_keeps_existing_session_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    monkeypatch.setattr(terminal_tool, "_session_scope", lambda: terminal_tool._SessionScope("docker", True))
    terminal_tool.record_session_cwd("default", str(tmp_path / "unregistered"))
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_explicit_shared_key_precedes_project_workspace_for_two_profiles(tmp_path, monkeypatch):
    repo = _project_db(tmp_path, monkeypatch, profile="one")
    monkeypatch.setattr(terminal_tool, "_session_scope", lambda: terminal_tool._SessionScope("docker", True))
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team")
    terminal_tool.record_session_cwd("default", str(repo))

    monkeypatch.setattr(terminal_tool, "_current_session_profile", lambda: "one")
    first = terminal_tool._resolve_container_task_id(None)

    other_db = tmp_path / "two" / "projects.db"
    other_db.parent.mkdir()
    monkeypatch.setattr(projects_db, "projects_db_path", lambda: other_db)
    conn = projects_db.connect(other_db)
    projects_db.create_project(conn, name="My App", slug="my-app", primary_path=str(repo))
    conn.close()
    monkeypatch.setattr(terminal_tool, "_current_session_profile", lambda: "two")
    second = terminal_tool._resolve_container_task_id(None)

    assert first == second == "shared:team"
