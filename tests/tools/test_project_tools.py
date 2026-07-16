from __future__ import annotations

import json

from hermes_cli import projects_db as pdb
from tools import project_tools


def _use_projects_db(monkeypatch, tmp_path):
    db_path = tmp_path / "projects.db"
    original_connect_closing = pdb.connect_closing
    monkeypatch.setattr(pdb, "connect_closing", lambda: original_connect_closing(db_path=db_path))
    return db_path, original_connect_closing


def test_agent_session_project_create_keeps_profile_active_pointer_and_moves_session(monkeypatch, tmp_path):
    db_path, connect_closing = _use_projects_db(monkeypatch, tmp_path)
    with connect_closing(db_path=db_path) as conn:
        active_id = pdb.create_project(conn, name="Foreground", folders=["/work/foreground"])
        pdb.set_active(conn, active_id)

    workspace_moves: list[tuple[str, str, str]] = []
    project_tools.set_project_workspace_callback(lambda *args: workspace_moves.append(args))

    try:
        result = json.loads(
            project_tools.project_create(
                "Background", path="/work/background", task_id="session-background"
            )
        )
    finally:
        project_tools.set_project_workspace_callback(None)

    with connect_closing(db_path=db_path) as conn:
        assert pdb.get_active_id(conn) == active_id
        created = pdb.get_project(conn, result["id"])

    assert result["success"] is True
    assert created is not None
    assert created.name == "Background"
    assert workspace_moves == [("session-background", "/work/background", "Background")]


def test_legacy_project_create_sets_profile_active_pointer(monkeypatch, tmp_path):
    db_path, connect_closing = _use_projects_db(monkeypatch, tmp_path)
    with connect_closing(db_path=db_path) as conn:
        previous_id = pdb.create_project(conn, name="Previous", folders=["/work/previous"])
        pdb.set_active(conn, previous_id)

    result = json.loads(project_tools.project_create("Created", path="/work/created"))

    with connect_closing(db_path=db_path) as conn:
        assert pdb.get_active_id(conn) == result["id"]

    assert result["success"] is True


def test_legacy_project_switch_sets_profile_active_pointer(monkeypatch, tmp_path):
    db_path, connect_closing = _use_projects_db(monkeypatch, tmp_path)
    with connect_closing(db_path=db_path) as conn:
        active_id = pdb.create_project(conn, name="Active", folders=["/work/active"])
        target_id = pdb.create_project(conn, name="Target", folders=["/work/target"])
        pdb.set_active(conn, active_id)

    result = json.loads(project_tools.project_switch(target_id))

    with connect_closing(db_path=db_path) as conn:
        assert pdb.get_active_id(conn) == target_id

    assert result["success"] is True
    assert result["id"] == target_id


def test_agent_session_project_switch_keeps_profile_active_pointer(monkeypatch, tmp_path):
    db_path = tmp_path / "projects.db"
    original_connect_closing = pdb.connect_closing

    with original_connect_closing(db_path=db_path) as conn:
        active_id = pdb.create_project(conn, name="Foreground", folders=["/work/foreground"])
        background_id = pdb.create_project(conn, name="Background", folders=["/work/background"])
        pdb.set_active(conn, active_id)

    monkeypatch.setattr(pdb, "connect_closing", lambda: original_connect_closing(db_path=db_path))
    workspace_moves: list[tuple[str, str, str]] = []
    project_tools.set_project_workspace_callback(lambda *args: workspace_moves.append(args))

    try:
        result = json.loads(project_tools.project_switch(background_id, task_id="session-background"))
    finally:
        project_tools.set_project_workspace_callback(None)

    with original_connect_closing(db_path=db_path) as conn:
        assert pdb.get_active_id(conn) == active_id

    assert result["success"] is True
    assert result["id"] == background_id
    assert workspace_moves == [("session-background", "/work/background", "Background")]
