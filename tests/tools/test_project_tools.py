from __future__ import annotations

import json

import pytest

from hermes_cli import projects_db as pdb
from tools import project_tools, terminal_tool
from tools.registry import registry


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


@pytest.fixture
def gui_sessions(monkeypatch):
    """Live GUI sessions wired through the real gateway workspace callback."""
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_persist_session_cwd_and_schedule_git_meta", lambda *a, **k: None)
    project_tools.set_project_workspace_callback(server._apply_project_workspace)

    def open_session(key, cwd=""):
        session = {"session_key": key, "cwd": cwd, "source": "desktop"}
        server._sessions[f"sid-{key}"] = session
        server._register_session_cwd(session)
        return session

    try:
        yield open_session
    finally:
        project_tools.set_project_workspace_callback(None)


def _tool(action, task_id=None, **args):
    return json.loads(registry.dispatch("desktop_project", {"action": action, **args}, task_id=task_id))


def test_two_sessions_switch_projects_independently(gui_sessions, tmp_path):
    folders = {name: tmp_path / name for name in ("foreground", "a", "b")}
    for folder in folders.values():
        folder.mkdir()
    with pdb.connect_closing() as conn:
        foreground = pdb.create_project(conn, name="Foreground", folders=[str(folders["foreground"])])
        pdb.set_active(conn, foreground)
    tab_a = gui_sessions("tab-a", str(folders["foreground"]))
    tab_b = gui_sessions("tab-b", str(folders["foreground"]))

    project_a = _tool("create", "tab-a", name="Project A", path=str(folders["a"]))["id"]
    project_b = _tool("create", "tab-b", name="Project B", path=str(folders["b"]))["id"]

    # Each tab's workspace moved; the profile-global pointer the sidebar owns did not.
    assert (tab_a["cwd"], tab_b["cwd"]) == (str(folders["a"]), str(folders["b"]))
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) == foreground
    listed_a, listed_b = _tool("list", "tab-a"), _tool("list", "tab-b")
    assert listed_a["active_id"] == project_a
    assert listed_b["active_id"] == project_b
    assert [p["id"] for p in listed_a["projects"] if p["active"]] == [project_a]

    # Tab B switching into A's project leaves tab A (and the global pointer) where they were.
    assert _tool("switch", "tab-b", name="Project A")["id"] == project_a
    assert _tool("list", "tab-b")["active_id"] == project_a
    assert _tool("list", "tab-a")["active_id"] == project_a
    assert _tool("switch", "tab-a", name="Foreground")["id"] == foreground
    assert _tool("list", "tab-a")["active_id"] == foreground
    assert _tool("list", "tab-b")["active_id"] == project_a
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) == foreground


def test_project_list_reports_no_project_for_a_detached_session(gui_sessions, tmp_path):
    with pdb.connect_closing() as conn:
        active = pdb.create_project(conn, name="Sidebar pick", folders=[str(tmp_path)])
        pdb.set_active(conn, active)
    gui_sessions("detached")

    listed = _tool("list", "detached")

    assert listed["active_id"] is None
    assert not any(p["active"] for p in listed["projects"])
    # A caller the gateway never registered keeps reading the profile-global pointer.
    assert _tool("list")["active_id"] == active


def test_idempotent_create_from_a_session_keeps_profile_active_pointer(gui_sessions, tmp_path):
    folders = {name: tmp_path / name for name in ("foreground", "existing")}
    for folder in folders.values():
        folder.mkdir()
    with pdb.connect_closing() as conn:
        foreground = pdb.create_project(conn, name="Foreground", folders=[str(folders["foreground"])])
        existing = pdb.create_project(
            conn, name="Existing", folders=[str(folders["existing"])], primary_path=str(folders["existing"]))
        pdb.set_active(conn, foreground)
    tab = gui_sessions("tab", str(folders["foreground"]))

    result = _tool("create", "tab", name="Again", path=str(folders["existing"]))

    assert result["id"] == existing
    assert tab["cwd"] == str(folders["existing"])
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) == foreground
