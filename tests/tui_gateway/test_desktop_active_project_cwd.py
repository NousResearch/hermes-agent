import contextlib
from pathlib import Path

from hermes_cli import projects_db
from tui_gateway import server


def _active_project(home: Path, primary: Path) -> None:
    with contextlib.closing(projects_db.connect(home / "projects.db")) as conn:
        project_id = projects_db.create_project(
            conn,
            name="Workspace",
            primary_path=str(primary),
        )
        projects_db.set_active(conn, project_id)


def test_desktop_session_falls_back_to_active_project(monkeypatch, tmp_path):
    launch_dir = tmp_path / "hermes-agent"
    workspace = tmp_path / "workspace"
    launch_dir.mkdir()
    workspace.mkdir()
    _active_project(tmp_path, workspace)

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    monkeypatch.setenv("TERMINAL_CWD", str(launch_dir))
    monkeypatch.chdir(launch_dir)

    assert server._completion_cwd({"source": "desktop"}) == str(workspace)


def test_explicit_and_configured_cwd_precede_active_project(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    for directory in (workspace, configured, explicit):
        directory.mkdir()
    _active_project(tmp_path, workspace)

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: str(configured))

    assert server._completion_cwd({"source": "desktop"}) == str(configured)
    assert server._completion_cwd({"source": "desktop", "cwd": str(explicit)}) == str(explicit)


def test_active_project_is_desktop_only_and_requires_existing_primary(monkeypatch, tmp_path):
    launch_dir = tmp_path / "launch"
    removed_workspace = tmp_path / "removed"
    launch_dir.mkdir()
    _active_project(tmp_path, removed_workspace)

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launch_dir)

    assert server._completion_cwd({"source": "desktop"}) == str(launch_dir)

    removed_workspace.mkdir()
    assert server._completion_cwd({"source": "cli"}) == str(launch_dir)


def test_named_profile_active_project_precedes_launch_profile_config(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    launch_configured = tmp_path / "launch-configured"
    workspace.mkdir()
    launch_configured.mkdir()

    profile_home = tmp_path / "profiles" / "other"
    profile_home.mkdir(parents=True)
    _active_project(profile_home, workspace)

    monkeypatch.setattr(server, "_profile_home", lambda profile: profile_home)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda home: None)
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: str(launch_configured))

    assert server._completion_cwd({"source": "desktop", "profile": "other"}) == str(workspace)


def test_active_project_cwd_is_persisted_for_desktop_session():
    session = {
        "source": "desktop",
        "cwd": "/active/project",
        "cwd_source": "active_project",
        "explicit_cwd": False,
    }

    assert server._persisted_session_cwd(session) == "/active/project"
