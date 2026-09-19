"""Blank project paths must not select the backend's working directory."""

import pytest

from hermes_cli import projects_db as pdb
from tui_gateway import server


@pytest.mark.parametrize("action", ["add_folder", "remove_folder", "set_primary"])
@pytest.mark.parametrize("path_params", [{}, {"path": ""}, {"path": " \t "}])
def test_blank_folder_mutations_leave_project_unchanged(
    tmp_path, monkeypatch, action, path_params
):
    cwd, other = tmp_path / "cwd", tmp_path / "other"
    cwd.mkdir()
    other.mkdir()
    monkeypatch.chdir(cwd)
    folders = [str(other)] + ([str(cwd)] if action != "add_folder" else [])
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="Project", folders=folders)
        before = pdb.get_project(conn, pid).to_dict()

        response = server._methods[f"projects.{action}"](1, {"id": pid, **path_params})

        assert pdb.get_project(conn, pid).to_dict() == before
        if action == "add_folder":
            assert response["error"]["code"] == 5063
        else:
            assert "error" not in response


@pytest.mark.parametrize("blank", ["", " \t "])
def test_blank_paths_never_bind_or_discover_cwd(tmp_path, monkeypatch, blank):
    monkeypatch.chdir(tmp_path)
    with pdb.connect_closing() as conn:
        cwd_id = pdb.create_project(conn, name="Cwd", folders=["."])
        assert pdb.find_by_primary_path(conn, ".").id == cwd_id
        assert pdb.find_by_primary_path(conn, blank) is None

        pid = pdb.create_project(
            conn, name="Unbound", folders=[blank], primary_path=blank
        )
        project = pdb.get_project(conn, pid)
        assert project.folders == []
        assert project.primary_path is None

        other = str(tmp_path / "other")
        pid = pdb.create_project(
            conn, name="Other", folders=[blank, other], primary_path=blank
        )
        assert pdb.get_project(conn, pid).primary_path == other
        assert pdb.record_discovered_repos(conn, [(blank, "blank")]) == 0
        assert pdb.list_discovered_repos(conn) == []
