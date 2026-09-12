"""Project ownership must use the host's path comparison semantics."""

import pytest

from hermes_cli import projects_db as pdb
from tui_gateway import server


@pytest.mark.windows_only
@pytest.mark.parametrize("surface", ["rpc", "status"])
def test_windows_project_lookup_ignores_case_and_keeps_folder_boundaries(
    tmp_path, surface
):
    outer = tmp_path / "MyProject"
    inner = outer / "Nested"
    child = inner / "Src"
    child.mkdir(parents=True)
    sibling = tmp_path / "MyProjectOther"
    sibling.mkdir()
    assert child.samefile(str(child).upper())

    with pdb.connect_closing() as conn:
        outer_id = pdb.create_project(conn, name="Outer", folders=[str(outer)])
        inner_id = pdb.create_project(conn, name="Inner", folders=[str(inner)])

        def lookup(path):
            if surface == "status":
                return server._project_info_for_cwd(path)
            response = server._methods["projects.for_cwd"](1, {"cwd": path})
            assert "error" not in response
            return response["result"]["project"]

        assert lookup(str(outer).upper())["id"] == outer_id
        assert lookup(str(child).upper())["id"] == inner_id
        assert lookup(str(sibling).upper()) is None
        assert pdb.get_project(conn, inner_id).primary_path == str(inner)
        pdb.archive_project(conn, inner_id)
        assert lookup(str(child).upper())["id"] == outer_id


@pytest.mark.linux_only
def test_linux_project_lookup_keeps_case_distinct(tmp_path):
    upper, lower = tmp_path / "Project", tmp_path / "project"
    upper.mkdir()
    lower.mkdir()
    with pdb.connect_closing() as conn:
        upper_id = pdb.create_project(conn, name="Upper", folders=[str(upper)])
        lower_id = pdb.create_project(conn, name="Lower", folders=[str(lower)])
        assert pdb.project_for_path(conn, str(upper / "src")).id == upper_id
        assert pdb.project_for_path(conn, str(lower / "src")).id == lower_id
