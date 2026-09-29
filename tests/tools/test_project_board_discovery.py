"""Project discovery must expose persisted board bindings (#91169)."""

import json

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import projects_db as pdb
from tools import project_tools  # noqa: F401 — register the public tool
from tools.registry import registry


def _list():
    return json.loads(registry.dispatch("desktop_project", {"action": "list"}))


def test_list_reports_each_stored_binding_and_tracks_rebind_and_unbind(tmp_path):
    assert _list()["projects"] == []
    with pdb.connect_closing() as conn:
        first = pdb.create_project(conn, name="First", folders=[str(tmp_path / "first")], board_slug="first-board")
        second = pdb.create_project(conn, name="Second", board_slug="second-board")
        unbound = pdb.create_project(conn, name="Unbound")
        pdb.set_active(conn, second)

    listed = _list()
    rows = {p["id"]: p for p in listed["projects"]}
    assert all("board_slug" in row for row in rows.values()), "project list omits persisted board bindings"
    assert {pid: row["board_slug"] for pid, row in rows.items()} == {
        first: "first-board", second: "second-board", unbound: None,
    }
    assert listed["active_id"] == second
    assert rows[first]["primary_path"] == str(tmp_path / "first")
    assert not rows[first]["active"] and rows[second]["active"]

    with pdb.connect_closing() as conn:
        pdb.update_project(conn, first, board_slug="replacement-board")
        pdb.update_project(conn, second, board_slug="")
    rows = {p["id"]: p for p in _list()["projects"]}
    assert rows[first]["board_slug"] == "replacement-board"
    assert rows[second]["board_slug"] is None
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) == second


def test_board_discovery_reads_the_calling_home_without_caching(tmp_path):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        token = set_hermes_home_override(home)
        try:
            with pdb.connect_closing() as conn:
                pdb.create_project(conn, name="Same name", board_slug=f"board-{home.name}")
        finally:
            reset_hermes_home_override(token)
    for home in [*homes, homes[0]]:
        token = set_hermes_home_override(home)
        try:
            rows = _list()["projects"]
            assert [p["board_slug"] for p in rows] == [f"board-{home.name}"]
        finally:
            reset_hermes_home_override(token)
