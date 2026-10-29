"""``session.workspace.move`` with the explicit project binding (``sessions.project_id``).

Into a project sets the binding AND the folder; an explicit null unassigns; a legacy cwd-only move
keeps the binding inside the project's folders and drops it when the session leaves them.
"""

from __future__ import annotations

import contextlib

import pytest

from hermes_cli import projects_db as pdb
from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def db(tmp_path, monkeypatch):
    store = SessionDB(tmp_path / "state.db")

    @contextlib.contextmanager
    def _db(_params, *, writer=False):
        yield store

    monkeypatch.setattr(server, "_profile_db", _db)
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: None)
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda cwd: None)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def projects(tmp_path):
    a, b = tmp_path / "alpha", tmp_path / "beta"
    a.mkdir()
    b.mkdir()
    (a / "src").mkdir()
    with pdb.connect_closing() as conn:
        alpha = pdb.create_project(conn, name="Alpha", slug="alpha", folders=[str(a)], board_slug="alpha")
        beta = pdb.create_project(conn, name="Beta", slug="beta", folders=[str(b)])
        return {"alpha": pdb.get_project(conn, alpha), "beta": pdb.get_project(conn, beta)}


def _move(**params):
    return server._methods["session.workspace.move"]("rid", {"session_key": "s1", **params})


def _row(db):
    return db.get_session("s1")


def test_move_into_a_project_binds_it_and_defaults_cwd_to_its_folder(db, projects):
    db.create_session("s1", "tui")
    alpha = projects["alpha"]
    res = _move(project_id=alpha.id)
    assert "error" not in res, res
    result = res["result"]
    assert result["cwd"] == alpha.primary_path
    assert result["project_id"] == alpha.id
    assert result["project"] == {"id": alpha.id, "slug": "alpha", "name": "Alpha", "icon": None, "color": None,
                                 "folder": alpha.primary_path, "board_slug": "alpha"}
    assert (_row(db)["project_id"], _row(db)["cwd"]) == (alpha.id, alpha.primary_path)


def test_move_by_slug_with_an_explicit_subfolder(db, projects):
    db.create_session("s1", "tui")
    alpha = projects["alpha"]
    sub = f"{alpha.primary_path}/src"
    res = _move(project_id="alpha", cwd=sub)
    assert res["result"]["project_id"] == alpha.id
    assert (_row(db)["project_id"], _row(db)["cwd"]) == (alpha.id, sub)


def test_moving_to_another_project_replaces_the_binding(db, projects):
    db.create_session("s1", "tui")
    _move(project_id=projects["alpha"].id)
    _move(project_id=projects["beta"].id)
    assert _row(db)["project_id"] == projects["beta"].id  # exactly one project
    assert _row(db)["cwd"] == projects["beta"].primary_path


def test_explicit_null_unassigns(db, projects, tmp_path):
    db.create_session("s1", "tui")
    _move(project_id=projects["alpha"].id)
    res = _move(project_id=None, cwd=str(tmp_path))
    assert res["result"]["project_id"] is None and res["result"]["project"] is None
    assert _row(db)["project_id"] is None
    assert _row(db)["cwd"] == str(tmp_path)


def test_legacy_cwd_only_move_keeps_the_binding_inside_the_project(db, projects):
    db.create_session("s1", "tui")
    alpha = projects["alpha"]
    _move(project_id=alpha.id)
    res = _move(cwd=f"{alpha.primary_path}/src")
    assert res["result"]["project_id"] == alpha.id
    assert _row(db)["project_id"] == alpha.id


def test_legacy_cwd_only_move_out_of_the_project_clears_the_binding(db, projects):
    db.create_session("s1", "tui")
    _move(project_id=projects["alpha"].id)
    res = _move(cwd=projects["beta"].primary_path)
    assert res["result"]["project_id"] is None
    assert _row(db)["project_id"] is None  # a cwd move never infers a NEW binding


@pytest.mark.parametrize("ref", ["nope", "archived"])
def test_unknown_or_archived_project_is_refused_without_writing(db, projects, ref):
    db.create_session("s1", "tui")
    if ref == "archived":
        with pdb.connect_closing() as conn:
            pdb.archive_project(conn, projects["beta"].id)
        ref = projects["beta"].id
    res = _move(project_id=ref)
    assert res["error"]["code"] == 5062
    assert _row(db)["project_id"] is None
    assert _row(db)["cwd"] is None

