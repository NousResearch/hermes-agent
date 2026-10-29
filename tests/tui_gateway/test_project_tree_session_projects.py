"""Project membership with the explicit binding (``sessions.project_id``): the binding wins over
any folder match, and unbound rows claimed by cwd alone are reported as ``folderMatches``
("In this folder") so a client can show them apart from the project's own sessions."""

from __future__ import annotations

from tui_gateway import project_tree as pt


def _session(sid, cwd, project_id=None, last_active=1000):
    return {"id": sid, "cwd": cwd, "git_branch": "", "git_repo_root": "", "started_at": last_active,
            "last_active": last_active, "title": None, "preview": None, "source": "desktop",
            "project_id": project_id}


def _project(pid, folder, **over):
    return {"id": pid, "name": pid.title(), "primary_path": folder, "archived": False,
            "folders": [{"path": folder, "is_primary": True}], **over}


def _node(tree, pid):
    return next(p for p in tree["projects"] if p["id"] == pid)


def _ids(node):
    return sorted(node["sessionIds"])


PROJECTS = [_project("outer", "/work"), _project("inner", "/work/inner")]


def test_project_id_wins_over_the_folder():
    # s1 lives in inner's folder but was deliberately put in outer: outer owns it.
    sessions = [_session("s1", "/work/inner", project_id="outer"), _session("s2", "/work/inner")]
    tree = pt.build_tree(PROJECTS, sessions, [])
    assert _ids(_node(tree, "outer")) == ["s1"]
    assert _ids(_node(tree, "inner")) == ["s2"]
    assert _node(tree, "outer")["folderMatchIds"] == []


def test_a_bound_session_outside_every_folder_still_belongs_to_its_project():
    tree = pt.build_tree(PROJECTS, [_session("s1", "/elsewhere", project_id="inner")], [])
    assert _ids(_node(tree, "inner")) == ["s1"]
    assert "s1" in tree["scoped_session_ids"]


def test_unbound_cwd_matches_are_listed_separately():
    sessions = [_session("bound", "/work/inner", project_id="inner", last_active=3),
                _session("loose-old", "/work/inner/src", last_active=1),
                _session("loose-new", "/work/inner", last_active=2)]
    hydrated = _node(pt.build_tree(PROJECTS, sessions, [], hydrate=True), "inner")
    assert _ids(hydrated) == ["bound", "loose-new", "loose-old"]
    assert hydrated["folderMatchIds"] == ["loose-new", "loose-old"]  # newest first
    assert [s["id"] for s in hydrated["folderMatches"]] == ["loose-new", "loose-old"]
    # The overview (projects.tree) carries ids only; rows are hydrated for the drill-in.
    overview = _node(pt.build_tree(PROJECTS, sessions, []), "inner")
    assert overview["folderMatchIds"] == ["loose-new", "loose-old"]
    assert overview["folderMatches"] == []


def test_binding_to_an_archived_project_falls_back_to_the_folder():
    projects = [_project("outer", "/work"), _project("gone", "/gone", archived=True)]
    tree = pt.build_tree(projects, [_session("s1", "/work/x", project_id="gone")], [])
    outer = _node(tree, "outer")
    assert _ids(outer) == ["s1"]
    assert outer["folderMatchIds"] == ["s1"]
