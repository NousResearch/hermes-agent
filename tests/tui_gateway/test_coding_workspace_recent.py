"""The checkout picker's project list through the real dispatch path: only folders that still exist,
one row per primary path, most recently used first — and listing never edits the registry."""
import sqlite3
import subprocess

import pytest

from tui_gateway import server


def rpc(method, **params):
    response = server.handle_request({"jsonrpc": "2.0", "id": "1", "method": method, "params": params})
    assert "error" not in response, response
    return response["result"]


def _repo(root, name):
    path = root / name
    path.mkdir()
    for args in (["init", "-b", "main"],
                 ["-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base"]):
        subprocess.check_output(["git", "-C", str(path), *args])
    return path


@pytest.fixture
def db(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    store = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: store)
    yield store
    store.close()


def _used(db, sid, cwd, at):
    db.create_session(sid, "desktop", cwd=str(cwd))
    db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?", (at, at, sid))
    db._conn.commit()


def test_recent_projects_skip_missing_folders_and_duplicates_and_follow_last_use(tmp_path, db):
    from hermes_cli import projects_db as pdb
    from hermes_constants import get_hermes_home
    old, used, idle = (_repo(tmp_path, name) for name in ("old", "used", "idle"))
    gone = tmp_path / "gone"
    with pdb.connect_closing() as conn:
        for path in (idle, gone, used, old):
            pdb.create_project(conn, name=path.name, folders=[str(path)])
        duplicate = pdb.create_project(conn, name="used again", folders=[str(used)], allow_duplicate_path=True)
    # Sessions predate registration (folders are often registered after their first chats), yet
    # use still outranks the idle project registered just now. A session in a worktree under
    # ``old`` counts as use of ``old``.
    (old / ".worktrees" / "task").mkdir(parents=True)
    _used(db, "s-old", old / ".worktrees" / "task", 1_000_000_000)
    _used(db, "s-used", used, 900_000_000)
    registry = get_hermes_home() / "projects.db"
    with sqlite3.connect(registry) as conn:
        before = conn.execute("SELECT * FROM projects ORDER BY id").fetchall()

    projects = rpc("projects.workspace.recent")["projects"]

    assert [p["primary_path"] for p in projects] == [str(old), str(used), str(idle)]
    assert duplicate not in {p["id"] for p in projects}
    with sqlite3.connect(registry) as conn:
        assert conn.execute("SELECT * FROM projects ORDER BY id").fetchall() == before
    gone.mkdir()
    assert str(gone) in {p["primary_path"] for p in rpc("projects.workspace.recent")["projects"]}


def test_the_offered_duplicate_is_the_one_prepare_accepts(tmp_path, db):
    """Two rows registered for one folder in the same second (a lost registration race) must not
    make the picker offer a project that Send then refuses as not owning the workspace."""
    from hermes_cli import projects_db as pdb
    repo = _repo(tmp_path, "repo")
    with pdb.connect_closing() as conn:
        for pid, slug in (("p_bbbb", "repo-2"), ("p_aaaa", "repo-3")):
            conn.execute("INSERT INTO projects (id, slug, name, primary_path, created_at) VALUES (?, ?, 'repo', ?, 7)",
                         (pid, slug, str(repo)))
            conn.execute("INSERT INTO project_folders (project_id, path, is_primary, added_at) VALUES (?, ?, 1, 7)",
                         (pid, str(repo)))
        conn.commit()

    offered = [p for p in rpc("projects.workspace.recent")["projects"] if p["primary_path"] == str(repo)]

    assert len(offered) == 1
    prepared = rpc("projects.workspace.prepare", path=str(repo), projectId=offered[0]["id"], mode="current")
    assert prepared["projectId"] == offered[0]["id"]
