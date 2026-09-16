"""Stored-thread moves through the real project registry and gateway persistence path."""
import json
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from hermes_cli import projects_db, profiles
from tools import project_tools
from tools.registry import registry
import tui_gateway.server as server


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    launch = get_hermes_home()
    homes = [launch, launch / "profiles" / "study"]
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: {"default": homes[0], "study": homes[1]}[name])
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    # Keep synchronous persistence; suppress only asynchronous Git enrichment.
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *a, **k: None)
    monkeypatch.setattr(project_tools, "_workspace_callback", server._apply_project_workspace)
    rows = []
    for i, home in enumerate(homes):
        home.mkdir(parents=True, exist_ok=True)
        old, dest = tmp_path / f"old-{i}", tmp_path / f"destination-{i}"
        old.mkdir()
        dest.mkdir()
        db = SessionDB(db_path=home / "state.db")
        for key in ("caller", "target"):
            db.create_session(key, "desktop", cwd=str(old))
            db.set_session_title(key, key)
            db.append_message(key, "user", f"history-{i}-{key}")
        caller = {"session_key": "caller", "cwd": str(old), "profile_home": str(home) if i else None}
        server._sessions[f"caller-{i}"] = caller
        with server._session_profile_runtime_scope(caller):
            with projects_db.connect_closing() as conn:
                active = projects_db.create_project(conn, name="Original", folders=[str(old)], primary_path=str(old))
                target = projects_db.create_project(conn, name="Destination", folders=[str(dest)], primary_path=str(dest))
                empty = projects_db.create_project(conn, name="Empty")
                projects_db.set_active(conn, active)
        rows.append((home, db, old, dest, active, target, empty, caller))
    monkeypatch.setattr(server, "_db", rows[0][1])
    try:
        yield rows
    finally:
        for row in rows:
            row[1].close()


def move(name="Destination", key="target", caller="caller"):
    return json.loads(registry.dispatch("desktop_project", {
        "action": "move", "name": name, "session_key": key,
    }, task_id=caller))


@pytest.mark.parametrize("live", [False, True])
def test_move_preserves_profile_caller_and_history(workspace, live):
    expected = [str(row[2]) for row in workspace]
    if live:
        # Same target ID in both profiles; insertion order must not decide ownership.
        for i, row in enumerate(workspace):
            server._sessions[f"target-{i}"] = {
                "session_key": "target", "cwd": expected[i],
                "profile_home": row[7]["profile_home"], "running": True,
            }
    for i in (0, 1, 0):  # A → B → A with real profile stores in one multiplexed process.
        home, db, old, dest, active, target, empty, caller = workspace[i]
        before = db.get_messages("target")
        with server._session_profile_runtime_scope(caller):
            result = move()
            assert result.get("success"), result
            assert Path(result["cwd"]) == dest
            with projects_db.connect_closing() as conn:
                assert projects_db.get_active_id(conn) == active
        expected[i] = str(dest)
        for j, row in enumerate(workspace):
            with SessionDB(db_path=row[0] / "state.db", read_only=True) as reopened:
                assert reopened.get_session("target")["cwd"] == expected[j]
                assert reopened.get_session("caller")["cwd"] == str(row[2])
            assert row[7]["cwd"] == str(row[2])
            if live:
                assert server._sessions[f"target-{j}"]["cwd"] == expected[j]
        assert db.get_messages("target") == before
        assert db.get_session("target")["title"] == "target"


def test_rejected_move_has_no_side_effects(workspace, monkeypatch):
    home, db, old, dest, active, target, empty, caller = workspace[0]
    with server._session_profile_runtime_scope(caller):
        cases = [
            ({"key": ""}, "session_key"),
            ({"key": " target "}, "exact stored session ID"),
            ({"key": "missing"}, "session not found"),
            ({"name": "missing"}, "no project"),
            ({"name": "Empty"}, "no folder"),
            ({"caller": "missing"}, "calling Desktop session"),
        ]
        for args, message in cases:
            result = move(**args)
            assert not result.get("success"), result
            assert message in result["error"], result
        dest.rmdir()
        result = move()
        assert not result.get("success") and "does not exist" in result["error"]
        monkeypatch.setattr(project_tools, "_workspace_callback", None)
        result = move()
        assert not result.get("success") and "connected Desktop" in result["error"]
        with projects_db.connect_closing() as conn:
            assert projects_db.get_active_id(conn) == active
    for row in workspace:
        assert row[1].get_session("target")["cwd"] == str(row[2])
        assert row[7]["cwd"] == str(row[2])
