"""Stored-thread moves through the real project registry and gateway persistence path."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profiles, projects_db
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import conversation, session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tools import project_tools
from tools.registry import registry
from tui_gateway import server


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    launch = tmp_path / "launch-home"
    homes = [launch, launch / "profiles" / "study"]
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_ENV", "local")
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
        db = open_persistence(profile_scope(home=home), eager=True)
        for key in ("caller", "target"):
            seed_session(db, key, "desktop", title=key, cwd=str(old))
            seed_messages(db, key, {"role": "user", "content": f"history-{i}-{key}"})
        caller = {"session_key": "caller", "cwd": str(old), "profile_home": str(home) if i else None}
        server._sessions[f"caller-{i}"] = caller
        with server._session_profile_runtime_scope(caller):
            with projects_db.connect_closing() as conn:
                active = projects_db.create_project(conn, name="Original", folders=[str(old)], primary_path=str(old))
                target = projects_db.create_project(conn, name="Destination", folders=[str(dest)], primary_path=str(dest))
                empty = projects_db.create_project(conn, name="Empty")
                projects_db.set_active(conn, active)
        rows.append(SimpleNamespace(
            home=home, db=db, old=old, dest=dest, active=active, target=target, empty=empty, caller=caller,
        ))
    monkeypatch.setattr(server, "_db", rows[0].db)
    try:
        yield rows
    finally:
        for row in rows:
            row.db.close()


def move(name="Destination", key="target", caller="caller"):
    return json.loads(registry.dispatch("desktop_project", {
        "action": "move", "name": name, "session_key": key,
    }, task_id=caller))


@pytest.mark.parametrize("live", [False, True])
def test_move_preserves_profile_caller_and_history(workspace, live):
    expected = [str(row.old) for row in workspace]
    if live:
        # Same target ID in both profiles; insertion order must not decide ownership.
        for i, row in enumerate(workspace):
            server._sessions[f"target-{i}"] = {
                "session_key": "target", "cwd": expected[i],
                "profile_home": row.caller["profile_home"], "running": True,
            }
    for i in (0, 1, 0):  # A → B → A with real profile stores in one multiplexed process.
        current = workspace[i]
        before = conversation(current.db, "target")
        with server._session_profile_runtime_scope(current.caller):
            result = move()
            assert result.get("success"), result
            assert Path(result["cwd"]) == current.dest
            with projects_db.connect_closing() as conn:
                assert projects_db.get_active_id(conn) == current.active
        expected[i] = str(current.dest)
        for j, row in enumerate(workspace):
            with open_persistence(profile_scope(home=row.home), eager=True) as reopened:
                assert session_row(reopened, "target")["cwd"] == expected[j]
                assert session_row(reopened, "caller")["cwd"] == str(row.old)
            assert row.caller["cwd"] == str(row.old)
            if live:
                assert server._sessions[f"target-{j}"]["cwd"] == expected[j]
        assert conversation(current.db, "target") == before
        assert session_row(current.db, "target")["title"] == "target"


def test_rejected_move_has_no_side_effects(workspace, monkeypatch):
    current = workspace[0]
    with server._session_profile_runtime_scope(current.caller):
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
        current.dest.rmdir()
        result = move()
        assert not result.get("success") and "does not exist" in result["error"]
        monkeypatch.setattr(project_tools, "_workspace_callback", None)
        result = move()
        assert not result.get("success") and "connected Desktop" in result["error"]
        with projects_db.connect_closing() as conn:
            assert projects_db.get_active_id(conn) == current.active
    for row in workspace:
        assert session_row(row.db, "target")["cwd"] == str(row.old)
        assert row.caller["cwd"] == str(row.old)


def test_move_rejects_running_caller_before_callback_across_profiles(workspace, monkeypatch):
    real_callback = server._apply_project_workspace
    calls = []
    histories = {(i, key): conversation(row.db, key)
                 for i, row in enumerate(workspace) for key in ("caller", "target")}

    def recording_callback(*args, **kwargs):
        calls.append((args, kwargs))
        return real_callback(*args, **kwargs)

    monkeypatch.setattr(project_tools, "_workspace_callback", recording_callback)
    for row in workspace:
        row.caller["running"] = True

    for i in (0, 1, 0):
        current = workspace[i]
        with server._session_profile_runtime_scope(current.caller):
            result = move(key="caller")
            assert result.get("success") is False, result
            assert "switch" in result["error"], result
            with projects_db.connect_closing() as conn:
                assert projects_db.get_active_id(conn) == current.active
        assert calls == []
        for j, row in enumerate(workspace):
            with open_persistence(profile_scope(home=row.home), eager=True) as reopened:
                for key in ("caller", "target"):
                    assert session_row(reopened, key)["cwd"] == str(row.old)
                    assert session_row(reopened, key)["title"] == key
                    assert conversation(reopened, key) == histories[j, key]
            with server._session_profile_runtime_scope(row.caller):
                with projects_db.connect_closing() as conn:
                    assert projects_db.get_active_id(conn) == row.active
            assert row.caller["cwd"] == str(row.old)

    # Also guard runtime-sid callers at the callback boundary (e.g. caller="caller-0", key="caller").
    with server._session_profile_runtime_scope(workspace[0].caller):
        runtime_self = move(caller="caller-0", key="caller")
        assert runtime_self.get("success") is False, runtime_self
        assert "switch" in runtime_self["error"], runtime_self
    with open_persistence(profile_scope(home=workspace[0].home), eager=True) as reopened:
        assert session_row(reopened, "caller")["cwd"] == str(workspace[0].old)
    assert workspace[0].caller["cwd"] == str(workspace[0].old)


def _seed_remote_destination(caller, target_id, remote_path):
    with server._session_profile_runtime_scope(caller):
        with projects_db.connect_closing() as conn:
            # create_project normalizes on the host; seed the stored remote spelling directly.
            conn.execute("UPDATE projects SET primary_path = ? WHERE id = ?", (remote_path, target_id))
            conn.execute("DELETE FROM project_folders WHERE project_id = ?", (target_id,))
            conn.execute(
                "INSERT INTO project_folders (project_id, path, is_primary, added_at) VALUES (?, ?, 1, 0)",
                (target_id, remote_path),
            )
            conn.commit()


def test_remote_ssh_stored_move_forwards_literal_path_and_isolates_profiles(workspace, monkeypatch):
    remote_path = "~/hermes-remote-only-project-move/target"
    study = workspace[1]
    histories = {(i, key): conversation(row.db, key)
                 for i, row in enumerate(workspace) for key in ("caller", "target")}
    (study.home / "config.yaml").write_text("terminal:\n  backend: ssh\n", encoding="utf-8")
    _seed_remote_destination(study.caller, study.target, remote_path)

    # Colliding live local target in launch profile must not hijack study's SSH home or get mutated.
    server._sessions["target-0"] = {
        "session_key": "target", "cwd": str(workspace[0].old), "profile_home": None, "running": True,
    }
    native_move = server._methods["session.workspace.move"]
    calls = []

    def recording_native(rid, params):
        calls.append(dict(params))
        return native_move(rid, params)

    monkeypatch.setitem(server._methods, "session.workspace.move", recording_native)
    with server._session_profile_runtime_scope(study.caller):
        result = move()
        assert result.get("success") is True, result
        assert result["cwd"] == remote_path
        with projects_db.connect_closing() as conn:
            assert projects_db.get_active_id(conn) == study.active
    assert calls == [{"session_key": "target", "cwd": remote_path, "profile": "study"}]

    with open_persistence(profile_scope(home=study.home), eager=True) as reopened_study:
        assert session_row(reopened_study, "target")["cwd"] == remote_path
        assert session_row(reopened_study, "target")["title"] == "target"
        assert conversation(reopened_study, "target") == histories[1, "target"]
        assert session_row(reopened_study, "caller")["cwd"] == str(study.old)
        assert session_row(reopened_study, "caller")["title"] == "caller"
        assert conversation(reopened_study, "caller") == histories[1, "caller"]
    assert study.caller["cwd"] == str(study.old)

    with open_persistence(profile_scope(home=workspace[0].home), eager=True) as reopened_launch:
        for key in ("caller", "target"):
            assert session_row(reopened_launch, key)["cwd"] == str(workspace[0].old)
            assert session_row(reopened_launch, key)["title"] == key
            assert conversation(reopened_launch, key) == histories[0, key]
    assert server._sessions["target-0"]["cwd"] == str(workspace[0].old)
    assert workspace[0].caller["cwd"] == str(workspace[0].old)


def test_remote_docker_missing_path_forwards_literal_path_to_native_refusal(workspace, monkeypatch):
    remote_path = "/workspace/hermes-remote-only-project-move/target"
    study = workspace[1]
    histories = {(i, key): conversation(row.db, key)
                 for i, row in enumerate(workspace) for key in ("caller", "target")}
    (study.home / "config.yaml").write_text("terminal:\n  backend: docker\n", encoding="utf-8")
    _seed_remote_destination(study.caller, study.target, remote_path)

    native_move = server._methods["session.workspace.move"]
    calls = []

    def recording_native(rid, params):
        calls.append(dict(params))
        return native_move(rid, params)

    monkeypatch.setitem(server._methods, "session.workspace.move", recording_native)
    with server._session_profile_runtime_scope(study.caller):
        result = move()
        assert result.get("success") is False, result
        assert "does not exist" in result["error"], result
        with projects_db.connect_closing() as conn:
            assert projects_db.get_active_id(conn) == study.active
    assert calls == [{"session_key": "target", "cwd": remote_path, "profile": "study"}]
    for i, row in enumerate(workspace):
        with open_persistence(profile_scope(home=row.home), eager=True) as reopened:
            for key in ("caller", "target"):
                assert session_row(reopened, key)["cwd"] == str(row.old)
                assert session_row(reopened, key)["title"] == key
                assert conversation(reopened, key) == histories[i, key]
        with server._session_profile_runtime_scope(row.caller):
            with projects_db.connect_closing() as conn:
                assert projects_db.get_active_id(conn) == row.active
        assert row.caller["cwd"] == str(row.old)


def test_move_survives_local_desktop_ui_workspace_wrapper(workspace, monkeypatch):
    from gateway import local_desktop_ui

    monkeypatch.setattr(local_desktop_ui, "_workspace_routes", {})
    local_desktop_ui._install_workspace_hook()
    assert getattr(project_tools.get_project_workspace_callback(), "_local_desktop_ui", False)

    launch = workspace[0]
    with server._session_profile_runtime_scope(launch.caller):
        result = move()
        assert result.get("success") is True, result
        assert Path(result["cwd"]) == launch.dest
        with projects_db.connect_closing() as conn:
            assert projects_db.get_active_id(conn) == launch.active
    with open_persistence(profile_scope(home=launch.home), eager=True) as reopened:
        assert session_row(reopened, "target")["cwd"] == str(launch.dest)
        assert session_row(reopened, "caller")["cwd"] == str(launch.old)
    assert launch.caller["cwd"] == str(launch.old)
