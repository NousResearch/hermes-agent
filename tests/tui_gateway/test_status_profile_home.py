"""Profile-owned /status path rendering for multiplexed TUI/Desktop sessions (#124500)."""

import threading

from tui_gateway import server


def _session(profile_home=None):
    row = {
        "session_key": "status-row",
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    if profile_home is not None:
        row["profile_home"] = str(profile_home)
    return row


def test_session_status_path_uses_owning_profile_home(monkeypatch, tmp_path):
    launch_home = tmp_path / "profiles" / "launch"
    profile_home = tmp_path / "profiles" / "ember"
    launch_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    class LaunchDB:
        def get_session(self, _key):
            raise AssertionError("secondary-profile status must not read the launch DB")

    class ProfileDB:
        def __init__(self, db_path=None):
            assert str(db_path) == str(profile_home / "state.db")

        def get_session(self, key):
            return {"id": key, "title": "owned", "started_at": 1}

        def close(self):
            pass

    server._sessions["status-profile-home"] = _session(profile_home)
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "status-profile-home"}}
        )
        output = resp["result"]["output"]
        assert f"Path: {profile_home}" in output
        assert f"Path: {launch_home}" not in output
    finally:
        server._sessions.pop("status-profile-home", None)


def test_session_status_path_without_profile_home_keeps_launch_home(monkeypatch, tmp_path):
    launch_home = tmp_path / "launch"
    launch_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    class LaunchDB:
        def get_session(self, key):
            return {"id": key, "title": "launch", "started_at": 1}

    server._sessions["status-launch-home"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "status-launch-home"}}
        )
        assert f"Path: {launch_home}" in resp["result"]["output"]
    finally:
        server._sessions.pop("status-launch-home", None)
