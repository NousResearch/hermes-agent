"""Session-scoped RPCs on a named-profile session use that profile's ``state.db``, never the launch one.

``session.title`` / ``session.history`` / ``session.status`` and teardown resolve the store through
``_session_db(session)``; the session carries the profile incarnation it was opened under.
"""

from tests.tui_gateway.test_tui_gateway_server import _TEST_PROFILE_INCARNATION, _stamp_test_profile_home
from tui_gateway import server


def test_session_title_uses_session_profile_db_not_launch(monkeypatch, tmp_path):
    """session.title on a non-launch profile session must not touch launch DB."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    _stamp_test_profile_home(profile_home)
    seen: dict = {}

    class LaunchDB:
        def get_session_title(self, _key):
            seen["launch_read"] = True
            return "from-launch"

        def set_session_title(self, _key, _title):
            seen["launch_write"] = True
            return True

        def get_session(self, _key):
            return {"id": _key, "title": "from-launch"}

    class ProfileDB:
        def __init__(self, db_path=None, **_kwargs):
            self.db_path = db_path
            seen["db_path"] = db_path

        def get_session_title(self, _key):
            return seen.get("title")

        def get_session(self, _key):
            if "title" in seen:
                return {"id": _key, "title": seen["title"]}
            return None

        def set_session_title(self, _key, title):
            seen["title"] = title
            seen["profile_write"] = True
            return True

        def close(self):
            seen["closed"] = True

    server._sessions["sid"] = {
        "session_key": "ml-sess",
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "pending_title": None,
        "profile_home": str(profile_home),
        "profile_incarnation": _TEST_PROFILE_INCARNATION,
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "profile-title"},
            }
        )
        assert "result" in set_resp, set_resp
        assert set_resp["result"]["title"] == "profile-title"
        assert seen.get("profile_write") is True
        assert seen.get("launch_write") is None
        assert str(seen.get("db_path")).endswith("state.db")

        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "profile-title"
        assert seen.get("launch_read") is None
    finally:
        server._sessions.pop("sid", None)


def test_session_history_uses_session_profile_db(monkeypatch, tmp_path):
    """session.history must read durable messages from the profile state.db."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    _stamp_test_profile_home(profile_home)
    seen: dict = {}

    class LaunchDB:
        def get_messages_as_conversation(self, _key, include_ancestors=True, **_kwargs):
            seen["launch"] = True
            return [{"role": "user", "content": "launch"}]

    class ProfileDB:
        def __init__(self, db_path=None, **_kwargs):
            seen["db_path"] = db_path

        def get_messages_as_conversation(self, _key, include_ancestors=True, **_kwargs):
            seen["profile"] = True
            return [{"role": "user", "content": "from-profile"}]

        def close(self):
            seen["closed"] = True

    server._sessions["sid"] = {
        "session_key": "ml-sess",
        "history": [{"role": "user", "content": "mem"}],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "profile_home": str(profile_home),
        "profile_incarnation": _TEST_PROFILE_INCARNATION,
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.history", "params": {"session_id": "sid"}}
        )
        assert "result" in resp, resp
        assert seen.get("profile") is True
        assert seen.get("launch") is None
        # Count comes from profile-backed conversation (1 msg), not bare mem list alone.
        assert resp["result"]["count"] == 1
    finally:
        server._sessions.pop("sid", None)


def test_session_status_uses_session_profile_db(monkeypatch, tmp_path):
    """session.status must load meta from the session profile state.db."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    _stamp_test_profile_home(profile_home)
    seen: dict = {}

    class LaunchDB:
        def get_session(self, _key):
            seen["launch"] = True
            return {"id": _key, "title": "launch-title", "started_at": 1}

    class ProfileDB:
        def __init__(self, db_path=None, **_kwargs):
            seen["db_path"] = db_path

        def get_session(self, _key):
            seen["profile"] = True
            return {"id": _key, "title": "profile-title", "started_at": 42}

        def close(self):
            seen["closed"] = True

    server._sessions["sid"] = {
        "session_key": "ml-sess",
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "profile_home": str(profile_home),
        "profile_incarnation": _TEST_PROFILE_INCARNATION,
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
        assert "result" in resp, resp
        assert "profile-title" in resp["result"]["output"]
        assert seen.get("profile") is True
        assert seen.get("launch") is None
    finally:
        server._sessions.pop("sid", None)


def test_teardown_ends_session_in_profile_db(monkeypatch, tmp_path):
    """_teardown_session must end_session on the profile store, not launch."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    _stamp_test_profile_home(profile_home)
    seen: dict = {}

    class LaunchDB:
        def get_session(self, _key):
            seen["launch"] = True
            return {"id": _key, "source": "tui"}

        def end_session(self, _key, _reason):
            seen["launch_end"] = True

    class ProfileDB:
        def __init__(self, db_path=None, **_kwargs):
            seen["db_path"] = db_path

        def get_session(self, _key):
            seen["profile"] = True
            return {"id": _key, "source": "tui"}

        def end_session(self, key, reason):
            seen["ended"] = (key, reason)

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    session = {
        "session_key": "ml-sess",
        "profile_home": str(profile_home),
        "profile_incarnation": _TEST_PROFILE_INCARNATION,
        "agent": None,
        "history": [],
        "source": "tui",
    }
    server._teardown_session(session, end_reason="closed")
    assert seen.get("ended") == ("ml-sess", "closed")
    assert seen.get("launch_end") is None
    assert seen.get("launch") is None
    assert str(seen.get("db_path")).endswith("state.db")
