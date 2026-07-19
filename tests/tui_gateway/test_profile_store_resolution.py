"""Profile-targeted TUI state-store resolution."""

from pathlib import Path

from hermes_state import SessionDB
from tui_gateway import server


def test_profile_home_does_not_require_state_db(monkeypatch, tmp_path):
    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "remote"
    launch_home.mkdir()
    profile_home.mkdir()

    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda _name: profile_home)
    monkeypatch.setattr(server, "_hermes_home", str(launch_home))

    assert server._profile_home("remote") == profile_home
    assert not (profile_home / "state.db").exists()


def test_profile_session_context_uses_home_resolver(monkeypatch, tmp_path):
    profile_home = tmp_path / "remote"
    profile_home.mkdir()
    opened = []

    class FakeStore:
        def close(self):
            pass

    fake_store = FakeStore()

    def open_for_home(home: Path, *, read_only=False):
        opened.append((home, read_only))
        return fake_store

    monkeypatch.setattr(SessionDB, "for_home", open_for_home)

    with server._session_db({"profile_home": str(profile_home)}) as store:
        assert store is fake_store

    assert opened == [(profile_home, False)]
    assert not (profile_home / "state.db").exists()


def test_selected_profile_resume_store_failure_fails_closed(monkeypatch, tmp_path):
    profile_home = tmp_path / "remote"
    profile_home.mkdir()
    launch_store_accessed = False

    def fail_open(home: Path, *, read_only=False, config=None, environ=None):
        assert home == profile_home
        raise RuntimeError("selected store unavailable")

    def launch_store():
        nonlocal launch_store_accessed
        launch_store_accessed = True
        raise AssertionError("must not substitute the launch-profile store")

    monkeypatch.setattr(SessionDB, "for_home", fail_open)
    monkeypatch.setattr(server, "_profile_home", lambda _profile: profile_home)
    monkeypatch.setattr(server, "_get_db", launch_store)

    response = server.handle_request({
        "id": "resume",
        "method": "session.resume",
        "params": {"session_id": "target", "profile": "remote"},
    })

    assert response["error"]["code"] == 5000
    assert response["error"]["message"] == "session store unavailable"
    assert not launch_store_accessed


def test_unresolvable_named_profile_resume_never_uses_launch_store(monkeypatch):
    from hermes_cli import profiles as profiles_mod

    launch_store_accessed = False

    def launch_store():
        nonlocal launch_store_accessed
        launch_store_accessed = True
        raise AssertionError("must not substitute the launch-profile store")

    monkeypatch.setattr(
        profiles_mod,
        "get_profile_dir",
        lambda _name: (_ for _ in ()).throw(RuntimeError("profile lookup failed")),
    )
    monkeypatch.setattr(server, "_get_db", launch_store)

    response = server.handle_request({
        "id": "resume",
        "method": "session.resume",
        "params": {"session_id": "target", "profile": "missing"},
    })

    assert response["error"]["code"] == 4025
    assert "profile unavailable" in response["error"]["message"]
    assert not launch_store_accessed


def test_deferred_profile_build_store_failure_never_uses_launch_store(monkeypatch, tmp_path):
    profile_home = tmp_path / "remote"
    profile_home.mkdir()
    sid = "deferred-profile"
    session = {
        "agent": None,
        "agent_ready": __import__("threading").Event(),
        "profile_home": str(profile_home),
        "session_key": "profile-key",
    }
    launch_store_accessed = False
    make_agent_called = False

    def fail_open(home: Path, *, read_only=False, config=None, environ=None):
        assert home == profile_home
        raise RuntimeError("selected store unavailable")

    def launch_store():
        nonlocal launch_store_accessed
        launch_store_accessed = True
        raise AssertionError("must not substitute the launch-profile store")

    def make_agent(*_args, **_kwargs):
        nonlocal make_agent_called
        make_agent_called = True
        raise AssertionError("agent build must stop before launch fallback")

    monkeypatch.setattr(SessionDB, "for_home", fail_open)
    monkeypatch.setattr(server, "_get_db", launch_store)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_set_session_context", lambda _key: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert session["agent_ready"].wait(timeout=3), "agent build did not finish"
        assert session["agent"] is None
        assert "selected profile session database unavailable" in session["agent_error"]
        assert not launch_store_accessed
        assert not make_agent_called
    finally:
        server._sessions.clear()
