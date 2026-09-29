"""Camofox session bookkeeping must retain its owning profile across cleanup."""

from agent import secret_scope
from gateway.run import _profile_runtime_scope
from hermes_constants import hermes_home_key


def test_camofox_cache_and_cleanup_are_profile_scoped(tmp_path, monkeypatch):
    from tools import browser_camofox as camofox
    from tools import browser_tool as browser_tool
    from tools import browser_tool_cdp as cdp
    from tools import browser_tool_lifecycle as lifecycle
    from tools.browser_task_identity import browser_task_key

    homes = []
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for name in ("a", "b"):
            home = tmp_path / name
            home.mkdir()
            (home / "config.yaml").write_text(
                "browser:\n  backend: camofox\n  camofox:\n    managed_persistence: true\n",
                encoding="utf-8",
            )
            (home / ".env").write_text(
                f"CAMOFOX_URL=http://camofox-{name}.invalid\n", encoding="utf-8")
            homes.append(home)

        camofox._sessions.clear()
        monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
        monkeypatch.setattr(cdp, "_stop_cdp_supervisor", lambda *_args: None)

        with _profile_runtime_scope(homes[0]):
            assert camofox.get_camofox_url() == "http://camofox-a.invalid"
            a_session = camofox._get_session("same-api-session")
            a_session["synthetic_marker"] = "profile-a"
            a_key = browser_task_key("same-api-session")
            a_identity = camofox.get_camofox_identity("same-api-session")

        with _profile_runtime_scope(homes[1]):
            assert camofox.get_camofox_url() == "http://camofox-b.invalid"
            b_session = camofox._get_session("same-api-session")
            b_session["synthetic_marker"] = "profile-b"
            b_key = browser_task_key("same-api-session")
            b_identity = camofox.get_camofox_identity("same-api-session")
            assert b_session is not a_session
            assert "synthetic_marker" not in a_session or a_session["synthetic_marker"] == "profile-a"
            assert a_identity["user_id"] != b_identity["user_id"]
            assert a_identity["session_key"] != b_identity["session_key"]

        # Cleanup is often run later by the global janitor. The typed owner key must
        # still select B even when the caller's current context belongs to A.
        with _profile_runtime_scope(homes[0]):
            lifecycle._cleanup_single_browser_session(b_key)
            assert b_key not in camofox._sessions
            assert camofox._sessions[a_key] is a_session
            assert a_session["synthetic_marker"] == "profile-a"

        with _profile_runtime_scope(homes[0]):
            assert camofox._get_session("same-api-session") is a_session
        assert str(hermes_home_key(str(homes[0]))) != str(hermes_home_key(str(homes[1])))
    finally:
        camofox._sessions.clear()
        secret_scope.set_multiplex_active(previous_multiplex)


def test_shutdown_uses_each_profiles_backend_and_persistence(tmp_path, monkeypatch):
    from tools import browser_camofox as camofox
    from tools import browser_tool as browser_tool
    from tools import browser_tool_cdp as cdp
    from tools import browser_tool_lifecycle as lifecycle
    from tools.browser_task_identity import browser_task_key

    for field in ("_active_sessions", "_session_last_activity", "_session_owner_homes",
                  "_cleanup_failures", "_last_active_session_key"):
        monkeypatch.setattr(browser_tool, field, {})
    monkeypatch.setattr(camofox, "_sessions", {})
    monkeypatch.setattr(cdp, "_stop_cdp_supervisor", lambda *_args: None)
    monkeypatch.setattr(browser_tool, "_maybe_stop_recording", lambda *_args: None)
    deleted = []
    monkeypatch.setattr(camofox, "_delete", lambda path, **_kwargs: deleted.append((camofox.get_camofox_url(), path)))
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    homes, users = {}, {}
    try:
        for name, persistent in (("a", True), ("b", False)):
            home = tmp_path / name
            home.mkdir()
            (home / "config.yaml").write_text(
                "browser:\n  backend: camofox\n  camofox:\n    managed_persistence: "
                + str(persistent).lower() + "\n", encoding="utf-8")
            (home / ".env").write_text(f"CAMOFOX_URL=http://camofox-{name}.invalid\n", encoding="utf-8")
            homes[name] = home
            with _profile_runtime_scope(home):
                key = browser_task_key("same-api-session")
                users[name] = camofox._get_session("same-api-session")["user_id"]
                # A task can have supervised state as well as Camofox state after
                # changing its browser configuration. Record its real owner.
                lifecycle._update_session_activity(key)
                browser_tool._active_sessions[key] = {
                    "session_name": "", "bb_session_id": None, "cdp_url": None,
                    "expires_at": 1,
                }
        with _profile_runtime_scope(homes["a"]):
            lifecycle.cleanup_all_browsers()
        assert deleted == [("http://camofox-b.invalid", f"/sessions/{users['b']}")]
        assert not camofox._sessions
        assert not browser_tool._active_sessions
    finally:
        secret_scope.set_multiplex_active(previous)
