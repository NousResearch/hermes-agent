"""Session-only tools ingress and rebuild failure preserve profile isolation."""
import threading
from types import SimpleNamespace

import pytest
import yaml


@pytest.mark.parametrize("explicit_profile", [None, "default"])
def test_tools_configure_uses_live_session_profile(tmp_path, monkeypatch, explicit_profile):
    from tui_gateway import server
    from hermes_constants import get_hermes_home

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = {"platform_toolsets": {"cli": ["terminal", "web"]}}
    for path in (home, profile):
        (path / "config.yaml").write_text(yaml.safe_dump(config))
    launch_before = (home / "config.yaml").read_bytes()
    seen = []
    monkeypatch.setattr(server, "_reset_session_agent", lambda *_: seen.append(get_hermes_home()) or {})
    monkeypatch.setitem(server._sessions, "profile-tools", {
        "profile_home": str(profile),
        "profile_incarnation": server._capture_profile_incarnation(profile),
    })
    params = {"session_id": "profile-tools", "action": "disable", "names": ["terminal"]}
    if explicit_profile is not None:
        params["profile"] = explicit_profile
    response = server._methods["tools.configure"](1, params)
    assert "error" not in response
    assert (home / "config.yaml").read_bytes() == launch_before
    assert "terminal" not in yaml.safe_load((profile / "config.yaml").read_text())["platform_toolsets"]["cli"]
    assert seen == [profile]
    assert get_hermes_home() == home
    worker_before = (profile / "config.yaml").read_bytes()
    monkeypatch.delitem(server._sessions, "profile-tools")
    response = server._methods["tools.configure"](2, params)
    assert response["error"]["code"] == 4001
    assert (home / "config.yaml").read_bytes() == launch_before
    assert (profile / "config.yaml").read_bytes() == worker_before
    response = server._methods["tools.configure"](3, {"action": "disable", "names": ["terminal"]})
    assert "error" not in response and not response["result"]["reset"]
    assert (home / "config.yaml").read_bytes() != launch_before
    assert (profile / "config.yaml").read_bytes() == worker_before


@pytest.mark.parametrize("path", ["reset", "capabilities"])
@pytest.mark.parametrize("has_agent_db", [True, False])
def test_rebuild_preparation_failure_keeps_reachable_owner(tmp_path, monkeypatch, path, has_agent_db):
    from tui_gateway import server
    from hermes_state import SessionDB
    from hermes_constants import get_hermes_home

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    db = SessionDB(db_path=profile / "state.db")
    old = SimpleNamespace(_session_db=db if has_agent_db else None,
                          _owns_session_db=has_agent_db, _session_title_hint="Bot Chat")
    session = {"agent": old, "profile_home": str(profile), "session_key": "profile-key",
               "bot_caps_seen": "before", "source": "desktop", "cwd": str(tmp_path),
               "history": [], "history_lock": threading.Lock(), "history_version": 0}
    built = []
    def make_agent(*_args, session_db=None, **_kwargs):
        replacement = SimpleNamespace(_session_db=session_db, _owns_session_db=False)
        built.append(replacement)
        return replacement
    def fail_config():
        raise RuntimeError("preparation failed")
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_config_model_target", fail_config)
    monkeypatch.setattr("tools.bot_mode_probe.capability_fingerprint", lambda _: "after")
    try:
        if path == "reset":
            with pytest.raises(RuntimeError, match="preparation failed"):
                server._reset_session_agent("profile-tools", session)
        else:
            server._sync_bot_capabilities("profile-tools", session)
        assert session["agent"] is old
        assert old._owns_session_db is has_agent_db
        assert not built, "prepare config before allocating a replacement"
        assert get_hermes_home() == home
        db.create_session("still-owned", "tui")
    finally:
        db.close()
        for agent in built:
            if agent._session_db is not db and agent._owns_session_db:
                agent._session_db.close()


@pytest.mark.parametrize(("boundary", "change", "has_agent_db"), [
    ("bind", None, False),
    ("publish", None, True),
    ("bind", "recreate", False),
    ("publish", "recreate", False),
    ("publish", "retire", True),
    ("publish", "closing", True),
    ("publish", "replace", True),
    ("publish", "rebind", True),
])
def test_rebuild_commits_only_to_live_profile_record(
    tmp_path, monkeypatch, boundary, change, has_agent_db,
):
    from hermes_cli.profile_incarnation import write_fresh_profile_incarnation
    from hermes_cli.profile_lifecycle import profile_lifecycle_lease
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tui_gateway import server
    from tui_gateway.profile_lifecycle import ProfileLifecycleFence

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(server, "_profile_lifecycle", ProfileLifecycleFence())
    monkeypatch.setattr(server, "_get_db", lambda: None)
    incarnation = server._capture_profile_incarnation(profile)
    db = SessionDB(db_path=profile / "state.db") if has_agent_db else None
    old = SimpleNamespace(_session_db=db, _owns_session_db=has_agent_db)
    session = {"agent": old, "profile_home": str(profile), "profile_incarnation": incarnation,
               "session_key": "profile-key", "config_model_seen": ("before", "")}
    monkeypatch.setitem(server._sessions, "profile-rebuild", session)
    built, closed, connections = [], [], []

    def change_target():
        with profile_lifecycle_lease():
            if change in {"retire", "recreate"}:
                server._profile_lifecycle.retire(profile, incarnation)
            if change == "recreate":
                fresh = write_fresh_profile_incarnation(profile)
                assert fresh != incarnation
                server.allow_profile_home(profile, fresh)
            if change == "rebind":
                other = home / "profiles" / "other"
                other.mkdir()
                session.update(profile_home=str(other),
                               profile_incarnation=server._capture_profile_incarnation(other))
            if change == "closing":
                session["_closing"] = True
            if change == "replace":
                server._sessions["profile-rebuild"] = dict(session)

    def config_target():
        if boundary == "bind":
            change_target()
        return ("after", "")

    def make_agent(*_args, session_db=None, **_kwargs):
        assert get_hermes_home() == profile
        assert session_db is not None
        # Retain the real connection to prove a rejected fresh acquisition is released.
        with session_db._read_ctx() as conn:
            connections.append(conn)
        replacement = SimpleNamespace(_session_db=session_db, _owns_session_db=False)
        replacement.close = lambda: closed.append(replacement._owns_session_db)
        built.append(replacement)
        if boundary == "publish":
            change_target()
        return replacement

    monkeypatch.setattr(server, "_config_model_target", config_target)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    try:
        if change is None:
            replacement = server._rebuild_session_agent("profile-rebuild", session)
            assert session["agent"] is replacement
            assert session["config_model_seen"] == ("after", "")
            assert replacement._owns_session_db is True
            assert old._owns_session_db is False
            assert not closed
            assert get_hermes_home() == home
            replacement._session_db.create_session("still-owned", "tui")
            assert replacement._session_db.get_session("still-owned") is not None
            assert not (home / "state.db").exists()
            return
        with pytest.raises((FileNotFoundError, RuntimeError)):
            server._rebuild_session_agent("profile-rebuild", session)
        assert session["agent"] is old
        assert session["config_model_seen"] == ("before", "")
        assert old._owns_session_db is has_agent_db
        assert get_hermes_home() == home
        if boundary == "bind":
            assert not built
            assert not (profile / "state.db").exists()
        else:
            assert closed == [False], "discard the replacement without borrowing DB ownership"
            if db is not None:
                assert connections[0].execute("SELECT 1").fetchone()[0] == 1
            else:
                import sqlite3
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    connections[0].execute("SELECT 1")
        if change == "replace":
            assert server._sessions["profile-rebuild"]["agent"] is old
    finally:
        if db is not None:
            db.close()
        for agent in built:
            if agent._session_db is not db:
                agent._session_db.close()


@pytest.mark.parametrize("boundary", ["resolved", "save"])
def test_tools_configure_cannot_write_across_profile_recreation(tmp_path, monkeypatch, boundary):
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    from hermes_cli import config as hc, profile_lifecycle
    from hermes_cli.profile_incarnation import write_fresh_profile_incarnation
    from hermes_constants import get_hermes_home
    from tui_gateway import server
    from tui_gateway.profile_lifecycle import ProfileLifecycleFence

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(server, "_profile_lifecycle", ProfileLifecycleFence())
    config = {"platform_toolsets": {"cli": ["terminal", "web"]}}
    for path in (home, profile):
        (path / "config.yaml").write_text(yaml.safe_dump(config))
    launch_before = (home / "config.yaml").read_bytes()
    incarnation = server._capture_profile_incarnation(profile)
    session = {"profile_home": str(profile), "profile_incarnation": incarnation}
    monkeypatch.setitem(server._sessions, "profile-configure", session)
    resets, mutation_attempts = [], []
    monkeypatch.setattr(server, "_reset_session_agent", lambda *_: resets.append(get_hermes_home()) or {})
    replacement_config = yaml.safe_dump({"platform_toolsets": {"cli": ["file", "web"]}})

    def recreate():
        with profile_lifecycle.profile_lifecycle_lease():
            server._profile_lifecycle.retire(profile, incarnation)
            shutil.rmtree(profile)
            profile.mkdir()
            fresh = write_fresh_profile_incarnation(profile)
            server.allow_profile_home(profile, fresh)
            (profile / "config.yaml").write_text(replacement_config)

    if boundary == "resolved":
        resolve = server._sess_nowait

        def resolve_then_recreate(params, rid):
            result = resolve(params, rid)
            assert result == (session, None)
            recreate()
            return result

        monkeypatch.setattr(server, "_sess_nowait", resolve_then_recreate)
    else:
        save_config = hc.save_config

        def try_recreate():
            # Nonblocking on another thread makes this a deterministic exclusion probe,
            # not an assertion that a scheduled delete happened to be slow.
            lock = profile_lifecycle._PROFILE_LIFECYCLE_LOCK
            if not lock.acquire(blocking=False):
                return False
            try:
                recreate()
                return True
            finally:
                lock.release()

        def save_during_recreate(cfg):
            with ThreadPoolExecutor(max_workers=1) as pool:
                mutation_attempts.append(pool.submit(try_recreate).result(timeout=10))
            save_config(cfg)

        monkeypatch.setattr(hc, "save_config", save_during_recreate)

    params = {"session_id": "profile-configure", "action": "disable", "names": ["terminal"]}
    response = server._methods["tools.configure"](1, params)
    assert (home / "config.yaml").read_bytes() == launch_before
    assert get_hermes_home() == home
    if boundary == "resolved":
        assert "error" in response
        assert (profile / "config.yaml").read_text() == replacement_config
        assert not resets
    else:
        assert "error" not in response
        assert mutation_attempts == [False], "the config write must exclude profile mutation"
        assert resets == [profile]
        assert "terminal" not in yaml.safe_load((profile / "config.yaml").read_text())["platform_toolsets"]["cli"]
        recreate()
        response = server._methods["tools.configure"](2, params)
        assert response["error"]["code"] == 4041
        assert (profile / "config.yaml").read_text() == replacement_config
