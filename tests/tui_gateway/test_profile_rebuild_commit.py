"""Session-only tools ingress and rebuild failure preserve profile isolation."""
import threading
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml


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
    monkeypatch.setitem(server._sessions, "profile-tools", {"profile_home": str(profile)})
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


def test_rebuild_keeps_session_runtime_picks_but_new_clears_them(monkeypatch):
    """Regression for #127449: a rebuild is not a conversation boundary, /new is.
    /model, /reasoning and /fast are all session pins the rebuild must carry."""
    from tui_gateway import server

    pick = {"model": "pick-b", "provider": "openrouter"}
    reasoning = {"enabled": True, "effort": "high"}
    carried = ("model_override", "reasoning_config_override", "service_tier_override")
    seen = []
    def make_agent(*_args, **kwargs):
        seen.append({k: kwargs.get(k) for k in carried})
        return SimpleNamespace(_session_db=None, _owns_session_db=False)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_config_model_target", lambda: "default-a")
    session = {"agent": None, "session_key": "k", "model_override": dict(pick),
               "create_reasoning_override": reasoning, "create_service_tier_override": "priority"}
    server._rebuild_session_agent("sid", session, session_id="k")
    assert seen == [dict(model_override=pick, reasoning_config_override=reasoning,
                         service_tier_override="priority")]
    server._rebuild_session_agent("sid", session, model_override={"model": "explicit"})
    assert seen[-1]["model_override"] == {"model": "explicit"}
    for pin in ("model_override", "create_reasoning_override", "create_service_tier_override"):
        session.pop(pin)
    server._rebuild_session_agent("sid", session)
    assert seen[-1] == dict.fromkeys(carried)


def test_rebuild_finishing_after_close_closes_the_replacement_and_its_handle(tmp_path, monkeypatch):
    """session.close claiming the record while _make_agent runs already tore down the agent it saw:
    the replacement must be closed (and its dedicated handle released), never installed (#49852)."""
    from tui_gateway import server

    closed: list[str] = []
    handle = SimpleNamespace(close=lambda: closed.append("db"))
    monkeypatch.setattr(server, "_open_profile_session_db", lambda _home: handle)
    monkeypatch.setattr(server, "_config_model_target", lambda: "default-a")
    session = {"agent": None, "session_key": "k", "profile_home": str(tmp_path / "worker")}

    def make_agent(*_args, session_db=None, **_kwargs):
        session["_closing"] = True  # _pop_session_by_id lands mid-build
        return SimpleNamespace(_session_db=session_db, _owns_session_db=False,
                               close=lambda: closed.append("agent"))

    monkeypatch.setattr(server, "_make_agent", make_agent)
    with pytest.raises(RuntimeError, match="closed"):
        server._rebuild_session_agent("sid", session, session_id="k")
    assert session["agent"] is None
    assert closed == ["agent", "db"]


@pytest.mark.parametrize("compressing", [False, True])
def test_tools_configure_preserves_busy_session_and_profile(tmp_path, monkeypatch, compressing):
    from tui_gateway import server
    from tui_gateway.user_messages import busy_message

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = "platform_toolsets:\n  cli: [file]\n"
    for path in (home, profile):
        (path / "config.yaml").write_text(config)
    before = [(path / "config.yaml").read_bytes() for path in (home, profile)]
    session = {
        "profile_home": str(profile), "running": True,
        "_manual_compress_active": compressing, "history_version": 7,
        "history": [{"role": "user", "content": "active input"}],
    }
    unchanged = {**session, "history": list(session["history"])}
    resets = []
    monkeypatch.setitem(server._sessions, "busy-tools", session)
    monkeypatch.setattr(server, "_reset_session_agent", lambda *args: resets.append(args) or {})

    response = server._methods["tools.configure"](
        "busy", {"session_id": "busy-tools", "action": "enable", "names": ["web"]},
    )

    assert {
        "response": response, "session": session, "resets": resets,
        "configs": [(path / "config.yaml").read_bytes() for path in (home, profile)],
    } == {
        "response": {"jsonrpc": "2.0", "id": "busy", "error": {
            "code": 4009, "message": busy_message("tools", compressing),
        }},
        "session": unchanged, "resets": [], "configs": before,
    }
