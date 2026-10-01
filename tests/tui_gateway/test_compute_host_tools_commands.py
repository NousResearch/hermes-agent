"""Isolated tool commands update the profile and the owning conversation."""

import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from tui_gateway.compute_host_bridge import _session_uses_compute_host
from tui_gateway.method_ctx import rebind
from tui_gateway.model_switch import _session_profile_runtime_scope


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("action,target,expected", [
    ("enable", "web", ["terminal", "web"]),
    ("disable", "terminal", []),
    ("disable", "local:tool", ["terminal"]),
])
def test_isolated_tools_update_only_the_owning_profile_and_conversation(
    tmp_path, monkeypatch, active, action, target, expected,
):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tui_gateway import server

    runtime_scope = rebind(_session_profile_runtime_scope, vars(server))
    uses_compute_host = rebind(_session_uses_compute_host, vars(server))
    from tui_gateway.compute_host import ComputeHost

    home = tmp_path / ".hermes"
    secondary = home / "profiles" / "worker"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = {"platform_toolsets": {"cli": ["terminal"]}, "dashboard": {"turn_isolation": True},
              "mcp_servers": {"local": {"command": "unused"}}}
    for path in (home, secondary):
        (path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(server, "_session_info", lambda agent, session: {
        "tools": {"core": getattr(agent, "enabled_toolsets", [])}})
    monkeypatch.setattr(server, "_emit", lambda *_: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_: None)
    built = []

    def make_agent(*args, **kwargs):
        assert server.os.environ.get("HERMES_COMPUTE_HOST_CHILD") == "1", "parent must not build an agent"
        from hermes_cli.config import load_config_readonly
        built.append(get_hermes_home())
        return SimpleNamespace(_session_db=kwargs["session_db"], _owns_session_db=False,
                               enabled_toolsets=load_config_readonly()["platform_toolsets"]["cli"])

    monkeypatch.setattr(server, "_make_agent", make_agent)
    observed = []
    token = set_multiplex_context(True)
    stores = []
    stdout = io.StringIO()
    host = ComputeHost(stdout=stdout, heartbeat_secs=0)
    try:
        for index, path in enumerate((home, secondary, home)):
            (path / "config.yaml").write_text(yaml.safe_dump(config))
            db = SessionDB(db_path=path / "state.db")
            stores.append(db)
            sid = f"isolated-tools-{index}"
            history = [{"role": "user", "content": "old conversation"}]
            session = {"agent": None, "agent_ready": threading.Event(), "profile_home": str(path),
                       "session_key": sid, "history": history, "history_lock": threading.Lock(),
                       "history_version": 3, "_compute_host_active": active,
                       "_metadata_mirror": {"tools": {"core": ["old_tool"]}}}
            child = {**session, "agent": SimpleNamespace(_session_db=db, _owns_session_db=False),
                     "history": list(history), "history_lock": threading.Lock()}
            monkeypatch.setitem(server._sessions, sid, session)

            def send_control(_sid, *, route_name, payload):
                assert (_sid, route_name) == (sid, "tools.configure")
                with monkeypatch.context() as child_scope:
                    child_scope.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
                    child_scope.setitem(server._sessions, sid, child)
                    host._handle_control({"sid": sid, "request_id": index, "route_name": route_name, **payload})
                return json.loads(stdout.getvalue().splitlines()[-1])

            monkeypatch.setattr(server, "_send_compute_host_control", send_control)
            with runtime_scope(session):
                assert uses_compute_host(session)
                response = server._methods["slash.exec"](index, {
                    "session_id": sid, "command": f"/tools {action} {target}"})
            assert "error" not in response, response
            saved = yaml.safe_load((path / "config.yaml").read_text())
            observed.append({"home": path, "pin": saved["platform_toolsets"]["cli"],
                             "excluded": saved["mcp_servers"]["local"].get("tools", {}).get("exclude", []),
                             "parent_agent": session["agent"], "version": session["history_version"],
                             "owner_history": child["history"] if active else session["history"]})
        excluded = ["tool"] if ":" in target else []
        assert observed == [{"home": path, "pin": expected, "excluded": excluded,
                             "parent_agent": None, "version": 4, "owner_history": []}
                            for path in (home, secondary, home)]
        assert built == ([home, secondary, home] if active else [])
        assert get_hermes_home() == home
    finally:
        host.close()
        reset_multiplex_context(token)
        for db in stores:
            db.close()


@pytest.mark.parametrize("command,busy", [
    ("/tools list", False), ("/tools enable terminal", False), ("/tools enable web", True),
])
def test_isolated_tools_read_noop_and_busy_paths_preserve_the_conversation(
    tmp_path, monkeypatch, command, busy,
):
    from tui_gateway import server

    runtime_scope = rebind(_session_profile_runtime_scope, vars(server))

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = {"platform_toolsets": {"cli": ["terminal"]}, "dashboard": {"turn_isolation": True}}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    history = [{"role": "user", "content": "unchanged"}]
    session = {"agent": None, "agent_ready": threading.Event(), "profile_home": str(home),
               "session_key": "preserved-tools", "history": history,
               "history_lock": threading.Lock(), "history_version": 3, "running": busy}
    monkeypatch.setitem(server._sessions, "preserved-tools", session)
    with runtime_scope(session):
        response = server._methods["slash.exec"](1, {"session_id": "preserved-tools", "command": command})
    result = response.get("result") or {}
    assert {"error": (response.get("error") or {}).get("code"),
            "history": session["history"], "version": session["history_version"],
            "config": yaml.safe_load((home / "config.yaml").read_text())["platform_toolsets"],
            "listed": "terminal: enabled" in result.get("output", "")} == {
        "error": 4009 if busy else None, "history": history, "version": 3,
        "config": {"cli": ["terminal"]}, "listed": command == "/tools list"}
