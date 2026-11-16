"""Explicit legacy UI /new owns reset hooks, not agent construction (#156607)."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


@pytest.fixture
def lifecycle(monkeypatch):
    hook = Mock(return_value=[])
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)
    monkeypatch.setattr("hermes_cli.lifecycle._plugin_hooks", hook)
    for name in ("_wire_session_agent", "_register_session_cwd", "_hydrate_session_cwd",
                 "_session_todo_state", "_emit", "_schedule_mcp_late_refresh",
                 "_schedule_agent_build", "_schedule_session_cap_enforcement"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a: threading.Event())
    monkeypatch.setattr(server, "_session_info", lambda *a: {})
    monkeypatch.setattr(server, "_config_model_target", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(server, "_probe_config_health", lambda *a: None)
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, **kw: SimpleNamespace(session_id=key))
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(server, "_sessions", {})
    yield hook
    server._sessions.clear()


def call(method, **params):
    return server.handle_request({"id": method, "method": method, "params": params})


def resets(hook):
    return [c.kwargs for c in hook.call_args_list if c.args == ("on_session_reset",)]


def test_explicit_new_finalizes_then_emits_one_reset_with_actual_stored_ids(lifecycle):
    old_sid, old_key = "old-runtime", "compressed-stored-tip"
    server._init_session(old_sid, "stale-parent-key", SimpleNamespace(session_id=old_key), [], source="tui")
    lifecycle.reset_mock()
    created = call("session.create", cols=80, source="tui")["result"]
    new_sid, new_key = created["session_id"], created["stored_session_id"]
    assert new_key != old_key
    assert resets(lifecycle) == []  # merely opening another conversation is not /new

    closed = call("session.close", session_id=old_sid, reset_to_session_id=new_sid)
    assert closed["result"]["closed"] is True
    assert resets(lifecycle) == [{"session_id": new_key, "old_session_id": old_key,
                                  "new_session_id": new_key, "platform": "tui"}]
    events = [c.args[0] for c in lifecycle.call_args_list]
    assert events.index("on_session_finalize") < events.index("on_session_reset")
    assert old_sid not in server._sessions

    current = server._sessions[new_sid]
    server._start_agent_build(new_sid, current)
    assert current["agent_ready"].wait(timeout=10)
    assert current["agent_error"] is None
    assert current["agent"].session_id == new_key
    server._start_session_services(new_sid, new_key, current)
    assert len(resets(lifecycle)) == 1  # deferred build/rebuild cannot duplicate it
    assert call("session.close", session_id=old_sid, reset_to_session_id=new_sid)["result"]["closed"] is False
    assert len(resets(lifecycle)) == 1  # a repeated close cannot duplicate it either


@pytest.mark.parametrize("replacement", ["missing", "self", "same-key", "other-profile", "other-client", "other-user"])
def test_invalid_replacement_does_not_close_or_reset(lifecycle, replacement):
    old_sid, key = "old-runtime", "old-stored"
    server._init_session(old_sid, key, SimpleNamespace(session_id=key), [], source="tui")
    target = "replacement-runtime"
    if replacement == "self":
        target = old_sid
    elif replacement != "missing":
        server._init_session(target, key if replacement == "same-key" else "new-stored",
                             SimpleNamespace(session_id=key if replacement == "same-key" else "new-stored"), [], source="tui")
        if replacement == "other-profile":
            server._sessions[target]["profile_home"] = "/not-the-owning-profile"
        elif replacement == "other-client":
            server._sessions[target]["transport"] = object()
        elif replacement == "other-user":
            server._sessions[target]["auth_user_id"] = "another-login"
    lifecycle.reset_mock()
    result = call("session.close", session_id=old_sid, reset_to_session_id=target)
    assert "error" in result
    assert old_sid in server._sessions
    lifecycle.assert_not_called()


def test_ordinary_initial_open_and_close_are_not_reset(lifecycle):
    created = call("session.create", cols=80)["result"]
    sid = created["session_id"]
    record = server._sessions[sid]
    server._start_agent_build(sid, record)
    assert record["agent_ready"].wait(timeout=10)
    assert record["agent_error"] is None
    assert call("session.close", session_id=sid)["result"]["closed"] is True
    assert resets(lifecycle) == []


def test_reset_hook_runs_in_predecessor_profile_scope(lifecycle, tmp_path):
    from hermes_constants import get_hermes_home

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("model: test\n")
    for sid in ("old", "new"):
        server._init_session(sid, sid + "-stored", SimpleNamespace(session_id=sid + "-stored"), [],
                             source="desktop", profile_home=str(home))
    observed = []

    def observe(event, **kwargs):
        if event == "on_session_reset":
            observed.append(get_hermes_home())
        return []

    lifecycle.side_effect = observe
    result = call("session.close", session_id="old", reset_to_session_id="new")
    assert result["result"]["closed"] is True
    assert observed == [home]
    assert resets(lifecycle)[0]["platform"] == "desktop"
