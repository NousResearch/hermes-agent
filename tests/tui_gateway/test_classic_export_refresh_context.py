"""Real late-MCP thread policy isolation and staged classic grant publication."""

import contextvars
import json
import os
import threading

import pytest

import model_tools
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.mcp_tool_agent import refresh_agent_mcp_tools
from tools.registry import registry
from tui_gateway import classic_exports, server
from tests.tui_gateway.test_classic_export_tool_refresh import build, _tool_agent, _assert_share


@pytest.mark.parametrize("restriction", ["disabled", "deferred"])
@pytest.mark.parametrize("launch_restricts", [True, False])
def test_real_late_refresh_uses_owning_profile(tmp_path, monkeypatch, restriction, launch_restricts):
    import run_agent
    from tui_gateway import entry

    launch = tmp_path / "launch"
    writer = launch / "profiles" / "writer"
    writer.mkdir(parents=True)

    def config(restricted):
        return {
            "platform_toolsets": {"cli": ["file"]},
            "agent": {"disabled_toolsets": ["bot_room"] if restricted and restriction == "disabled" else []},
            "tools": {"tool_search": {
                "enabled": "on", "defer": ["share_group_file"] if restricted and restriction == "deferred" else []}},
        }

    (launch / "config.yaml").write_text(json.dumps(config(launch_restricts)))
    (writer / "config.yaml").write_text(json.dumps(config(False)))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *a: ("offline", {}))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda _: {})
    monkeypatch.setattr(run_agent, "AIAgent", _tool_agent)
    token = set_hermes_home_override(writer)
    try:
        agent = server._make_agent("context", "writer", platform_override="desktop")
        session = {"agent": agent, "session_key": "writer", "profile_home": str(writer), "room_plumbing": True}
        classic_exports.install_schema(session)
        _assert_share(agent, 1)
    finally:
        reset_hermes_home_override(token)

    (writer / "config.yaml").write_text(json.dumps(config(not launch_restricts)))
    token = set_hermes_home_override(writer)
    try:
        refresh_agent_mcp_tools(agent)
        _assert_share(agent, int(launch_restricts))
    finally:
        reset_hermes_home_override(token)

    monkeypatch.setattr(server, "_sessions", {"context": session})
    monkeypatch.setattr(server, "_session_info", lambda *a: {})
    monkeypatch.setattr(server, "_emit", lambda *a: None)
    monkeypatch.setattr(entry, "mcp_discovery_in_flight", lambda: True)
    monkeypatch.setattr(entry, "join_mcp_discovery", lambda **kw: True)
    real_thread = threading.Thread
    spawned = []

    def track_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        spawned.append(thread)
        return thread

    monkeypatch.setattr(server.threading, "Thread", track_thread)
    ambient_home = get_hermes_home()
    environment_home = os.environ["HERMES_HOME"]
    server._schedule_mcp_late_refresh("context", agent)
    assert len(spawned) == 1
    spawned[0].join(10)
    assert not spawned[0].is_alive()
    _assert_share(agent, int(launch_restricts))
    assert get_hermes_home() == ambient_home
    assert os.environ["HERMES_HOME"] == environment_home


def test_stale_snapshot_does_not_publish_classic_revocation(build, monkeypatch):
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    original_enabled = list(agent.enabled_toolsets)
    entered, release = threading.Event(), threading.Event()
    original_defs = model_tools.get_tool_definitions
    errors = []

    def delayed(**kwargs):
        definitions = original_defs(**kwargs)
        if kwargs.get("enabled_toolsets") == []:
            entered.set()
            assert release.wait(10)
        return definitions

    monkeypatch.setattr(model_tools, "get_tool_definitions", delayed)

    def older():
        try:
            refresh_agent_mcp_tools(agent, enabled_override=[])
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=contextvars.copy_context().run, args=(older,))
    thread.start()
    try:
        assert entered.wait(10)
        registry.register(name="classic_generation_probe", toolset="refresh_probe",
                          schema={"name": "classic_generation_probe", "parameters": {"type": "object"}},
                          handler=lambda args, **kw: "{}")
        refresh_agent_mcp_tools(agent, enabled_override=original_enabled)
        generation = agent._tool_snapshot_generation
        assert agent._classic_export_enabled is True
        _assert_share(agent, 1)
    finally:
        release.set()
        thread.join(10)
        registry.deregister("classic_generation_probe")
    assert not thread.is_alive() and not errors
    assert agent._tool_snapshot_generation == generation
    _assert_share(agent, 1)
    assert agent._classic_export_enabled is True


def test_revocation_commits_even_when_schema_snapshot_is_unchanged(build):
    agent, session = build(room_plumbing=True)
    agent.enabled_toolsets = []
    classic_exports.install_schema(session)
    refresh_agent_mcp_tools(agent, disabled_override=["bot_room"])
    _assert_share(agent, 0)
    assert agent._classic_export_enabled is True
    snapshot = agent.tools
    # The direct schema is already gone; explicit removal must still commit the flag.
    refresh_agent_mcp_tools(agent, enabled_override=[])
    assert agent.tools is snapshot
    assert agent._classic_export_enabled is False
    snapshot = agent.tools
    refresh_agent_mcp_tools(agent, enabled_override=[])
    assert agent.tools is snapshot
    assert agent._classic_export_enabled is False


@pytest.mark.parametrize("fail", [False, True])
def test_refresh_restores_callers_profile_and_stages_grant_until_publish(build, tmp_path, monkeypatch, fail):
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
    from tools.terminal_scope import get_terminal_scope
    from tools import mcp_tool_agent

    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    other = tmp_path / "other"
    other.mkdir()
    token = set_hermes_home_override(other)
    secret_token = set_secret_scope({"CLASSIC_TEST_SCOPE": "caller"})
    before_terminal = get_terminal_scope()
    original_publish = mcp_tool_agent._publish_tool_snapshot

    def publish(*args, **kwargs):
        assert str(get_hermes_home()) == session["profile_home"]
        assert agent._classic_export_enabled is True
        assert kwargs["staged_classic_enabled"] is False
        if fail:
            raise RuntimeError("publication failed")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(mcp_tool_agent, "_publish_tool_snapshot", publish)
    environment_home = os.environ.get("HERMES_HOME")
    try:
        if fail:
            with pytest.raises(RuntimeError, match="publication failed"):
                refresh_agent_mcp_tools(agent, enabled_override=[])
            _assert_share(agent, 1)
            assert agent._classic_export_enabled is True
        else:
            refresh_agent_mcp_tools(agent, enabled_override=[])
            _assert_share(agent, 0)
            assert agent._classic_export_enabled is False
        assert get_hermes_home() == other
        assert get_secret("CLASSIC_TEST_SCOPE") == "caller"
        assert get_terminal_scope() is before_terminal
        assert os.environ.get("HERMES_HOME") == environment_home
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(token)
