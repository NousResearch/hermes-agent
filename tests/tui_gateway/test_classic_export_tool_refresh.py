"""Offline classic construction -> registry snapshot -> compaction refresh."""

import json
import time
from types import SimpleNamespace

import pytest

from agent.agent_init import _load_tools
from agent.conversation_compression import _refresh_agent_tool_definitions, _rebuild_system_prompt_at_boundary
from gateway.classic_output_exports import ClassicExports
from gateway.hosted_room_artifacts import RoomArtifactError
from gateway.hosted_room_execution_policy import execution_policy_mapping
from hermes_constants import get_hermes_home
from tools.hosted_room_artifact import ensure_share_group_file_tool
from tools.mcp_tool_agent import refresh_agent_mcp_tools
from tools.registry import registry
from tui_gateway import classic_exports, server


def _tool_agent(**kwargs):
    # Run the real constructor's tool phase, without constructing provider clients.
    agent = SimpleNamespace(disabled_toolsets=None, **kwargs)
    _load_tools(agent, agent.enabled_toolsets, agent.disabled_toolsets)
    return agent


@pytest.fixture
def build(monkeypatch, tmp_path):
    import run_agent
    import hermes_cli.config

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    cfg = {"platform_toolsets": {"cli": ["file"], "api_server": ["file"]}}
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *a: ("offline", {}))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda _: {})
    monkeypatch.setattr(run_agent, "AIAgent", _tool_agent)

    def construct(**session_fields):
        agent = server._make_agent("runtime", "writer", platform_override="desktop")
        session = {"agent": agent, "session_key": "writer", "profile_home": str(get_hermes_home()), **session_fields}
        return agent, session

    construct.config = cfg
    return construct


def _assert_share(agent, count):
    names = [entry["function"]["name"] for entry in agent.tools]
    assert names.count("share_group_file") == count
    assert ("share_group_file" in agent.valid_tool_names) == bool(count)
    assert set(names) == agent.valid_tool_names


def _compact(agent):
    agent._cached_system_prompt = None
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _: "offline prompt"
    assert _rebuild_system_prompt_at_boundary(agent, "") == "offline prompt"


@pytest.mark.parametrize("refresh", [refresh_agent_mcp_tools, _compact])
def test_classic_retains_share_after_real_refresh(build, refresh):
    agent, session = build(room_plumbing=True)
    assert "bot_room" not in agent.enabled_toolsets
    _assert_share(agent, 0)
    classic_exports.install_schema(session)
    _assert_share(agent, 1)
    assert "bot_room" not in agent.enabled_toolsets
    refresh(agent)
    _assert_share(agent, 1)
    classic_exports.install_schema(session)
    refresh(agent)
    _assert_share(agent, 1)


def test_hosted_policy_retains_share():
    policy = execution_policy_mapping(target_profile="writer", config={
        "platform_toolsets": {"api_server": ["file"]}})
    agent = _tool_agent(enabled_toolsets=policy["enabled_toolsets"], quiet_mode=True, platform="bot_room")
    assert ensure_share_group_file_tool(agent, force=True)
    _assert_share(agent, 1)
    _compact(agent)
    _assert_share(agent, 1)
    refresh_agent_mcp_tools(agent)
    _assert_share(agent, 1)


def test_ordinary_and_unmanaged_are_not_enabled(build):
    ordinary, session = build()
    classic_exports.install_schema(session)
    _refresh_agent_tool_definitions(ordinary)
    _assert_share(ordinary, 0)
    unmanaged, _ = build(room_plumbing=True)
    _refresh_agent_tool_definitions(unmanaged)
    _assert_share(unmanaged, 0)


def test_disabled_toolset_is_not_force_injected(build):
    agent, session = build(room_plumbing=True)
    agent.disabled_toolsets = ["bot_room"]
    classic_exports.install_schema(session)
    _assert_share(agent, 0)
    _refresh_agent_tool_definitions(agent)
    _assert_share(agent, 0)


@pytest.mark.parametrize("preserve_prefix", [False, True])
@pytest.mark.parametrize("override", [{"disabled_override": ["bot_room"]}, {"enabled_override": ["file"]}])
def test_explicit_removal_stays_removed(build, override, preserve_prefix, monkeypatch):
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    refresh_agent_mcp_tools(agent, **override, preserve_prefix=preserve_prefix)
    _assert_share(agent, 0)
    classic_exports.install_schema(session)
    _refresh_agent_tool_definitions(agent)
    _assert_share(agent, 0)
    monkeypatch.setattr(classic_exports, "owned", lambda _: session)
    with pytest.raises(RoomArtifactError, match="Reopen"):
        classic_exports.preflight("runtime", session, {}, "share")


@pytest.mark.parametrize("disabled", [["bot_room"], "['bot_room']", "bot_room"])
def test_config_disable_is_respected_at_classic_install(build, disabled):
    build.config["agent"] = {"disabled_toolsets": disabled}
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    _compact(agent)
    _assert_share(agent, 0)


@pytest.mark.parametrize("added", [[], ["todo"]])
def test_unchanged_or_additive_explicit_reload_retains_classic_grant(build, added):
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    refresh_agent_mcp_tools(agent, enabled_override=[*agent.enabled_toolsets, *added])
    _assert_share(agent, 1)
    _compact(agent)
    _assert_share(agent, 1)


def test_config_disable_after_install_is_respected_on_reload(build):
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    build.config["agent"] = {"disabled_toolsets": ["bot_room"]}
    refresh_agent_mcp_tools(agent, enabled_override=list(agent.enabled_toolsets), preserve_prefix=True)
    _assert_share(agent, 0)
    _compact(agent)
    _assert_share(agent, 0)


def test_background_agent_does_not_inherit_classic_grant(build):
    classic, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    background = _tool_agent(**server._background_agent_kwargs(classic, "detached"))
    _compact(background)
    _assert_share(background, 0)
    ordinary, session = build()
    classic_exports.install_schema(session)
    _compact(ordinary)
    _assert_share(ordinary, 0)
    _compact(classic)
    _assert_share(classic, 1)


@pytest.mark.parametrize("preserve_prefix", [False, True])
def test_deregistered_tool_is_not_resurrected(build, preserve_prefix):
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    entry = registry.get_entry("share_group_file")
    registry.deregister(entry.name)
    try:
        refresh_agent_mcp_tools(agent, preserve_prefix=preserve_prefix)
        _assert_share(agent, 0)
        classic_exports.install_schema(session)
        _compact(agent)
        _assert_share(agent, 0)
    finally:
        registry.register(name=entry.name, toolset=entry.toolset, schema=entry.schema,
                          handler=entry.handler, check_fn=entry.check_fn)


def test_explicit_deferral_is_not_bypassed(build):
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    build.config["tools"] = {"tool_search": {"enabled": "on", "defer": ["share_group_file"]}}
    # The real tool snapshot cache invalidates against the profile config file.
    (get_hermes_home() / "config.yaml").write_text(json.dumps(build.config))
    _compact(agent)
    _assert_share(agent, 0)


@pytest.mark.parametrize("state", ["active", "unadmitted", "retired", "cancelled"])
def test_compaction_does_not_change_classic_admission_authority(build, state):
    agent, session = build(room_plumbing=True)
    classic_exports.install_schema(session)
    store = ClassicExports(get_hermes_home())
    path = get_hermes_home() / "report.txt"
    path.write_bytes(b"offline refresh proof\n")
    request = {"request_id": "refresh", "group_id": "group", "thread_id": "thread", "issued_at": time.time(),
               "recipients": [{"installation": "other", "profile": "reviewer"}]}
    row, _ = store.admit(session["session_key"], request, "share")
    if state == "retired":
        store.retire_group("group")
    if state == "cancelled":
        session["_turn_cancel_requested"] = True
    admission = None if state == "unadmitted" else classic_exports.Admission(store, row)
    token = classic_exports.bind(session, admission)
    try:
        _compact(agent)
        _assert_share(agent, 1)  # Stable schema is not a publication grant.
        result = json.loads(registry.dispatch("share_group_file", {"path": str(path)}))
        assert result["ok"] is (state == "active"), result
        if state == "active":
            store.settle(row["export_id"], "shared", True)
            assert store.read(row["export_id"], result["artifact_id"])[1] == path.read_bytes()
        elif state == "retired":
            with pytest.raises(RoomArtifactError, match="retired"):
                store.admit(session["session_key"], {**request, "request_id": "fresh"}, "share")
    finally:
        classic_exports.reset(token)
