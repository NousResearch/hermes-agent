"""Workspace changes reach new Codex threads without rebinding memory or prompts."""

import socket
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import codex_runtime, runtime_cwd
from agent.transports import codex_app_server_session
from hermes_state import SessionDB
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from run_agent import AIAgent
from tools import terminal_tool


@pytest.fixture
def workspace_runtime(monkeypatch, tmp_path):
    # server imports start a background GitHub update check; disable it before import.
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    old, new, other = (tmp_path / name for name in ("old-repo", "new-repo", "other-repo"))
    for path in (old, new, other):
        path.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setenv("TERMINAL_CWD", str(other))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda agent, session: {"cwd": session["cwd"]})
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *a, **k: None)
    monkeypatch.setattr("tools.terminal_tool_lifecycle.cleanup_vm", lambda *a: None)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"memory": {"provider": "honcho"}})
    # Real Honcho initialization/routing, but tools-only lazy mode never creates peers.
    config = HonchoClientConfig(
        enabled=True, api_key="test-key", session_strategy="per-directory",
        recall_mode="tools", init_on_session_start=False,
    )
    monkeypatch.setattr(HonchoClientConfig, "from_global_config", lambda *a, **k: config)
    providers, agents, starts, network_attempts = [], [], [], []

    def load_provider(*a, **k):
        provider = HonchoMemoryProvider()
        providers.append(provider)
        return provider

    def blocked_connect(self, address):
        network_attempts.append(address)
        raise AssertionError(f"unexpected network access: {address}")

    class CodexClient:
        def __init__(self, **kwargs):
            pass

        def initialize(self, **kwargs):
            pass

        def request(self, method, params, **kwargs):
            assert method == "thread/start"
            starts.append(params["cwd"])
            return {"thread": {"id": "test-thread"}}

        def close(self):
            pass

    monkeypatch.setattr("plugins.memory.load_memory_provider", load_provider)
    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(codex_app_server_session, "CodexAppServerClient", CodexClient)
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)

    def build(cwd, key="conversation"):
        agent = AIAgent(
            model="test-model", api_key="test-key", base_url="https://example.invalid/v1",
            quiet_mode=True, skip_context_files=True, skip_background_review=True,
            save_trajectories=False, platform="gui", session_id=key, session_db=db,
            cwd=cwd,
        )
        agents.append(agent)
        agent._cached_system_prompt = f"Stable prompt for {key}"
        return agent

    def move(action, session, cwd):
        if action == "project-tool":
            server._apply_project_workspace(session["session_key"], str(cwd))
        else:
            response = server._methods[action]("move", {
                "session_id": "ui-session", "session_key": session["session_key"], "cwd": str(cwd),
            })
            assert "error" not in response, response
        assert session["cwd"] == str(cwd)

    try:
        yield SimpleNamespace(old=old, new=new, other=other, build=build, move=move,
                              providers=providers, starts=starts, server=server)
    finally:
        for agent in agents:
            codex_runtime._close_codex_session(agent)
            agent.close()
        db.close()
        assert not network_attempts


@pytest.mark.parametrize("action", ["project-tool", "session.cwd.set", "session.workspace.move"])
@pytest.mark.parametrize("restart", [False, True], ids=["first-thread", "restarted-thread"])
def test_explicit_workspace_moves_reach_new_codex_threads(workspace_runtime, action, restart):
    runtime = workspace_runtime
    server = runtime.server
    agent = runtime.build(str(runtime.old))
    neighbor = runtime.build(str(runtime.other), key="neighbor")
    session = {"agent": agent, "session_key": agent.session_id, "cwd": str(runtime.old), "source": "desktop"}
    server._sessions["ui-session"] = session
    provider = runtime.providers[0]
    prompt, memory_key = agent._cached_system_prompt, provider._session_key
    assert memory_key == runtime.old.name
    existing = None
    if restart:
        codex_runtime._ensure_codex_session(agent)
        agent._codex_session.ensure_started()
        assert runtime.starts[-1] == str(runtime.old)
        existing = agent._codex_session
    runtime.move(action, session, runtime.new)
    if restart:
        # Moving the workspace must not replace an already-running Codex thread.
        codex_runtime._ensure_codex_session(agent)
        assert agent._codex_session is existing
        codex_runtime._close_codex_session(agent)
    tokens = server._set_session_context(agent.session_id)
    try:
        assert runtime_cwd.resolve_agent_cwd() == runtime.new
        codex_runtime._ensure_codex_session(agent)
        agent._codex_session.ensure_started()
        assert runtime.starts[-1] == str(runtime.new)
    finally:
        server._clear_session_context(tokens)
    assert terminal_tool._task_env_overrides[agent.session_id]["cwd"] == str(runtime.new)
    assert neighbor.session_cwd == str(runtime.other)
    assert runtime_cwd.resolve_agent_cwd() == runtime.other
    assert provider._session_key == memory_key
    assert provider._lazy_init_kwargs["cwd"] == str(runtime.old)
    assert agent._cached_system_prompt == prompt


def test_workspace_move_during_deferred_build_reaches_first_codex_thread(workspace_runtime, monkeypatch):
    runtime = workspace_runtime
    server = runtime.server
    constructed, release, ready = threading.Event(), threading.Event(), threading.Event()
    session = {"agent": None, "agent_ready": ready, "session_key": "deferred",
               "cwd": str(runtime.old), "source": "desktop"}
    server._sessions["ui-session"] = session

    def make_agent(sid, key, **kwargs):
        agent = runtime.build(kwargs["cwd_override"], key)
        constructed.set()
        assert release.wait(timeout=15), "test did not release deferred build"
        return agent

    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a: False)
    monkeypatch.setattr(server, "_announce_built_agent", lambda *a: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    server._start_agent_build("ui-session", session)
    try:
        assert constructed.wait(timeout=15), "agent was not constructed"
        runtime.move("session.workspace.move", session, runtime.new)
    finally:
        release.set()
        assert ready.wait(timeout=15), "deferred build did not finish"
    assert not session.get("agent_error")
    agent = session["agent"]
    codex_runtime._ensure_codex_session(agent)
    agent._codex_session.ensure_started()
    assert runtime.starts[-1] == str(runtime.new)
    assert runtime.providers[0]._session_key == runtime.old.name
    assert runtime.providers[0]._lazy_init_kwargs["cwd"] == str(runtime.old)
    assert agent._cached_system_prompt == f"Stable prompt for {agent.session_id}"
