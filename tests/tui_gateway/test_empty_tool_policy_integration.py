"""Persisted deny-all policy composes across the real GUI gateway boundaries.

Only the model-runtime/agent-constructor boundary is replaced: the constructor
records gateway inputs and builds its tool snapshot with the real resolver.
No policy, config loader, inspector, or MCP refresh function is mocked.
"""

import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_cli.banner as banner
import hermes_cli.config as config
import run_agent
import tui_gateway.server as server
from model_tools import get_tool_definitions
from tools.mcp_tool_agent import refresh_agent_mcp_tools
from tools.registry import registry


class _OfflineAgent(SimpleNamespace):
    """Constructor sink, not a simulated model or a hardcoded empty snapshot."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disabled_toolsets = kwargs.get("disabled_toolsets")
        self.tools = get_tool_definitions(
            enabled_toolsets=self.enabled_toolsets,
            disabled_toolsets=self.disabled_toolsets,
            quiet_mode=True,
        )
        self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}


@pytest.fixture(params=["tui", "desktop"])
def empty_session(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("HERMES_TUI_TOOLSETS", "HERMES_DESKTOP", "HERMES_DESKTOP_TERMINAL",
                 "HERMES_TUI_SKILLS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    # A genuine code workspace with opt-in focus competes with the empty policy.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'empty-policy'\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "platform_toolsets:\n  cli: []\nagent:\n  coding_context: focus\n",
        encoding="utf-8",
    )
    connect = Mock(side_effect=AssertionError("integration test attempted network access"))
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
    # Cosmetic update checks and provider auth are outside the tool-policy path.
    monkeypatch.setattr(banner, "get_update_result", lambda **kwargs: None)
    monkeypatch.setattr(server, "_probe_credentials", lambda agent: "")
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server, "_resolve_agent_model_runtime",
        lambda *args: ("offline-policy-test", {"provider": "openai"}),
    )
    constructor = Mock(side_effect=_OfflineAgent)
    monkeypatch.setattr(run_agent, "AIAgent", constructor)

    cfg = config.load_config()
    assert cfg["platform_toolsets"]["cli"] == []
    assert cfg["agent"]["coding_context"] == "focus"
    sid = f"empty-policy-{request.param}"
    agent = server._make_agent(sid, sid, platform_override=request.param)
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["enabled_toolsets"] == []
    assert constructor.call_args.kwargs["platform"] == request.param
    session = {"agent": agent, "session_key": sid, "source": request.param, "cwd": str(tmp_path)}
    monkeypatch.setitem(server._sessions, sid, session)
    yield sid, agent, monkeypatch
    connect.assert_not_called()


def _assert_empty_inspection(sid, agent):
    assert agent.enabled_toolsets == []
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    shown = server._methods["tools.show"]("show-empty", {"session_id": sid})
    assert shown["result"] == {"sections": [], "total": len(agent.tools)}
    for method in ("tools.list", "toolsets.list"):
        listed = server._methods[method]("list-empty", {"session_id": sid})
        rows = listed["result"]["toolsets"]
        assert rows  # A real catalog, not a vacuously empty inspector response.
        assert not any(row["enabled"] for row in rows)
    # session.info is an event payload, not a registered request method.
    assert server._session_info(agent, server._sessions[sid])["tools"] == {}


def test_persisted_empty_policy_reaches_gateway_and_detached_inspectors(empty_session):
    sid, agent, monkeypatch = empty_session
    _assert_empty_inspection(sid, agent)
    for kind, build_kwargs in (
        ("background", server._background_agent_kwargs),
        ("preview", server._ephemeral_preview_agent_kwargs),
    ):
        child_sid = f"{sid}-{kind}"
        kwargs = build_kwargs(agent, child_sid)
        assert kwargs["enabled_toolsets"] == []
        child = _OfflineAgent(**kwargs)
        monkeypatch.setitem(server._sessions, child_sid, {
            "agent": child, "session_key": child_sid, "source": kwargs["platform"],
        })
        _assert_empty_inspection(child_sid, child)


def test_late_mcp_registry_refresh_cannot_widen_persisted_empty_policy(empty_session):
    sid, agent, _ = empty_session
    name = "mcp_empty_policy_late_probe"
    toolset = "mcp-empty-policy"
    _assert_empty_inspection(sid, agent)
    # Simulate completed discovery at its registry boundary, never launch an MCP
    # process. Refresh and its policy/tool-definition resolvers remain real.
    registry.register(
        name=name, toolset=toolset,
        schema={"name": name, "description": "Offline late-discovery probe",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: "unused",
    )
    try:
        assert registry.get_entry(name) is not None
        for enabled in (None, server._load_enabled_toolsets(agent.platform)):
            assert refresh_agent_mcp_tools(agent, enabled_override=enabled) == set()
            _assert_empty_inspection(sid, agent)
    finally:
        registry.deregister(name)
