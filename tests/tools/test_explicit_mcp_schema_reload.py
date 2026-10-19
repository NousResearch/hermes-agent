"""Explicit reloads refresh a changed MCP schema even when tool names are unchanged."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest


@pytest.mark.parametrize("surface", ["gateway", "tui", "cli"])
def test_explicit_reload_updates_same_name_schema_and_retains_history(surface, monkeypatch):
    import model_tools
    from tools import mcp_tool
    from tools import mcp_tool_discovery, mcp_tool_lifecycle

    name = "mcp_calendar_create_event"
    old = {"type": "function", "function": {"name": name, "description": "Create a date",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}}}}
    new = deepcopy(old)
    new["function"]["description"] = "Add a date to an existing item with event notes"
    new["function"]["parameters"]["properties"].update({
        "itemId": {"type": "string"}, "body": {"type": "string"}})
    history = [{"role": "user", "content": "Keep this conversation"},
               {"role": "assistant", "content": "Saved the item"}]
    original_history = deepcopy(history)
    db = Mock()
    agent = SimpleNamespace(tools=[old], valid_tool_names={name}, enabled_toolsets=None,
        disabled_toolsets=None, quiet_mode=True, session_id="retained-session", _session_db=db)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [deepcopy(new)])
    monkeypatch.setattr(mcp_tool_lifecycle, "shutdown_mcp_servers", lambda **kw: None)
    monkeypatch.setattr(mcp_tool_discovery, "discover_mcp_tools", lambda: [name])
    monkeypatch.setattr(mcp_tool, "_servers", {})

    if surface == "gateway":
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache_lock = threading.RLock()
        runner._agent_cache = {"conversation": (agent, 0)}
        runner._mcp_reload_refresh_cached_agents(False, None)
    elif surface == "tui":
        import tui_gateway.server as srv
        session = {"agent": agent, "history": history, "history_lock": threading.RLock(),
                   "running": False, "profile_home": None}
        monkeypatch.setattr(srv, "_sessions", {"conversation": session})
        monkeypatch.setattr(srv, "_compute_mcp_rev", lambda: "stable-config")
        monkeypatch.setattr(srv, "_emit", lambda *a, **kw: True)
        monkeypatch.setattr(srv, "_session_info", lambda *a, **kw: {})
        monkeypatch.setattr(srv, "_load_enabled_toolsets", lambda *a: None)
        monkeypatch.setattr(srv, "_load_disabled_toolsets", lambda: None)
        monkeypatch.setattr(srv, "_mcp_reload_gen", 0)
        monkeypatch.setattr(srv, "_mcp_reload_loaded_rev", "")
        response = srv._methods["reload.mcp"](1, {"session_id": "conversation", "confirm": True})
        assert response["result"]["status"] == "reloaded"
    else:
        from hermes_cli.cli_info_mixin import CLIInfoMixin
        cli = CLIInfoMixin.__new__(CLIInfoMixin)
        cli.agent, cli.enabled_toolsets = agent, None
        cli._command_running = False
        cli.conversation_history = history
        cli._reload_mcp()

    assert agent.tools == [new]
    assert agent.valid_tool_names == {name}
    assert history[:len(original_history)] == original_history
    db.update_session_tool_names.assert_called_once()
    saved_id, pin = db.update_session_tool_names.call_args.args
    assert saved_id == agent.session_id
    assert pin["tools"] == [new]
