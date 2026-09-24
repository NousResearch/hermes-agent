"""#87770 — a Desktop/TUI ``plugins.manage install`` is a mid-run load path too: the install performs a REAL
forced rescan in the serving process, ``PluginManager.on_plugin_loaded`` fires from inside it for the newcomer,
and the RESULT carries the activation summary (``live_now.mcp_servers`` naming the plugin's mcp.json servers,
connected in place) plus ``gateway_reloaded`` / ``restart_required``.

The broken invariant: before, ``_plugins_install`` returned with no rescan, so a later non-forced
``discover_plugins()`` (``tools/mcp_tool_config._portable_mcp_servers`` → ``reload.mcp``) short-circuited on
``_discovered`` and never saw the new plugin's MCP servers until a full relaunch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from hermes_cli.plugins import get_plugin_manager
from tui_gateway import server


def _fake_install_core(identifier, *, force=False, ref=None):
    """Stand-in for the git clone: a portable plugin (mcp.json + a skill) under the sandbox plugins dir,
    returning ``(target, manifest, name)`` like the real core."""
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1
    from hermes_cli.plugins_cmd import _plugins_dir, _read_manifest
    target = _plugins_dir() / "late-mcp"
    (target / "skills" / "late").mkdir(parents=True)
    (target / "plugin.json").write_text(json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": "late-mcp"}))
    (target / "skills" / "late" / "SKILL.md").write_text("---\nname: late\ndescription: d\n---\nBody.\n")
    (target / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1, "mcpServers": {"worker": {"type": "stdio", "command": "python"}}}))
    return target, _read_manifest(target), "late-mcp"


def test_plugins_manage_install_rescans_fires_on_plugin_loaded_and_exposes_mcp_servers():
    (Path(os.environ["HERMES_HOME"]) / "plugins").mkdir(exist_ok=True)
    manager = get_plugin_manager()
    manager.discover_and_load()  # the serve process booted long before the install
    events: list = []
    manager.on_plugin_loaded(events.append)
    with patch("hermes_cli.plugins_cmd._install_plugin_core", _fake_install_core), \
         patch("hermes_cli.plugins_cmd._install_python_dependencies_quietly", return_value=[]), \
         patch("gateway.control_socket.reload_gateway_plugins", return_value=None), \
         patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=[]), \
         patch("tools.connectors.mcp._registered_tool_names", return_value=[]):  # no gateway, no real server
        resp = server.handle_request({"id": "1", "method": "plugins.manage",
                                      "params": {"action": "install", "repo": "owner/late-mcp", "enable": True}})
    result = resp["result"]
    assert [e["name"] for e in events[-1]] == ["late-mcp"]  # fired from inside the rescan, newcomer only
    servers = [row["name"] for row in result["activation"]["live_now"]["mcp_servers"]]
    assert servers == ["worker"]  # exactly the mcp.json name, as mcp.servers.* know it
    assert result["gateway_reloaded"] is False and result["restart_required"] is True
    # The invariant that was broken: a later NON-forced discovery (what reload.mcp runs) sees the servers.
    from tools.mcp_tool_config import _portable_mcp_servers
    merged: dict = {}
    _portable_mcp_servers(merged)
    assert set(merged) == set(servers)
