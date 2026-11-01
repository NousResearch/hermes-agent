"""A plugin settings save reconnects only the MCP servers whose resolved launch inputs changed:
a new ``${VAR}`` value the server's ``env`` passes through restarts it with that value; a save that
changes nothing it launches with leaves the live connection alone. Real stdio child, real config."""

import json
import os
import sys
import textwrap

import pytest

SERVER = textwrap.dedent('''\
    import os
    from mcp.server.mcpserver import MCPServer
    app = MCPServer("envecho")

    @app.tool()
    def key() -> str:
        return os.environ.get("DEMO_KEY", "")

    app.run()
''')


def _pid_of(core, name):
    srv = core._servers.get(name)
    session = getattr(srv, "session", None)
    return id(session) if session is not None else None


@pytest.mark.no_isolate
@pytest.mark.platforms("posix")
def test_restart_only_when_resolved_launch_inputs_change(monkeypatch, tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    home = tmp_path / "home"
    home.mkdir()
    # A portable plugin's server: it joins the config AFTER load's ${VAR} render pass, the case where
    # an unrendered recompute reads every save as "changed".
    plugin = home / "plugins" / "envecho"
    plugin.mkdir(parents=True)
    (plugin / "server.py").write_text(SERVER, encoding="utf-8")
    launcher = plugin / "run.sh"  # portable commands are bare or ./-relative: wrap this interpreter
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/server.py"\n', encoding="utf-8")
    launcher.chmod(0o755)
    (plugin / "plugin.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "envecho"}), encoding="utf-8")
    (plugin / "mcp.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json", "mcpServers": {"envecho": {
        "type": "stdio", "command": "./run.sh", "cwd": "${PLUGIN_ROOT}",
        "env": {"DEMO_KEY": "${DEMO_KEY}"}}}}), encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [envecho]\n", encoding="utf-8")
    (home / ".env").write_text("DEMO_KEY=first\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("DEMO_KEY", "first")

    from hermes_cli.config import save_env_value
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as _loop
    from tools.mcp_tool_common import _core

    try:
        disc.discover_mcp_tools()
        first = _core._servers["envecho"]._resolved_identity
        session = _pid_of(_core, "envecho")
        assert first and session

        assert disc.restart_mcp_servers_with_new_inputs({"envecho"}) == []  # nothing it launches with moved
        assert _pid_of(_core, "envecho") == session

        save_env_value("DEMO_KEY", "second")  # what the settings form's secret save does
        assert disc.restart_mcp_servers_with_new_inputs({"envecho"}) == ["envecho"]
        # Returns only once reconnected, so a later save compares against THIS connection's inputs.
        assert _pid_of(_core, "envecho") not in (None, session)
        assert _core._servers["envecho"]._resolved_identity != first
        assert os.environ["DEMO_KEY"] == "second"
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        shutdown_mcp_servers()
        _loop._stop_mcp_loop()
