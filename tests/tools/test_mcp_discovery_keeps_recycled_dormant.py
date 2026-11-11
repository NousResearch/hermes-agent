"""A discovery pass (cron runs one before every job) leaves a stdio server that was recycled for
idleness dormant: it wakes on its next tool call, not on discovery."""
import sys


def test_discovery_does_not_wake_idle_recycled_stdio_server(tmp_path, monkeypatch):
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_discovery import discover_mcp_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"mcp_servers:\n  idle-srv:\n    command: {sys.executable}\n    idle_timeout_seconds: 60\n", encoding="utf-8")
    server = MCPServerTask("idle-srv")
    server._config = {"command": sys.executable, "idle_timeout_seconds": 60}
    server._mark_stdio_recycled("idle_timeout_seconds")
    monkeypatch.setitem(mcp_tool._servers, "idle-srv", server)

    discover_mcp_tools()

    assert not server._reconnect_event.is_set()
