"""Regression test: MCP server respawns re-read config from disk.

External tooling can rotate credentials in ``config.yaml`` between spawns
(e.g. a cron pre-run script rewriting ``BW_SESSION`` for the Bitwarden MCP
server). Previously ``MCPServerTask.run()`` captured the server config once
at first connect and reused that snapshot for every subsequent respawn in
the reconnect loop, so rotated values never reached the child process.
The fix re-reads ``_load_mcp_config()`` at the top of every (re)spawn
iteration; this test locks that behaviour in.
"""

import asyncio

import tools.mcp_tool as mcp_tool


def test_respawn_uses_refreshed_config(monkeypatch):
    """After a transport death, the respawn must use the on-disk config,
    not the snapshot captured at first connect."""
    spawn_markers = []

    async def fake_run_stdio(self, config):
        # Record the env marker this spawn was given, "die" once, then
        # shut the loop down on the second entry.
        spawn_markers.append((config.get("env") or {}).get("MARKER"))
        if len(spawn_markers) == 1:
            raise RuntimeError("simulated transport death")
        self._shutdown_event.set()
        return "shutdown"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", fake_run_stdio)
    # Keep backoff sleeps negligible.
    monkeypatch.setattr(mcp_tool, "_MAX_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr(mcp_tool, "_CONNECT_RETRY_BASE_BACKOFF_SEC", 0.01)

    # On-disk config evolves between spawns: MARKER=one, then MARKER=two.
    disk_configs = iter([
        {"fake": {"command": "cat", "env": {"MARKER": "one"}}},
        {"fake": {"command": "cat", "env": {"MARKER": "two"}}},
    ])
    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: next(
            disk_configs,
            {"fake": {"command": "cat", "env": {"MARKER": "two"}}},
        ),
    )

    async def main():
        task = mcp_tool.MCPServerTask("fake")
        task._ready.set()  # simulate a previously-healthy connection
        await task.run({"command": "cat", "env": {"MARKER": "one"}})

    asyncio.run(main())

    assert spawn_markers == ["one", "two"], (
        "respawn after transport death did not pick up the refreshed "
        f"on-disk config: {spawn_markers}"
    )


def test_respawn_keeps_last_config_when_server_removed(monkeypatch):
    """If the server disappears from config.yaml, the reconnect loop must
    not crash — it keeps using the last known config."""
    spawn_count = []

    async def fake_run_stdio(self, config):
        spawn_count.append(1)
        self._shutdown_event.set()
        return "shutdown"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", fake_run_stdio)
    monkeypatch.setattr(mcp_tool, "_load_mcp_config", lambda: {})

    async def main():
        task = mcp_tool.MCPServerTask("fake")
        task._ready.set()
        await task.run({"command": "cat", "env": {"MARKER": "one"}})

    asyncio.run(main())

    assert spawn_count == [1]
