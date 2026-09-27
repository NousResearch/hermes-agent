"""A remote MCP server definition edited in config.yaml while the process runs must take effect on
the next transport rebuild (#113907): ``run()`` used to keep its start-time dict forever, so a URL
change (catalog migration, dashboard re-auth to a new endpoint) left the loop probing the old URL
and its stale-URL OAuth provider evicted the fresh one built for the new URL."""

import asyncio

import pytest


@pytest.mark.no_isolate
def test_reconnect_adopts_changed_url_from_config_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool_config
    from tools.mcp_tool import MCPServerTask

    old = {"url": "https://mcp.example.test/sse", "auth": "oauth", "oauth": {"client_id": "a"}}
    new = {"url": "https://mcp.example.test/mcp", "auth": "oauth", "oauth": {"client_id": "a"}}
    disk = {"srv": dict(old)}
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {k: dict(v) for k, v in disk.items()})
    seen: list = []

    class _Task(MCPServerTask):
        async def _prepare_run(self, config):
            self._config = config
            self._auth_type = config["auth"]
            return True

        async def _run_http(self, config):
            seen.append((config["url"], self._config["url"], self._auth_type))
            self._ready.set()
            self.session = object()
            self._session_proven = True
            if len(seen) == 1:
                disk["srv"] = dict(new)  # the dashboard saved the migrated definition mid-run
                return "reconnect"
            self._shutdown_event.set()
            return "shutdown"

    async def _scenario():
        task = _Task("srv")
        await asyncio.wait_for(task.run(dict(old)), timeout=5)

    asyncio.run(_scenario())
    assert seen == [(old["url"], old["url"], "oauth"), (new["url"], new["url"], "oauth")]


@pytest.mark.no_isolate
def test_unchanged_or_missing_definition_keeps_the_running_config(monkeypatch, tmp_path):
    """No rebuild churn when config.yaml agrees with the running dict (or no longer lists the
    server): the start-time config stays bound byte-for-byte."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool_config
    from tools.mcp_tool import MCPServerTask

    cfg = {"url": "https://mcp.example.test/mcp", "headers": {"X": "1"}, "connect_timeout": 5}
    task = MCPServerTask("srv")
    task._config = cfg
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {"srv": dict(cfg, connect_timeout=9)})
    assert task._refresh_remote_config(cfg) is cfg  # non-endpoint keys do not trigger a rebind
    monkeypatch.setattr(mcp_tool_config, "_load_mcp_config", lambda: {})
    assert task._refresh_remote_config(cfg) is cfg
