"""A secret rendered ONLY into an MCP server's ``url`` must still enter its redaction snapshot.

Security-audit follow-up for #74809 (runtime-confirmed leads): with no header/env literal to fall
back on, (1) a lazy schema-cache registration stored ``dict(config)`` and dropped the resolving
env_file snapshot, and (2) a value resolved through the profile/process secret fallback (not the
env_file overlay) was never snapshotted at all, initially or on re-render. The preflight peer then
reflected the URL secret into WARNING records and the cached status error.
"""

import asyncio
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import mcp_tool, mcp_tool_config, mcp_tool_discovery, mcp_tool_lifecycle
from tools.mcp_tool_scope import _server_key

SECRET = "OpaqueUrlOnlyFixture739Tail"
OTHER_PROFILE = "OtherProfileFixture739Tail"
UNUSED = "UnusedOverlayFixture739Tail"


@pytest.fixture
def reflecting_peer():
    """Answers HEAD/POST with ``Content-Type: application/<last URL path segment>``: a non-MCP
    endpoint whose rejection reflects the URL-only secret into Hermes' diagnostics."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self):
            received.append(self.path)
            segment = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", f"application/{segment}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_HEAD = do_POST = respond

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/mcp", received
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def isolated_lazy_ledgers(monkeypatch):
    mcp_tool_lifecycle.shutdown_mcp_servers()
    for name in ("_lazy_server_configs", "_lazy_server_fingerprints", "_lazy_server_tool_names"):
        monkeypatch.setattr(mcp_tool, name, {})
    yield
    from tools.mcp_tool_registration import _forget_mcp_tool_server
    from tools.registry import registry
    for key, names in mcp_tool._lazy_server_tool_names.items():
        for tool_name in names:
            registry.deregister(tool_name, scope=mcp_tool._server_registry_scope(key))
            _forget_mcp_tool_server(tool_name)
    mcp_tool_lifecycle.shutdown_mcp_servers()


def _profile(tmp_path, name: str, value: str):
    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    (home / ".env").write_text(f"MCP_PRIVATE={value}\n")
    return home


@pytest.mark.parametrize("source", ["profile_fallback", "rerender", "lazy_env_file"])
def test_url_only_secret_is_redacted_from_diagnostics(
        source, reflecting_peer, isolated_lazy_ledgers, tmp_path, monkeypatch, caplog):
    from gateway.run import _profile_runtime_scope
    from tools.mcp_schema_cache import config_fingerprint, write_cache_entry

    base_url, received = reflecting_peer
    monkeypatch.delenv("MCP_PRIVATE", raising=False)
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    home = _profile(tmp_path, "a", SECRET)
    other = _profile(tmp_path, "b", OTHER_PROFILE)
    caplog.set_level(logging.DEBUG, logger="tools.mcp_tool")
    output = ""

    if source == "lazy_env_file":
        monkeypatch.setenv("HERMES_HOME", str(home))
        env_file = home / "server.env"
        env_file.write_text(f"MCP_TOKEN={SECRET}\nUNUSED={UNUSED}\n")
        raw = {"url": f"{base_url}/${{MCP_TOKEN}}", "env_file": str(env_file), "lazy": True}
        resolved = mcp_tool_config._resolve_mcp_server_config(raw)
        write_cache_entry("private", config_fingerprint(resolved), tools=[{
            "name": "inspect", "description": "inspect", "inputSchema": {"type": "object"}}])
        mcp_tool_discovery.register_mcp_servers({"private": resolved})
        assert not received  # registered from the schema cache, nothing connected yet
        stored = mcp_tool._lazy_server_configs[_server_key("private")]
        assert UNUSED not in json.dumps(stored)  # overlay-only values stay unserialized
        env_file.write_text("MCP_TOKEN=rotated-dummy-7391\n")  # first use after rotation
        assert not mcp_tool_discovery._ensure_lazy_server_connected("private")
        status = mcp_tool_discovery.get_mcp_status({"private": resolved})
        assert status[0]["status"] == "failed"
        output = json.dumps(status)
        snapshot = mcp_tool_config._mcp_redaction_values(stored)
    else:
        raw = {"url": f"{base_url}/${{MCP_PRIVATE}}?home=${{userHome}}"}
        with _profile_runtime_scope(other, hydrate_secrets=False):
            assert mcp_tool_config._resolve_mcp_server_config(raw)["url"].startswith(
                f"{base_url}/{OTHER_PROFILE}")
        if source == "rerender":
            # Resolved before the profile's secret was available, rendered later under its scope.
            resolved = mcp_tool_config._resolve_mcp_server_config(raw)
            assert "${MCP_PRIVATE}" in resolved["url"]
            assert "${MCP_PRIVATE}" not in mcp_tool_config._mcp_redaction_values(resolved)
        with _profile_runtime_scope(home, hydrate_secrets=False):
            if source == "rerender":
                resolved = mcp_tool_config._rerender_resolved(resolved)
            else:
                resolved = mcp_tool_config._resolve_mcp_server_config(raw)

            async def prepare():
                server = mcp_tool.MCPServerTask("private")
                try:
                    return await server._prepare_run(resolved)
                finally:
                    await server.shutdown()

            assert asyncio.run(prepare()) is False
        snapshot = mcp_tool_config._mcp_redaction_values(resolved)
        # Only substitutions actually used: no other profile's value, no context variables.
        assert OTHER_PROFILE not in snapshot and os.path.expanduser("~") not in snapshot

    assert any(SECRET in path for path in received)  # the peer really saw (and reflected) it
    text = caplog.text + output
    assert SECRET not in text
    assert "[REDACTED]" in text
    assert SECRET in snapshot
    if source == "lazy_env_file":
        assert UNUSED in snapshot  # the whole resolving overlay survives until first use
