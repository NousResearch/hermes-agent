"""Env-file rotation must not change the secrets redacted for an in-flight config."""

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli import mcp_config
from tools import mcp_tool, mcp_tool_config, mcp_tool_discovery, mcp_tool_lifecycle


SECRET_A = "opaque-snapshot-a-7391"
SECRET_B = "opaque-snapshot-b-8264"
UNUSED_SECRET = "unused-overlay-value-1937"


@pytest.fixture
def reflecting_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    env_file = tmp_path / "server.env"
    env_file.write_text(f"MCP_TOKEN={SECRET_A}\nUNUSED={UNUSED_SECRET}\n")
    received = []
    on_request = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self):
            value = self.headers["X-Private"]
            received.append(value)
            if on_request:
                on_request.pop()()
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", f"application/{value}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_HEAD = respond
        do_POST = respond

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    raw = {
        "url": f"http://127.0.0.1:{httpd.server_port}/mcp",
        "env_file": str(env_file),
        "headers": {"X-Private": "${MCP_TOKEN}"},
        "lazy": True,
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"snapshot": raw}}))
    mcp_tool_lifecycle.shutdown_mcp_servers()
    # shutdown reaps live tasks but intentionally retains lazy registrations.
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {})
    monkeypatch.setattr(mcp_tool, "_lazy_server_fingerprints", {})
    monkeypatch.setattr(mcp_tool, "_lazy_server_tool_names", {})
    try:
        yield SimpleNamespace(raw=raw, env_file=env_file, received=received, on_request=on_request)
    finally:
        from tools.mcp_tool_registration import _forget_mcp_tool_server
        from tools.registry import registry

        for name in mcp_tool._lazy_server_tool_names.get("snapshot", []):
            registry.deregister(name, scope=mcp_tool._server_registry_scope("snapshot"))
            _forget_mcp_tool_server(name)
        mcp_tool_lifecycle.shutdown_mcp_servers()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _mutate(env_file, mutation):
    if mutation == "rotate":
        env_file.write_text(f"MCP_TOKEN={SECRET_B}\n")
    elif mutation == "delete":
        env_file.unlink()


@pytest.mark.parametrize("mutation", ["intact", "rotate", "delete"])
@pytest.mark.parametrize("sink", ["prepare", "discovery", "lazy", "probe_log"])
def test_runtime_errors_keep_resolving_snapshot(reflecting_server, mutation, sink, caplog):
    from tools.mcp_schema_cache import config_fingerprint, write_cache_entry
    from tools.mcp_tool_loop import _ensure_mcp_loop, _run_on_mcp_loop

    fixture = reflecting_server
    configured = mcp_tool_config._load_mcp_config()
    resolved = configured["snapshot"]
    assert resolved["headers"]["X-Private"] == SECRET_A
    assert UNUSED_SECRET not in json.dumps(resolved)
    _mutate(fixture.env_file, mutation)
    # A later load of the SAME server must not replace an older attempt's secrets.
    if mutation == "rotate":
        newer = mcp_tool_config._load_mcp_config()["snapshot"]
        assert newer["headers"]["X-Private"] == SECRET_B
        assert SECRET_B not in mcp_config._sanitize_mcp_probe_error(SECRET_B, newer)

    caplog.set_level(logging.DEBUG, logger="tools.mcp_tool")
    if sink == "prepare":
        async def prepare():
            server = mcp_tool.MCPServerTask("snapshot")
            try:
                assert not await server._prepare_run(resolved)
                server.mark_suspect(f"reflected {SECRET_A}")
                return server._suspect_reason
            finally:
                await server.shutdown()

        output = asyncio.run(prepare())
    elif sink == "probe_log":
        # Supply the already-resolved config, not a mocked resolver or connector.
        from unittest.mock import patch
        with patch.object(mcp_tool_config, "_load_mcp_config", return_value=configured):
            assert mcp_tool_discovery.probe_mcp_server_tools() == {}
        output = caplog.text
    else:
        if sink == "lazy":
            write_cache_entry("snapshot", config_fingerprint(resolved), tools=[{
                "name": "example", "description": "example", "inputSchema": {"type": "object"},
            }])
            mcp_tool_discovery.register_mcp_servers(configured)
            assert not fixture.received
            assert not mcp_tool_discovery._ensure_lazy_server_connected("snapshot")
        else:
            _ensure_mcp_loop()
            _run_on_mcp_loop(lambda: mcp_tool_discovery._discover_all(configured), timeout=10)
        status = mcp_tool_discovery.get_mcp_status(configured)
        assert status[0]["status"] == "failed"
        output = json.dumps(status)

    assert fixture.received and set(fixture.received) == {SECRET_A}
    assert SECRET_A not in output
    assert "[REDACTED]" in output
    assert SECRET_A not in caplog.text
    assert UNUSED_SECRET not in caplog.text
    assert SECRET_A not in mcp_config._sanitize_mcp_probe_error(SECRET_A, resolved)
    # A copied config retains even overlay-only values, without serializing them.
    copied = resolved.copy()
    assert UNUSED_SECRET not in json.dumps(copied)
    assert UNUSED_SECRET not in mcp_config._sanitize_mcp_probe_error(UNUSED_SECRET, copied)
    # Plain external dicts deliberately use known literal env/header values, not file I/O.
    assert SECRET_A not in mcp_config._sanitize_mcp_probe_error(SECRET_A, dict(resolved))
    if mutation == "rotate":
        assert mcp_config._sanitize_mcp_probe_error(SECRET_B, resolved) == SECRET_B


@pytest.mark.parametrize("mutation", ["intact", "rotate", "delete"])
@pytest.mark.parametrize("sink", ["sanitize", "cli", "dashboard", "oauth", "catalog"])
def test_probe_surfaces_keep_resolving_snapshot(
    reflecting_server, tmp_path, monkeypatch, mutation, sink, capsys, caplog
):
    from hermes_cli import mcp_catalog, web_server_mcp
    from hermes_cli.web_routers.mcp import test_mcp_server as dashboard_test
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow
    from tools.mcp_tool_errors import NonMcpEndpointError

    fixture = reflecting_server
    # The real probe resolves A before the peer rotates/deletes its file and reflects A.
    fixture.on_request.append(lambda: _mutate(fixture.env_file, mutation))
    caplog.set_level(logging.DEBUG, logger="tools.mcp_tool")
    if sink == "sanitize":
        with pytest.raises(NonMcpEndpointError) as caught:
            mcp_config._probe_single_server("snapshot", fixture.raw)
        output = mcp_config._sanitize_mcp_probe_error(caught.value, fixture.raw)
    elif sink == "cli":
        mcp_config.cmd_mcp_test(SimpleNamespace(name="snapshot"))
        output = capsys.readouterr().out
    elif sink == "dashboard":
        result = asyncio.run(dashboard_test("snapshot"))
        assert result["ok"] is False
        assert result["tools"] == []
        output = json.dumps(result)
    elif sink == "oauth":
        # Exercise the worker's humanized-message branch without replacing the probe.
        monkeypatch.setattr(
            "tools.mcp_oauth.humanize_oauth_registration_error",
            lambda _name, exc, **_kwargs: f"Humanized: {exc}",
        )
        flow = DashboardOAuthFlow(
            flow_id="snapshot-flow", server_name="snapshot", profile=None,
            hermes_home=str(tmp_path), redirect_uri="http://127.0.0.1/callback",
        )
        web_server_mcp._run_dashboard_mcp_oauth(flow, fixture.raw)
        assert flow.worker_done
        assert flow.status == "error"
        assert "Humanized:" in flow.error
        output = json.dumps(flow.snapshot())
    else:
        assert mcp_catalog._probe_tools("snapshot") is None
        output = capsys.readouterr().out

    assert fixture.received and set(fixture.received) == {SECRET_A}
    assert SECRET_A not in output
    assert "[REDACTED]" in output
    assert SECRET_A not in caplog.text
    assert UNUSED_SECRET not in output
