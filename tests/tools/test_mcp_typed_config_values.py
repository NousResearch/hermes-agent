"""Configured MCP ``headers`` and stdio ``args`` reach the server as the text the user wrote.

YAML types an unquoted ``Notion-Version: 2022-06-28`` as a date, ``X-Api-Version: 2`` or a
``8080`` port argument as an int and ``true`` as a bool. httpx refuses any header value that is not
str or bytes, and the SDK's ``StdioServerParameters`` takes only ``list[str]`` args, so a
server configured that way never connected.
"""

from __future__ import annotations

import asyncio
import datetime
import http.server
import socketserver
import threading

import pytest

import hermes_yaml

_CONFIG_YAML = """
url: {url}
headers:
  Notion-Version: 2022-06-28
  X-Api-Version: 2
  X-Debug: true
"""
_EXPECTED = {"notion-version": "2022-06-28", "x-api-version": "2", "x-debug": "true"}


def _config(url: str) -> dict:
    config = hermes_yaml.safe_load(_CONFIG_YAML.format(url=url))
    assert isinstance(config["headers"]["Notion-Version"], datetime.date)  # the YAML shape at issue
    return config


def test_connection_inputs_carry_header_values_as_text():
    from tools.mcp_tool_transport import _connect_inputs

    (_url, headers), _ = _connect_inputs("versioned", _config("https://mcp.example.invalid/mcp"))
    assert {key.lower(): value for key, value in headers.items()} == _EXPECTED


def test_preflight_sends_typed_header_values(monkeypatch):
    import tools.mcp_tool as _mcp
    from tools import mcp_tool_errors as _mcp_errors

    seen: dict[str, str] = {}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            seen.update({key.lower(): value for key, value in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            pass

    async def _stop_after_preflight(self, config):
        raise asyncio.CancelledError()

    monkeypatch.setattr(_mcp_errors, "_validate_remote_mcp_url", lambda n, u: None)
    monkeypatch.setattr(_mcp.MCPServerTask, "_run_http", _stop_after_preflight)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        task = _mcp.MCPServerTask("versioned")
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(task.run(_config(f"http://127.0.0.1:{httpd.server_address[1]}/mcp")))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
    assert {key: seen.get(key) for key in _EXPECTED} == _EXPECTED


def test_stdio_args_reach_the_sdk_as_text(monkeypatch):
    import sys

    import tools.mcp_tool as _mcp

    class _Spawned(Exception):
        pass

    real_params = _mcp.StdioServerParameters
    seen = []

    def _capture(**kwargs):
        seen.append(real_params(**kwargs).args)
        raise _Spawned

    monkeypatch.setattr("tools.mcp_tool_config._managed_launcher", lambda *a, **k: None)
    monkeypatch.setattr(_mcp, "StdioServerParameters", _capture)
    config = hermes_yaml.safe_load(
        f"command: {sys.executable}\nargs: [-m, server, --port, 8080, --verbose, true]\n")
    with pytest.raises(_Spawned):
        asyncio.run(_mcp.MCPServerTask("typed_args")._run_stdio(config))
    assert seen == [["-m", "server", "--port", "8080", "--verbose", "true"]]
