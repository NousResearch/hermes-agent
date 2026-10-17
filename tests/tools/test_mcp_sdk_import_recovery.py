"""A failed MCP SDK import is retried, never cached for the process lifetime.

Regression for #134933: ``_ensure_mcp_sdk()`` decided transport availability
once per process. A single failed ``mcp.client.streamable_http`` import (a
dependency mid-swap during an update, a racing import) left HTTP "unavailable"
forever, so every HTTP server in that process parked with "requires HTTP
transport ... Upgrade the mcp package" — an error that hid the real import
failure — and no parked self-probe or ``/reload-mcp`` could recover it, while a
fresh process with the same venv connected fine. The same latch held for the
SSE client and for the core SDK import.
"""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import mcp_tool
from tools.mcp_tool import MCPServerTask, sdk_httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _DummyAsyncClient:
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _DummySession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def initialize(self):
        return None


def _streams(*args, **kwargs):
    class _Ctx:
        async def __aenter__(self):
            return MagicMock(), MagicMock()

        async def __aexit__(self, *a):
            return False

    return _Ctx()


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _connects(server: MCPServerTask, config: dict, sdk_module: str, client_name: str) -> bool:
    """Run ``_run_http`` with ``sdk_module`` importable again (serving a fake client)."""
    connected = []

    async def _discover_tools(self):
        connected.append(True)
        self._shutdown_event.set()

    with patch.dict(sys.modules, {sdk_module: _module(sdk_module, **{client_name: _streams})}), \
         patch.object(sdk_httpx(), "AsyncClient", _DummyAsyncClient), \
         patch.object(mcp_tool, "ClientSession", _DummySession), \
         patch.object(MCPServerTask, "_discover_tools", _discover_tools):
        asyncio.run(server._run_http(config))
    return bool(connected)


def test_http_transport_recovers_once_its_import_does(monkeypatch):
    # State a first-use probe leaves behind when the Streamable HTTP import failed;
    # the re-probe rebinds the client, so restore the real one afterwards.
    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", False)
    monkeypatch.setattr(mcp_tool, "_MCP_NEW_HTTP", False)
    monkeypatch.setattr(mcp_tool, "_MCP_LEGACY_HTTP", False)
    monkeypatch.setattr(mcp_tool, "streamable_http_client", mcp_tool.streamable_http_client)
    server = MCPServerTask("remote")
    config = {"url": "https://example.com/mcp"}

    with patch.dict(sys.modules, {"mcp.client.streamable_http": None}):
        with pytest.raises(ImportError) as still_failing:
            importlib.import_module("mcp.client.streamable_http")
        with pytest.raises(ImportError, match="requires HTTP transport") as parked:
            asyncio.run(server._run_http(config))
    # The connect error names the failed import instead of only blaming the package version.
    assert str(still_failing.value) in str(parked.value)

    # The next connect (parked self-probe, /reload-mcp) picks the transport up.
    assert _connects(server, config, "mcp.client.streamable_http", "streamable_http_client")


def test_sse_transport_recovers_once_its_import_does(monkeypatch):
    monkeypatch.setattr(mcp_tool, "sse_client", None)
    server = MCPServerTask("remote")

    assert _connects(server, {"url": "https://example.com/sse", "transport": "sse"},
                     "mcp.client.sse", "sse_client")


def test_core_sdk_import_failure_is_retried_not_latched():
    """Fresh interpreter (the issue's long-lived process): two uses hit a failing core import
    (on mcp 2.x a failing Streamable HTTP import fails ``import mcp`` too), the third happens
    after it imports again."""
    script = textwrap.dedent("""
        import logging, sys
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        sys.modules["mcp.client.stdio"] = None
        from tools import mcp_tool
        failing = [mcp_tool._ensure_mcp_sdk(), mcp_tool._ensure_mcp_sdk()]
        del sys.modules["mcp.client.stdio"]
        recovered = mcp_tool._ensure_mcp_sdk()
        print(*failing, recovered, mcp_tool.__dict__.get("stdio_client") is not None)
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=_REPO_ROOT,
                            capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    # Failing import -> unavailable (not a half-bound "available"); working import -> loaded.
    assert result.stdout.split()[-4:] == ["False", "False", "True", "True"]
    # Reported with its cause, once per cause rather than once per retry.
    assert result.stderr.count("mcp.client.stdio failed to import (ModuleNotFoundError") == 1, result.stderr
