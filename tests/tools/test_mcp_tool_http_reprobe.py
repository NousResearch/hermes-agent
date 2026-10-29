"""Regression tests for the MCP HTTP-transport re-probe (#134933).

``_ensure_mcp_sdk`` attempts the SDK import once per process; a transient
``mcp.client.streamable_http`` import failure on that first probe latched
``_MCP_HTTP_AVAILABLE=False`` forever, so every later connect attempt — parked
self-probes included — failed with the canned upgrade message although the
package imports fine. These tests pin the re-probe contract that lets each
attempt clear a latched unavailability verdict.
"""

import asyncio
from unittest.mock import patch

import pytest

from tools import mcp_tool


# ---------------------------------------------------------------------------
# HTTP config
# ---------------------------------------------------------------------------

class TestHTTPConfig:
    """Tests for HTTP transport detection and handling."""

    def test_is_http_with_url(self):
        from tools.mcp_tool import MCPServerTask
        server = MCPServerTask("remote")
        server._config = {"url": "https://example.com/mcp"}
        assert server._is_http() is True

    def test_http_unavailable_raises(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        config = {"url": "https://example.com/mcp"}

        async def _test():
            # Re-probe disabled too: with the SDK importable a live re-probe would clear
            # the patched flag and never raise (#134933).
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", False), \
                 patch("tools.mcp_tool._reprobe_mcp_http_availability", return_value=False), \
                 pytest.raises(ImportError):
                await server._run_http(config)

        asyncio.run(_test())

    def test_stdio_unavailable_raises_importerror_not_nameerror(self):
        """Regression test for #30904.

        When the mcp SDK isn't installed, ``_run_stdio`` previously leaked a
        bare ``NameError: name 'StdioServerParameters' is not defined``. The
        gate now raises a clear ``ImportError`` with install instructions,
        mirroring ``_run_http``'s behaviour when the HTTP transport is
        unavailable.
        """
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("local")
        config = {"command": "python3", "args": ["/tmp/echo.py"]}

        async def _test():
            with patch("tools.mcp_tool._MCP_AVAILABLE", False):
                with pytest.raises(ImportError):
                    await server._run_stdio(config)

        asyncio.run(_test())


class TestHTTPReprobe:
    def test_transient_http_import_failure_recovered_by_reprobe(self, monkeypatch):
        calls = {"n": 0}
        real_import = mcp_tool._import_sdk_names

        def _flaky_import(module, names, missing_msg=None):
            if module == "mcp.client.streamable_http":
                calls["n"] += 1
                return calls["n"] > 1  # transient failure, then recovered
            return real_import(module, names, missing_msg)

        monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", False)  # latched verdict
        monkeypatch.setattr(mcp_tool, "_import_sdk_names", _flaky_import)
        assert mcp_tool._reprobe_mcp_http_availability() is True
        assert mcp_tool._MCP_HTTP_AVAILABLE is True
        assert calls["n"] == 2  # both spellings retried, not one latched attempt

    def test_reprobe_noops_when_http_already_available(self, monkeypatch):
        monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", True)
        monkeypatch.setattr(
            mcp_tool, "_import_sdk_names",
            lambda *a, **k: pytest.fail("available verdicts must never be re-imported"))
        assert mcp_tool._reprobe_mcp_http_availability() is True

    def test_reprobe_honors_sdk_unavailable_without_import(self, monkeypatch):
        monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", False)
        monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", False)
        monkeypatch.setattr(
            mcp_tool, "_import_sdk_names",
            lambda *a, **k: pytest.fail("SDK-unavailable verdicts must not import"))
        assert mcp_tool._reprobe_mcp_http_availability() is False
