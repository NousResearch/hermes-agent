"""Tests for MCP server log notification handling (port of anomalyco/opencode#34529).

MCP servers can emit ``notifications/message`` logging notifications
(RFC 5424 syslog levels). The MCP SDK's default ``logging_callback``
silently discards them; Hermes now passes ``_make_logging_callback()``
to ``ClientSession`` so server-side diagnostics land in agent.log,
tagged with the server name.
"""

import logging
from types import SimpleNamespace

import pytest

from tools.mcp_tool import (
    _MCP_LOG_LEVEL_MAP,
    _MCP_LOGGING_CALLBACK_SUPPORTED,
    MCPServerTask,
)


def _params(level="info", data="hello", logger_name=None):
    return SimpleNamespace(level=level, data=data, logger=logger_name)


class TestLogLevelMap:
    def test_all_mcp_levels_mapped(self):
        # MCP spec (RFC 5424) defines these eight levels.
        for lvl in ("debug", "info", "notice", "warning",
                    "error", "critical", "alert", "emergency"):
            assert lvl in _MCP_LOG_LEVEL_MAP

    def test_severity_ordering(self):
        assert _MCP_LOG_LEVEL_MAP["debug"] == logging.DEBUG
        assert _MCP_LOG_LEVEL_MAP["notice"] == logging.INFO
        assert _MCP_LOG_LEVEL_MAP["warning"] == logging.WARNING
        assert _MCP_LOG_LEVEL_MAP["emergency"] == logging.ERROR


class TestLoggingCallback:
    @pytest.mark.asyncio
    async def test_routes_to_hermes_logger_with_server_tag(self, caplog):
        server = MCPServerTask("log_srv")
        callback = server._make_logging_callback()
        with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
            await callback(_params(level="info", data="server started"))
        assert any(
            "MCP server log [log_srv]: server started" in rec.getMessage()
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_includes_sub_logger_name(self, caplog):
        server = MCPServerTask("log_srv")
        callback = server._make_logging_callback()
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            await callback(_params(level="warning", data="rate limited",
                                   logger_name="http"))
        assert any(
            "MCP server log [log_srv/http]: rate limited" in rec.getMessage()
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_error_family_maps_to_error_level(self, caplog):
        server = MCPServerTask("log_srv")
        callback = server._make_logging_callback()
        with caplog.at_level(logging.ERROR, logger="tools.mcp_tool"):
            for lvl in ("error", "critical", "alert", "emergency"):
                await callback(_params(level=lvl, data=f"boom-{lvl}"))
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 4

    @pytest.mark.asyncio
    async def test_non_string_data_is_json_serialized(self, caplog):
        server = MCPServerTask("log_srv")
        callback = server._make_logging_callback()
        with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
            await callback(_params(data={"event": "connect", "port": 8080}))
        assert any(
            '"event": "connect"' in rec.getMessage() for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_unknown_level_defaults_to_info(self, caplog):
        server = MCPServerTask("log_srv")
        callback = server._make_logging_callback()
        with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
            await callback(_params(level="bogus", data="odd level"))
        assert any(
            rec.levelno == logging.INFO and "odd level" in rec.getMessage()
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_oversized_payload_truncated(self, caplog):
        server = MCPServerTask("log_srv")
        callback = server._make_logging_callback()
        with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
            await callback(_params(data="x" * 10_000))
        msg = next(
            rec.getMessage() for rec in caplog.records
            if "MCP server log" in rec.getMessage()
        )
        assert "... [truncated]" in msg
        assert len(msg) < 3000

    @pytest.mark.asyncio
    async def test_handler_never_raises(self):
        server = MCPServerTask("log_srv")
        callback = server._make_logging_callback()
        # A params object missing every attribute must not blow up the
        # SDK's notification dispatch loop.
        await callback(object())


class TestSDKSupportGate:
    def test_current_sdk_supports_logging_callback(self):
        # The pinned MCP SDK in this repo supports logging_callback; if this
        # starts failing after an SDK downgrade the feature silently degrades
        # (by design), but we want to know.
        import inspect
        from mcp import ClientSession
        expected = "logging_callback" in inspect.signature(ClientSession).parameters
        assert _MCP_LOGGING_CALLBACK_SUPPORTED == expected
