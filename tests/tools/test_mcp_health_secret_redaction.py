"""Server-local secrets must not escape through health diagnostics."""
import asyncio
import logging
from unittest.mock import AsyncMock

from tools.mcp_tool import MCPServerTask


def test_mark_suspect_redacts_stored_reason_and_logs(tmp_path, caplog):
    from tools.mcp_tool_config import _load_mcp_server_env

    secret = "opaque-server-health-value-7391"
    env_file = tmp_path / "server.env"
    env_file.write_text(f"MCP_PRIVATE_TOKEN={secret}\n")
    server = MCPServerTask("private")
    server._redaction_values = tuple(_load_mcp_server_env({"env_file": str(env_file)}).values())
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        server.mark_suspect(f"keepalive failed: request at /{secret}")
    assert server._suspect_reason is not None
    assert secret not in server._suspect_reason
    assert secret not in caplog.text
    assert "keepalive failed" in server._suspect_reason
    assert "[REDACTED]" in server._suspect_reason


def test_suspect_health_probe_redacts_exception_and_reconnects(caplog):
    secret = "opaque-server-health-value-7391"
    server = MCPServerTask("private")
    server._redaction_values = (secret,)
    server.session = object()
    server._keepalive_probe = AsyncMock(side_effect=RuntimeError(f"request at /{secret}"))
    server.mark_suspect("stale connection")
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        assert asyncio.run(server.ensure_healthy()) is False
    assert server.session is None
    assert server._reconnect_event.is_set()
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text
