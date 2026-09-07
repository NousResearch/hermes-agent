"""Recovery diagnostics retain their state semantics without reflecting secrets."""
import logging
from unittest.mock import Mock

import pytest

from tools import mcp_tool as core
from tools import mcp_tool_discovery as discovery
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_loop as loop
from tools.mcp_tool_config import _resolve_mcp_server_config


@pytest.mark.parametrize("recovery", ["retry", "oauth", "session", "stdio"])
def test_recovery_failure_redacts_logs_and_result(recovery, monkeypatch, caplog):
    secret = "opaque-recovery-value-7391"
    server = core.MCPServerTask("private")
    server._redaction_values = (secret,)
    monkeypatch.setattr(core, "_servers", {"private": server})
    monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
    reconnect = Mock(return_value=True)
    monkeypatch.setattr(loop, "_signal_reconnect_and_wait", reconnect)
    monkeypatch.setattr(core, "_bump_server_error", Mock())

    def fail(*args, **kwargs):
        raise RuntimeError(f"request failed at /{secret}")

    with caplog.at_level(logging.INFO, logger="tools.mcp_tool"):
        if recovery == "retry":
            result = handlers._retry_once("private", fail, "call_tool", "reconnect")
            assert result is None
        elif recovery == "oauth":
            monkeypatch.setattr(handlers, "_is_auth_error", lambda exc: True)
            monkeypatch.setattr(loop, "_run_on_mcp_loop", fail)
            result = handlers._handle_auth_error_and_retry("private", RuntimeError(secret), fail, "call_tool")
            assert result is not None
            assert '"needs_reauth": true' in result
        elif recovery == "session":
            monkeypatch.setattr(handlers, "_is_session_expired_error", lambda exc: True)
            result = handlers._handle_session_expired_and_retry("private", RuntimeError(secret), fail, "call_tool")
            assert result is None
            reconnect.assert_called_once()
        else:
            result = handlers._handle_stdio_child_exited_and_retry(
                "private", handlers._StdioChildExited(secret), fail, "call_tool"
            )
            assert result is not None
            assert "MCP call failed after respawning" in result
            reconnect.assert_called_once()
    assert secret not in (result or "")
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_lazy_connect_failure_redacts_status_and_logs(tmp_path, monkeypatch, caplog):
    secret = "opaque-lazy-connect-value-7391"
    env_file = tmp_path / "server.env"
    env_file.write_text(f"PRIVATE_TOKEN={secret}\n", encoding="utf-8")
    # Real resolution chain, as production registers lazy configs: the resolving read
    # attaches the redaction snapshot that failure handling must consume.
    config = _resolve_mcp_server_config({"url": "https://example.invalid/mcp", "env_file": str(env_file)})
    monkeypatch.setattr(core, "_servers", {})
    monkeypatch.setattr(core, "_lazy_server_configs", {"private": config})
    monkeypatch.setattr(core, "_server_connecting", set())
    monkeypatch.setattr(core, "_server_connect_errors", {})
    monkeypatch.setattr(core, "_server_connect_retry_after", {})
    monkeypatch.setattr(core, "_server_connect_failures", {})
    monkeypatch.setattr(loop, "_ensure_mcp_loop", lambda: None)

    def fail(*args, **kwargs):
        raise RuntimeError(f"request failed at /{secret}")

    monkeypatch.setattr(loop, "_run_on_mcp_loop", fail)
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        assert discovery._ensure_lazy_server_connected("private") is False
    assert "private" not in core._server_connecting
    assert core._server_connect_failures["private"] == 1
    assert secret not in core._server_connect_errors["private"]
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text
