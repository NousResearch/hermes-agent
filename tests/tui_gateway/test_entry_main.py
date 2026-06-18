"""Tests for tui_gateway.entry.main() (coverage for #36611).

Covers the main() function branches: startup write, stdin dispatch
loop, parse errors, response write failures, and MCP discovery paths.
"""

import json
import os
import sys
import threading
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.entry as entry


class TestMain:
    """Cover main() function branches."""

    def _run_main(self, stdin_data="", monkeypatch=None, **patches):
        """Helper: feed stdin_data to main() and return captured stdout."""
        for key, value in patches.items():
            monkeypatch.setattr(key, value)

        # Mock stdin
        monkeypatch.setattr(sys, "stdin", StringIO(stdin_data))

        # Capture stdout
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)

        # Prevent sys.exit from killing the test
        exit_codes = []
        monkeypatch.setattr(sys, "exit", exit_codes.append)

        # Suppress MCP discovery
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))

        # Default: write_json returns True
        monkeypatch.setattr("tui_gateway.entry.write_json", MagicMock(return_value=True))

        # Default: dispatch returns None
        monkeypatch.setattr("tui_gateway.entry.dispatch", MagicMock(return_value=None))

        # Default: resolve_skin returns empty dict
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))

        # No sidecar
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())

        return out, exit_codes

    def test_startup_write_failure_exits(self, monkeypatch):
        """Startup write_json returns False -> _log_exit + sys.exit(0)."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry.write_json", MagicMock(return_value=False))
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())
        monkeypatch.setattr(sys, "exit", MagicMock(side_effect=SystemExit(0)))

        with pytest.raises(SystemExit):
            entry.main()

        entry._log_exit.assert_called_once_with(
            "startup write failed (broken stdout pipe before first event)"
        )

    def test_empty_stdin_line_skipped(self, monkeypatch):
        """Blank lines are skipped (continue)."""
        monkeypatch.setattr(sys, "stdin", StringIO("\n\n"))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        write_json = MagicMock(return_value=True)
        monkeypatch.setattr("tui_gateway.entry.write_json", write_json)
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        entry.main()

        # write_json should have been called only once (for gateway.ready startup),
        # not for the blank lines
        assert write_json.call_count == 1

    def test_valid_jsonrpc_request_dispatched(self, monkeypatch):
        """Valid JSON-RPC line gets dispatched."""
        request = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})
        monkeypatch.setattr(sys, "stdin", StringIO(request))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        mock_dispatch = MagicMock(return_value={"jsonrpc": "2.0", "result": "ok", "id": 1})
        monkeypatch.setattr("tui_gateway.entry.dispatch", mock_dispatch)
        write_json = MagicMock(return_value=True)
        monkeypatch.setattr("tui_gateway.entry.write_json", write_json)
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        entry.main()

        mock_dispatch.assert_called_once()
        # write_json called: once for gateway.ready, once for response
        assert write_json.call_count == 2

    def test_json_parse_error_sends_error_response(self, monkeypatch):
        """Invalid JSON sends -32700 parse error."""
        monkeypatch.setattr(sys, "stdin", StringIO("not valid json\n"))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        write_json = MagicMock(return_value=True)
        monkeypatch.setattr("tui_gateway.entry.write_json", write_json)
        monkeypatch.setattr("tui_gateway.entry.dispatch", MagicMock())
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        entry.main()

        # Check the parse error was written
        parse_error_call = None
        for call_args in write_json.call_args_list:
            args, _ = call_args
            if isinstance(args[0], dict) and args[0].get("error", {}).get("code") == -32700:
                parse_error_call = args[0]
                break
        assert parse_error_call is not None
        assert parse_error_call["error"]["message"] == "parse error"

    def test_json_parse_error_response_write_failure_exits(self, monkeypatch):
        """Parse error + write_json returns False -> _log_exit + exit."""
        monkeypatch.setattr(sys, "stdin", StringIO("not valid json\n"))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        # First call (gateway.ready) returns True, second (parse error) returns False
        write_json = MagicMock(side_effect=[True, False])
        monkeypatch.setattr("tui_gateway.entry.write_json", write_json)
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())
        monkeypatch.setattr(sys, "exit", MagicMock(side_effect=SystemExit(0)))

        with pytest.raises(SystemExit):
            entry.main()

        assert entry._log_exit.call_count >= 1

    def test_response_write_failure_exits(self, monkeypatch):
        """Dispatch response + write_json fails -> _log_exit + exit."""
        request = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})
        monkeypatch.setattr(sys, "stdin", StringIO(request))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry.dispatch", MagicMock(return_value={"result": "ok"}))
        # First call (gateway.ready) True, second (response) False
        write_json = MagicMock(side_effect=[True, False])
        monkeypatch.setattr("tui_gateway.entry.write_json", write_json)
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())
        monkeypatch.setattr(sys, "exit", MagicMock(side_effect=SystemExit(0)))

        with pytest.raises(SystemExit):
            entry.main()

    def test_dispatch_async_returns_none_no_write(self, monkeypatch):
        """dispatch returns None (async handler) -> no response written."""
        request = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})
        monkeypatch.setattr(sys, "stdin", StringIO(request))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry.dispatch", MagicMock(return_value=None))
        write_json = MagicMock(return_value=True)
        monkeypatch.setattr("tui_gateway.entry.write_json", write_json)
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        entry.main()

        # write_json called exactly once (gateway.ready), not for response
        assert write_json.call_count == 1

    def test_stdin_eof_logs_exit(self, monkeypatch):
        """EOF on stdin calls _log_exit('stdin EOF')."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        write_json = MagicMock(return_value=True)
        monkeypatch.setattr("tui_gateway.entry.write_json", write_json)
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        entry.main()

        entry._log_exit.assert_called_once_with("stdin EOF (TUI closed the command pipe)")

    def test_mcp_discovery_has_servers(self, monkeypatch):
        """read_raw_config returns mcp_servers -> discovery thread started."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry.write_json", MagicMock(return_value=True))
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(
            return_value={"mcp_servers": {"test-server": {"command": "echo"}}}
        ))
        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", MagicMock())
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            entry.main()
            assert entry._mcp_discovery_thread is not None
            assert entry._mcp_discovery_thread.name == "tui-mcp-discovery"
        finally:
            entry._mcp_discovery_thread = saved

    def test_mcp_discovery_no_servers(self, monkeypatch):
        """read_rawConfig returns empty config -> no discovery thread."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry.write_json", MagicMock(return_value=True))
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            entry.main()
            assert entry._mcp_discovery_thread is None
        finally:
            entry._mcp_discovery_thread = saved

    def test_mcp_discovery_config_read_fails(self, monkeypatch):
        """read_raw_config raises -> fallback to True, thread started."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry.write_json", MagicMock(return_value=True))
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(side_effect=OSError("config corrupted")))
        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", MagicMock())
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            entry.main()
            # Fallback to True means the MCP discovery thread IS started
            assert entry._mcp_discovery_thread is not None
        finally:
            entry._mcp_discovery_thread = saved

    def test_mcp_discovery_thread_background_failure_logged(self, monkeypatch):
        """discover_mcp_tools exception inside thread is logged, not fatal."""
        monkeypatch.setattr(sys, "stdin", StringIO(""))
        out = StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr("tui_gateway.entry._install_sidecar_publisher", MagicMock())
        monkeypatch.setattr("tui_gateway.entry.resolve_skin", MagicMock(return_value={}))
        monkeypatch.setattr("tui_gateway.entry.write_json", MagicMock(return_value=True))
        monkeypatch.setattr("hermes_cli.config.read_raw_config", MagicMock(
            return_value={"mcp_servers": {"test-server": {"command": "echo"}}}
        ))
        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", MagicMock(side_effect=RuntimeError("fail")))
        monkeypatch.setattr("tui_gateway.entry._log_exit", MagicMock())

        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            entry.main()
            # Thread was started and finished; ensure no crash
            thread = entry._mcp_discovery_thread
            if thread:
                thread.join(timeout=2)
        finally:
            entry._mcp_discovery_thread = saved
