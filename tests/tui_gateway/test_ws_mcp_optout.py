"""WS-sidecar MCP discovery honors the dashboard server opt-out."""

import sys
import types
from unittest.mock import patch

from tui_gateway.ws import _maybe_start_ws_mcp_discovery


def _fake_mcp_startup():
    calls = []
    mod = types.SimpleNamespace(
        start_background_mcp_discovery=lambda **kw: calls.append(kw)
    )
    return mod, calls


class TestWsMcpOptOut:
    def test_no_mcp_env_skips_discovery(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_NO_MCP", "1")
        mod, calls = _fake_mcp_startup()
        with patch.dict(sys.modules, {"hermes_cli.mcp_startup": mod}):
            _maybe_start_ws_mcp_discovery()
        assert calls == []

    def test_discovery_runs_by_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_NO_MCP", raising=False)
        mod, calls = _fake_mcp_startup()
        with patch.dict(sys.modules, {"hermes_cli.mcp_startup": mod}):
            _maybe_start_ws_mcp_discovery()
        assert len(calls) == 1
        assert calls[0]["thread_name"] == "tui-ws-mcp-discovery"

    def test_non_one_value_keeps_discovery(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_NO_MCP", "0")
        mod, calls = _fake_mcp_startup()
        with patch.dict(sys.modules, {"hermes_cli.mcp_startup": mod}):
            _maybe_start_ws_mcp_discovery()
        assert len(calls) == 1
