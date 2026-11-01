"""HERMES_MCP_OVERLAY loading: merge order, error handling, safe-mode skip."""

import json
from types import SimpleNamespace
from unittest.mock import patch


class TestLoadMCPOverlay:
    def test_overlay_merges_after_portable_and_wins_on_clash(self, tmp_path, monkeypatch):
        overlay = tmp_path / "overlay.json"
        overlay.write_text(
            json.dumps({
                "native": {"url": "https://overlay.example/mcp", "headers": {"Authorization": "Bearer secret"}},
                "overlay-only": {"url": "https://only.example/mcp"},
                "bad-scheme": {"url": "ftp://nope.example/mcp"},
                "non-string-header": {"url": "https://ok.example/mcp", "headers": {"X-Num": 1, "X-Ok": "yes"}},
            }),
            encoding="utf-8",
        )
        native = {"native": {"command": "node", "args": ["srv"]}}
        portable = {"portable": {"command": "python", "args": ["p"]}}
        manager = SimpleNamespace(get_portable_mcp_servers=lambda: portable)
        monkeypatch.setenv("HERMES_MCP_OVERLAY", str(overlay))
        with (
            patch("hermes_cli.config.load_config", return_value={"mcp_servers": native}),
            patch("hermes_cli.plugins.discover_plugins"),
            patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
        ):
            from tools.mcp_tool_config import _load_mcp_config

            result = _load_mcp_config()

        assert result["native"]["url"] == "https://overlay.example/mcp"
        assert result["native"]["headers"]["Authorization"] == "Bearer secret"
        assert "command" not in result["native"]
        assert result["overlay-only"]["url"] == "https://only.example/mcp"
        assert result["portable"]["command"] == "python"
        assert "bad-scheme" not in result
        assert result["non-string-header"]["headers"] == {"X-Ok": "yes"}

    def test_overlay_read_error_returns_empty_without_leaking_contents(self, tmp_path, monkeypatch, caplog):
        missing = tmp_path / "missing-overlay.json"
        monkeypatch.setenv("HERMES_MCP_OVERLAY", str(missing))
        with caplog.at_level("WARNING", logger="tools.mcp_tool"):
            from tools.mcp_tool_config import _load_mcp_overlay

            assert _load_mcp_overlay() == {}
        messages = [r.getMessage() for r in caplog.records]
        assert any("Failed to load MCP overlay" in m for m in messages)
        assert not any("Bearer" in m for m in messages)

    def test_overlay_ignored_when_safe_mode(self, tmp_path, monkeypatch):
        overlay = tmp_path / "overlay.json"
        overlay.write_text(json.dumps({"x": {"url": "https://x.example/mcp"}}), encoding="utf-8")
        monkeypatch.setenv("HERMES_MCP_OVERLAY", str(overlay))
        monkeypatch.setenv("HERMES_SAFE_MODE", "1")
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": {}}):
            from tools.mcp_tool_config import _load_mcp_config

            assert _load_mcp_config() == {}

    def test_overlay_accepts_utf8_bom(self, tmp_path, monkeypatch):
        overlay = tmp_path / "overlay.json"
        payload = json.dumps({"bom": {"url": "https://bom.example/mcp"}})
        overlay.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
        monkeypatch.setenv("HERMES_MCP_OVERLAY", str(overlay))
        from tools.mcp_tool_config import _load_mcp_overlay

        assert _load_mcp_overlay()["bom"]["url"] == "https://bom.example/mcp"
