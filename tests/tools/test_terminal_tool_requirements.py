"""Tests for terminal/file tool availability in local dev environments."""

import importlib

from model_tools import get_tool_definitions

terminal_tool_module = importlib.import_module("tools.terminal_tool")


class TestTerminalRequirements:
    def test_local_backend_does_not_require_minisweagent_package(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        assert terminal_tool_module.check_terminal_requirements() is True

    def test_terminal_and_file_tools_resolve_for_local_backend(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}
        assert "terminal" in names
        assert {"read_file", "write_file", "patch", "search_files"}.issubset(names)
