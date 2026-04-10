"""Tests for terminal/file tool availability in local dev environments."""

import importlib
import sys
import types

from model_tools import get_tool_definitions
from tools.registry import registry

terminal_tool_module = importlib.import_module("tools.terminal_tool")


class TestTerminalRequirements:
    def test_local_backend_requirements(self, monkeypatch):
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

    def test_terminal_and_execute_code_tools_resolve_for_managed_modal(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True)
        monkeypatch.setattr(terminal_tool_module, "managed_nous_tools_enabled", lambda: True)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "modal", "modal_mode": "managed"},
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "is_managed_tool_gateway_ready",
            lambda _vendor: True,
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "execute_code" in names

    def test_terminal_and_execute_code_tools_resolve_for_vercel_sandbox(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "vercel_sandbox", "container_disk": 51200},
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "execute_code" in names

    def test_terminal_and_execute_code_tools_hide_for_unsupported_vercel_runtime(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "vercel_sandbox",
                "container_disk": 51200,
                "vercel_runtime": "node20",
            },
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" not in names
        assert "execute_code" not in names

    def test_terminal_and_execute_code_tools_hide_for_vercel_without_auth(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "vercel_sandbox",
                "container_disk": 51200,
                "vercel_runtime": "node22",
            },
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" not in names
        assert "execute_code" not in names

    def test_terminal_tool_recovers_after_stubbed_import(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setitem(sys.modules, "tools.terminal_tool", types.ModuleType("tools.terminal_tool"))
        registry.deregister("terminal")

        tools = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
