"""Tests for terminal/file tool availability in local dev environments."""

import importlib
import sys
import types

from model_tools import get_tool_definitions
from tools.registry import registry

terminal_tool_module = importlib.import_module("tools.terminal_tool")


class TestTerminalRequirements:
    def test_local_backend_requirements(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        assert terminal_tool_module.check_terminal_requirements() is True

    def test_terminal_and_file_tools_resolve_for_local_backend(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
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
        monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        live_check_fn = registry._tools["terminal"].check_fn

        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "modal", "modal_mode": "managed"},
        )
        monkeypatch.setitem(
            live_check_fn.__globals__,
            "_get_env_config",
            lambda: {"env_type": "modal", "modal_mode": "managed"},
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "is_managed_tool_gateway_ready",
            lambda _vendor: True,
        )
        monkeypatch.setitem(
            live_check_fn.__globals__,
            "is_managed_tool_gateway_ready",
            lambda _vendor: True,
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "ensure_minisweagent_on_path",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        monkeypatch.setitem(
            live_check_fn.__globals__,
            "ensure_minisweagent_on_path",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        monkeypatch.setattr(
            live_check_fn.__globals__["importlib"].util,
            "find_spec",
            lambda _name: (_ for _ in ()).throw(AssertionError("should not be called")),
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

    def test_android_linux_terminal_tools_resolve_when_prefix_env_is_present(self, monkeypatch, tmp_path):
        prefix = tmp_path / "prefix"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "lib").mkdir(parents=True)
        (prefix / "home").mkdir(parents=True)
        (prefix / "tmp").mkdir(parents=True)

        bash_path = importlib.import_module("shutil").which("bash")
        assert bash_path is not None

        monkeypatch.setenv("TERMINAL_ENV", "android_linux")
        monkeypatch.setenv("HERMES_ANDROID_BOOTSTRAP", "1")
        monkeypatch.setenv("HERMES_ANDROID_LINUX_PREFIX", str(prefix))
        monkeypatch.setenv("HERMES_ANDROID_LINUX_BASH", bash_path)
        monkeypatch.setenv("HERMES_ANDROID_LINUX_BIN", str(prefix / "bin"))
        monkeypatch.setenv("HERMES_ANDROID_LINUX_LIB", str(prefix / "lib"))
        monkeypatch.setenv("HERMES_ANDROID_LINUX_HOME", str(prefix / "home"))
        monkeypatch.setenv("HERMES_ANDROID_LINUX_TMP", str(prefix / "tmp"))

        live_check_fn = registry._tools["terminal"].check_fn
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "android_linux",
                "android_linux_prefix": str(prefix),
                "android_linux_bash": bash_path,
                "android_linux_home": str(prefix / "home"),
                "android_linux_tmp": str(prefix / "tmp"),
            },
        )
        monkeypatch.setitem(
            live_check_fn.__globals__,
            "_get_env_config",
            lambda: {
                "env_type": "android_linux",
                "android_linux_prefix": str(prefix),
                "android_linux_bash": bash_path,
                "android_linux_home": str(prefix / "home"),
                "android_linux_tmp": str(prefix / "tmp"),
            },
        )

        tools = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "process" in names

    def test_terminal_tool_recovers_after_stubbed_import(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setitem(sys.modules, "tools.terminal_tool", types.ModuleType("tools.terminal_tool"))
        registry.deregister("terminal")

        tools = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
