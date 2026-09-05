"""Profile shell hooks must be wired on the TUI/Desktop agent path."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_make_agent_registers_shell_hooks_from_active_profile_config():
    from tui_gateway import server

    cfg = {
        "agent": {"system_prompt": ""},
        "model": {"default": "test-model"},
        "hooks": {
            "pre_tool_call": [
                {"matcher": "write_file", "command": "protect-data"},
            ],
        },
    }
    runtime = SimpleNamespace(
        runtime={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
        used_fallback=False,
    )

    with (
        patch("tui_gateway.server._load_cfg", return_value=cfg),
        patch(
            "tui_gateway.server._resolve_startup_runtime",
            return_value=("test-model", "openrouter"),
        ),
        patch(
            "tui_gateway.server._resolve_runtime_with_fallback", return_value=runtime
        ),
        patch("tui_gateway.server._load_provider_routing", return_value={}),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch("tui_gateway.server._load_fallback_model", return_value=None),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._agent_cbs", return_value={}),
        patch("agent.shell_hooks.register_from_config") as register_hooks,
        patch("run_agent.AIAgent") as agent_cls,
    ):
        server._make_agent(
            "desktop-session",
            "session-key",
            context_cwd_is_launch_artifact=False,
        )

    register_hooks.assert_called_once_with(cfg, accept_hooks=False)
    agent_cls.assert_called_once()


def test_make_agent_real_shell_hooks_are_isolated_by_profile(tmp_path, monkeypatch):
    from agent import shell_hooks
    from hermes_cli import plugins
    from tui_gateway import server

    root = tmp_path / "hermes"
    first_home = root / "profiles" / "first"
    second_home = root / "profiles" / "second"
    first_home.mkdir(parents=True)
    second_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    first_missing_command = str(tmp_path / "missing-first-profile-hook")
    second_missing_command = str(tmp_path / "missing-second-profile-hook")
    ambient_missing_command = str(tmp_path / "missing-ambient-root-hook")

    def write_fail_closed_config(home, command):
        config = {
            "hooks_auto_accept": True,
            "hooks": {
                "pre_tool_call": [
                    {
                        "matcher": "write_file",
                        "command": command,
                        "fail_closed": True,
                    }
                ]
            },
        }
        (home / "config.yaml").write_text(
            yaml.safe_dump(config),
            encoding="utf-8",
        )

    write_fail_closed_config(root, ambient_missing_command)
    write_fail_closed_config(first_home, first_missing_command)
    write_fail_closed_config(second_home, second_missing_command)

    runtime = SimpleNamespace(
        runtime={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
        used_fallback=False,
    )

    def make_agent_under(home, sid):
        token = set_hermes_home_override(home)
        try:
            server._make_agent(
                sid,
                f"{sid}-key",
                context_cwd_is_launch_artifact=False,
            )
        finally:
            reset_hermes_home_override(token)

    def block_message_under(home):
        token = set_hermes_home_override(home)
        try:
            return plugins.get_pre_tool_call_block_message(
                "write_file", {"path": "safe.txt", "content": "safe"}
            )
        finally:
            reset_hermes_home_override(token)

    @contextmanager
    def isolated_hook_registries():
        try:
            plugins._reset_plugin_managers_for_tests()
            shell_hooks.reset_for_tests()
            yield
        finally:
            try:
                plugins._reset_plugin_managers_for_tests()
            finally:
                shell_hooks.reset_for_tests()

    with isolated_hook_registries():
        with (
            patch(
                "tui_gateway.server._resolve_startup_runtime",
                return_value=("test-model", "openrouter"),
            ),
            patch(
                "tui_gateway.server._resolve_runtime_with_fallback",
                return_value=runtime,
            ),
            patch("tui_gateway.server._load_provider_routing", return_value={}),
            patch("tui_gateway.server._load_reasoning_config", return_value=None),
            patch("tui_gateway.server._load_service_tier", return_value=None),
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
            patch("tui_gateway.server._load_fallback_model", return_value=None),
            patch("tui_gateway.server._get_db", return_value=MagicMock()),
            patch("tui_gateway.server._agent_cbs", return_value={}),
            patch("run_agent.AIAgent"),
        ):
            make_agent_under(first_home, "first-session")
            assert block_message_under(first_home) == (
                f"hook {first_missing_command} failed closed: command not found"
            )

            make_agent_under(second_home, "second-session")
            assert block_message_under(second_home) == (
                f"hook {second_missing_command} failed closed: command not found"
            )

            make_agent_under(first_home, "first-session-again")
            assert block_message_under(first_home) == (
                f"hook {first_missing_command} failed closed: command not found"
            )
