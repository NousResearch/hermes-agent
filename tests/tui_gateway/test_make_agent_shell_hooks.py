"""Regression test: _make_agent must register shell hooks from config.yaml.

Without register_from_config(), shell hooks defined in config.yaml
(pre_tool_call, subagent_stop, on_session_end) are silently ignored
even when hooks_auto_accept is enabled — the Desktop/TUI paths only
called discover_plugins() which handles Python plugin hooks, not
config-driven shell hooks.
"""

from unittest.mock import MagicMock, patch


def test_make_agent_calls_register_from_config_without_accept_hooks():
    """_make_agent must call register_from_config(cfg) — without
    an explicit accept_hooks argument — so the internal consent
    resolution in shell_hooks.py:849-854 normalises string values
    like "false" correctly."""
    fake_cfg = {
        "model": {"default": "test-model"},
        "agent": {"system_prompt": "test"},
    }

    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "test",
                "base_url": "https://test.example.com",
                "api_key": "sk-test",
                "api_mode": "chat_completions",
                "command": None,
                "args": None,
                "credential_pool": None,
            },
        ),
        patch("run_agent.AIAgent") as mock_agent,
        patch(
            "agent.shell_hooks.register_from_config"
        ) as mock_reg,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-reg-test", "key-reg-test")

        # Must call register_from_config exactly once
        mock_reg.assert_called_once()

        # First positional arg: cfg dict
        assert mock_reg.call_args[0][0] is fake_cfg

        # Must NOT pass accept_hooks explicitly — let the function
        # resolve consent internally (line 849-854 normalises strings)
        assert "accept_hooks" not in mock_reg.call_args.kwargs


def test_make_agent_survives_register_from_config_error():
    """register_from_config is wrapped in try/except — a broken
    config or import error must not crash the agent creation."""
    fake_cfg = {
        "model": {"default": "test-model"},
        "agent": {"system_prompt": "test"},
    }

    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "test",
                "base_url": "https://test.example.com",
                "api_key": "sk-test",
                "api_mode": "chat_completions",
                "command": None,
                "args": None,
                "credential_pool": None,
            },
        ),
        patch("run_agent.AIAgent") as mock_agent,
        patch(
            "agent.shell_hooks.register_from_config",
            side_effect=RuntimeError("simulated hook registration failure"),
        ) as mock_reg,
    ):
        from tui_gateway.server import _make_agent

        # Must NOT raise — the try/except swallows the error
        _make_agent("sid-err-test", "key-err-test")

        # register_from_config was called but raised
        mock_reg.assert_called_once()

        # Despite the hook error, AIAgent was still created
        mock_agent.assert_called_once()
