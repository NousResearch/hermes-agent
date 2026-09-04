"""Profile shell hooks must be wired on the TUI/Desktop agent path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


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
        patch("tui_gateway.server._resolve_startup_runtime", return_value=("test-model", "openrouter")),
        patch("tui_gateway.server._resolve_runtime_with_fallback", return_value=runtime),
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
