"""_make_agent resolves ephemeral prompt from display.personality."""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch


def _runtime():
    return {
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
        "credential_pool": None,
    }


def _call_make_agent(cfg):
    with ExitStack() as stack:
        stack.enter_context(patch("tui_gateway.server._load_cfg", return_value=cfg))
        stack.enter_context(patch("tui_gateway.server._get_db", return_value=MagicMock()))
        stack.enter_context(
            patch("tui_gateway.server._load_tool_progress_mode", return_value="compact")
        )
        stack.enter_context(
            patch("tui_gateway.server._load_reasoning_config", return_value=None)
        )
        stack.enter_context(patch("tui_gateway.server._load_service_tier", return_value=None))
        stack.enter_context(
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None)
        )
        stack.enter_context(
            patch(
                "tui_gateway.server._resolve_startup_runtime",
                return_value=("test-model", None),
            )
        )
        stack.enter_context(
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=_runtime(),
            )
        )
        mock_agent = stack.enter_context(patch("run_agent.AIAgent"))
        from tui_gateway.server import _make_agent

        _make_agent("sid-1", "key-1")
        return mock_agent.call_args.kwargs


def test_make_agent_uses_display_personality_when_set():
    cfg = {
        "display": {"personality": "helpful"},
        "agent": {
            "system_prompt": "manual forever",
            "personalities": {"helpful": "You are helpful."},
        },
    }
    kwargs = _call_make_agent(cfg)
    assert kwargs["ephemeral_system_prompt"] == "You are helpful."


def test_make_agent_preserves_manual_prompt_without_personality():
    cfg = {
        "display": {"personality": "none"},
        "agent": {
            "system_prompt": "manual forever",
            "personalities": {"helpful": "You are helpful."},
        },
    }
    kwargs = _call_make_agent(cfg)
    assert kwargs["ephemeral_system_prompt"] == "manual forever"
