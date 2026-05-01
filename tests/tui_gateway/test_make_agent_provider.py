"""Regression test for #11884: _make_agent must resolve runtime provider.

Without resolve_runtime_provider(), bare-slug models in config
(e.g. ``claude-opus-4-6`` with ``model.provider: anthropic``) leave
provider/base_url/api_key empty in AIAgent, causing HTTP 404.
"""

from unittest.mock import MagicMock, patch


def test_make_agent_passes_resolved_provider():
    """_make_agent forwards provider/base_url/api_key/api_mode from
    resolve_runtime_provider to AIAgent."""

    fake_runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test-key",
        "api_mode": "anthropic_messages",
        "command": None,
        "args": None,
        "credential_pool": None,
    }

    fake_cfg = {
        "model": {"default": "claude-opus-4-6", "provider": "anthropic"},
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
            return_value=fake_runtime,
        ) as mock_resolve,
        patch("run_agent.AIAgent") as mock_agent,
    ):

        from tui_gateway.server import _make_agent

        _make_agent("sid-1", "key-1")

        # target_model comes from _resolve_startup_runtime() which reads
        # _load_cfg().  Due to module-level caching in tui_gateway.server,
        # the patched config may not take effect when the module was already
        # imported by an earlier test.  Assert the stable part of the call.
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs.get("requested") is None

        call_kwargs = mock_agent.call_args
        assert call_kwargs.kwargs["provider"] == "anthropic"
        assert call_kwargs.kwargs["base_url"] == "https://api.anthropic.com"
        assert call_kwargs.kwargs["api_key"] == "sk-test-key"
        assert call_kwargs.kwargs["api_mode"] == "anthropic_messages"


def test_make_agent_ignores_display_personality_without_system_prompt():
    """The TUI matches the classic CLI: personality only becomes active once
    it has been saved to agent.system_prompt."""

    fake_runtime = {
        "provider": "openrouter",
        "base_url": "https://api.synthetic.new/v1",
        "api_key": "sk-test",
        "api_mode": "chat_completions",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    fake_cfg = {
        "agent": {
            "system_prompt": "",
            "personalities": {"kawaii": "sparkle system prompt"},
        },
        "display": {"personality": "kawaii"},
        "model": {"default": "glm-5"},
    }

    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-default-personality", "key-default-personality")

        assert mock_agent.call_args.kwargs["ephemeral_system_prompt"] is None


def test_probe_config_health_flags_null_sections():
    """Bare YAML keys (`agent:` with no value) parse as None and silently
    drop nested settings; probe must surface them so users can fix."""
    from tui_gateway.server import _probe_config_health

    assert _probe_config_health({"agent": {"x": 1}}) == ""
    assert _probe_config_health({}) == ""

    msg = _probe_config_health({"agent": None, "display": None, "model": {}})
    assert "agent" in msg and "display" in msg
    assert "model" not in msg


def test_probe_config_health_flags_null_personalities_with_active_personality():
    from tui_gateway.server import _probe_config_health

    msg = _probe_config_health(
        {
            "agent": {"personalities": None},
            "display": {"personality": "kawaii"},
            "model": {},
        }
    )
    assert "display.personality" in msg
    assert "agent.personalities" in msg


def test_make_agent_tolerates_null_config_sections():
    """Bare `agent:` / `display:` keys in ~/.hermes/config.yaml parse as
    None. cfg.get("agent", {}) returns None (default only fires on missing
    key), so downstream .get() chains must be guarded. Reported via Twitter
    against the new TUI."""

    fake_runtime = {
        "provider": "openrouter",
        "base_url": "https://api.synthetic.new/v1",
        "api_key": "sk-test",
        "api_mode": "chat_completions",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    null_cfg = {"agent": None, "display": None, "model": {"default": "glm-5"}}

    with (
        patch("tui_gateway.server._load_cfg", return_value=null_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):

        from tui_gateway.server import _make_agent

        _make_agent("sid-null", "key-null")

        assert mock_agent.called


def test_make_agent_tolerates_null_personalities_with_active_personality():
    fake_runtime = {
        "provider": "openrouter",
        "base_url": "https://api.synthetic.new/v1",
        "api_key": "sk-test",
        "api_mode": "chat_completions",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    cfg = {
        "agent": {"personalities": None},
        "display": {"personality": "kawaii"},
        "model": {"default": "glm-5"},
    }

    with (
        patch("tui_gateway.server._load_cfg", return_value=cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("cli.load_cli_config", return_value={"agent": {"personalities": None}}),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-null-personality", "key-null-personality")

        assert mock_agent.called
        assert mock_agent.call_args.kwargs["ephemeral_system_prompt"] is None
