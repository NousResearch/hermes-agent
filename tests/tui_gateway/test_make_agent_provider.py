"""Regression test for #11884: _make_agent must resolve runtime provider.

Without resolve_runtime_provider(), bare-slug models in config
(e.g. ``claude-opus-4-6`` with ``model.provider: anthropic``) leave
provider/base_url/api_key empty in AIAgent, causing HTTP 404.
"""

import os
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
        assert mock_resolve.call_args.kwargs.get("requested") == "anthropic"

        call_kwargs = mock_agent.call_args
        assert call_kwargs.kwargs["provider"] == "anthropic"
        assert call_kwargs.kwargs["base_url"] == "https://api.anthropic.com"
        assert call_kwargs.kwargs["api_key"] == "sk-test-key"
        assert call_kwargs.kwargs["api_mode"] == "anthropic_messages"


def test_resolve_model_prefers_config_without_explicit_tui_provider():
    fake_cfg = {
        "model": {
            "default": "ollama/gemma4:12b-mlx",
            "provider": "ollama",
        }
    }

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
    ):
        os.environ["HERMES_MODEL"] = "deepseek/deepseek-v4-flash"

        from tui_gateway import server

        assert server._resolve_model() == "ollama/gemma4:12b-mlx"


def test_resolve_startup_runtime_prefers_config_provider_without_explicit_tui_provider():
    fake_cfg = {
        "model": {
            "default": "ollama/gemma4:12b-mlx",
            "provider": "ollama",
        }
    }

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
    ):
        os.environ["HERMES_MODEL"] = "deepseek/deepseek-v4-flash"

        from tui_gateway import server

        assert server._resolve_startup_runtime() == (
            "ollama/gemma4:12b-mlx",
            "ollama",
        )


def test_make_agent_forwards_provider_routing():
    """Parity with the messaging gateway + CLI: ``provider_routing`` in
    config.yaml must reach AIAgent so OpenRouter honors the user's sort /
    only / ignore / order / require_parameters / data_collection prefs.

    Regression for the desktop report (LewisDB): Discord respected
    provider_routing but the desktop app (tui_gateway backend) built agents
    with no routing prefs, so OpenRouter selected providers at random.
    """

    fake_runtime = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test",
        "api_mode": "chat_completions",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    fake_cfg = {
        "agent": {"system_prompt": ""},
        "model": {"default": "openrouter/some-model"},
        "provider_routing": {
            "only": ["anthropic", "google"],
            "ignore": ["deepinfra"],
            "order": ["anthropic", "together"],
            "sort": "throughput",
            "require_parameters": True,
            "data_collection": "deny",
        },
    }

    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-pr", "key-pr")

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["providers_allowed"] == ["anthropic", "google"]
        assert kwargs["providers_ignored"] == ["deepinfra"]
        assert kwargs["providers_order"] == ["anthropic", "together"]
        assert kwargs["provider_sort"] == "throughput"
        assert kwargs["provider_require_parameters"] is True
        assert kwargs["provider_data_collection"] == "deny"


def test_make_agent_provider_routing_defaults_when_unset():
    """No ``provider_routing`` section → no routing prefs forwarded (None /
    False), so behavior is unchanged for users who never configured it."""

    fake_runtime = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test",
        "api_mode": "chat_completions",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "glm-5"}}

    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-pr-default", "key-pr-default")

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["providers_allowed"] is None
        assert kwargs["providers_ignored"] is None
        assert kwargs["providers_order"] is None
        assert kwargs["provider_sort"] is None
        assert kwargs["provider_require_parameters"] is False
        assert kwargs["provider_data_collection"] is None


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


def test_make_agent_honors_tui_launch_env_flags():
    fake_runtime = {
        "provider": "openrouter",
        "base_url": "https://api.synthetic.new/v1",
        "api_key": "sk-test",
        "api_mode": "chat_completions",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "glm-5"}}

    with (
        patch.dict(
            os.environ,
            {
                "HERMES_TUI_MAX_TURNS": "7",
                "HERMES_TUI_CHECKPOINTS": "1",
                "HERMES_TUI_PASS_SESSION_ID": "1",
                "HERMES_IGNORE_RULES": "1",
            },
        ),
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-env", "key-env")

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["max_iterations"] == 7
        assert kwargs["checkpoints_enabled"] is True
        assert kwargs["pass_session_id"] is True
        assert kwargs["skip_context_files"] is True
        assert kwargs["skip_memory"] is True


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


def test_make_agent_honors_per_session_model_override():
    """Regression for cross-session model contamination: a per-session
    ``model_override`` (set by an in-session /model switch) must drive the
    rebuilt agent's model/provider/base_url, NOT global config — and without
    reading process-global env vars that a sibling session may have changed.
    """

    # resolve_runtime_provider echoes the requested provider so we can prove
    # the override's provider (not the global default) was passed through.
    def echo_runtime(requested=None, target_model=None):
        return {
            "provider": requested or "GLOBAL_DEFAULT",
            "base_url": "global-url",
            "api_key": "global-key",
            "api_mode": "chat_completions",
            "command": None,
            "args": None,
            "credential_pool": None,
        }

    fake_cfg = {
        "agent": {"system_prompt": ""},
        "model": {"default": "global/model", "provider": "globalprov"},
    }

    override = {
        "model": "zai/glm-5.1",
        "provider": "zai",
        "base_url": "https://api.z.ai/v1",
        "api_key": "sk-glm",
        "api_mode": "chat_completions",
    }

    with (
        # Ensure no leaked env biases _resolve_startup_runtime (it must not even
        # be consulted when an override is present).
        patch.dict(os.environ, {}, clear=False),
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=echo_runtime,
        ),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        for var in (
            "HERMES_MODEL",
            "HERMES_INFERENCE_MODEL",
            "HERMES_TUI_PROVIDER",
            "HERMES_INFERENCE_PROVIDER",
        ):
            os.environ.pop(var, None)

        from tui_gateway.server import _make_agent

        _make_agent(
            "sid-override", "key-override", model_override=override
        )

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["model"] == "zai/glm-5.1"
        assert kwargs["provider"] == "zai"
        # Concrete credentials from the switch survive the rebuild.
        assert kwargs["base_url"] == "https://api.z.ai/v1"
        assert kwargs["api_key"] == "sk-glm"


def test_apply_model_switch_does_not_leak_process_env():
    """Core fix for cross-session contamination: an in-session /model switch
    must mutate only the target session (record a per-session override + switch
    that session's agent in place) and must NOT write process-global env vars,
    which the single-process desktop backend shares across every live session.
    """
    from tui_gateway import server

    class _FakeResult:
        success = True
        error_message = ""
        warning_message = ""
        new_model = "zai/glm-5.1"
        target_provider = "zai"
        base_url = "https://api.z.ai/v1"
        api_key = "sk-glm"
        api_mode = "chat_completions"

    class _FakeAgent:
        def __init__(self):
            self.model = "minimax/m3"
            self.provider = "minimax"
            self.base_url = ""
            self.api_key = ""

        def switch_model(self, **kw):
            self.model = kw["new_model"]
            self.provider = kw["new_provider"]

    env_keys = (
        "HERMES_MODEL",
        "HERMES_INFERENCE_MODEL",
        "HERMES_TUI_PROVIDER",
        "HERMES_INFERENCE_PROVIDER",
    )

    sess_b = {"agent": _FakeAgent(), "session_key": "k-B", "model_override": None}
    sess_a = {"agent": _FakeAgent(), "session_key": "k-A", "model_override": None}

    with (
        patch("hermes_cli.model_switch.parse_model_flags",
              return_value=("glm-5.1", None, False, False, True)),
        patch("hermes_cli.model_switch.resolve_persist_behavior",
              return_value=False),
        patch("hermes_cli.model_switch.switch_model", return_value=_FakeResult()),
        patch("tui_gateway.server._emit"),
        patch("tui_gateway.server._restart_slash_worker"),
        patch("tui_gateway.server._session_info", return_value={}),
        patch("tui_gateway.server._persist_model_switch") as mock_persist,
    ):
        before = {k: os.environ.get(k) for k in env_keys}
        result = server._apply_model_switch("sidB", sess_b, "glm-5.1")
        after = {k: os.environ.get(k) for k in env_keys}

    assert result["value"] == "zai/glm-5.1"
    # No process-global env mutation (the contamination vector).
    assert before == after
    # persist_global was False → config untouched.
    mock_persist.assert_not_called()
    # Target session recorded a per-session override.
    assert sess_b["model_override"]["model"] == "zai/glm-5.1"
    assert sess_b["model_override"]["provider"] == "zai"
    # The switched agent mutated in place.
    assert sess_b["agent"].model == "zai/glm-5.1"
    # Sibling session is completely untouched.
    assert sess_a["model_override"] is None
    assert sess_a["agent"].model == "minimax/m3"
