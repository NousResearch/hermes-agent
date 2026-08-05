"""Tests for TUI/desktop checkpoint config fallback (#79625).

Desktop sessions (via ``hermes serve`` → TUI gateway) never set
``HERMES_TUI_CHECKPOINTS``, so agent creation gated filesystem checkpoints
on that env var alone — ``checkpoints.enabled: true`` in config.yaml was
silently ignored on the desktop surface while the CLI and messaging gateway
honored it. The fix makes ``_make_agent`` fall back to the ``checkpoints``
config section when the env var is unset, mirroring the messaging gateway's
``_checkpoint_agent_kwargs`` (gateway/run.py).
"""

import os

import pytest

from tui_gateway import server


class TestLoadCheckpointsEnabled:
    def test_defaults_to_false_when_no_config_section(self):
        """No ``checkpoints`` section → DEFAULT_CONFIG default (disabled)."""
        assert server._load_checkpoints_enabled({}) is False

    def test_honors_config_enabled_true(self):
        """``checkpoints: {enabled: true}`` in config.yaml enables checkpoints."""
        assert server._load_checkpoints_enabled({"checkpoints": {"enabled": True}}) is True

    def test_honors_config_enabled_false(self):
        assert server._load_checkpoints_enabled({"checkpoints": {"enabled": False}}) is False

    def test_legacy_bool_form(self):
        """Legacy ``checkpoints: true`` (bare bool) still enables."""
        assert server._load_checkpoints_enabled({"checkpoints": True}) is True
        assert server._load_checkpoints_enabled({"checkpoints": False}) is False

    def test_malformed_section_falls_back_to_default(self):
        assert server._load_checkpoints_enabled({"checkpoints": "enabled"}) is False

    def test_none_cfg_falls_back(self):
        assert server._load_checkpoints_enabled(None) is False


class TestResolveCheckpointsEnabled:
    """The env-var-first precedence used by ``_make_agent``."""

    def test_config_enabled_used_when_env_unset(self, monkeypatch):
        """Desktop path: no HERMES_TUI_CHECKPOINTS → config ``enabled: true`` wins."""
        monkeypatch.delenv("HERMES_TUI_CHECKPOINTS", raising=False)
        assert server._resolve_checkpoints_enabled(
            {"checkpoints": {"enabled": True}}
        ) is True

    def test_config_disabled_used_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_TUI_CHECKPOINTS", raising=False)
        assert server._resolve_checkpoints_enabled(
            {"checkpoints": {"enabled": False}}
        ) is False

    def test_env_var_overrides_config(self, monkeypatch):
        """CLI ``--tui --checkpoints`` (env set) still wins over config."""
        monkeypatch.setenv("HERMES_TUI_CHECKPOINTS", "1")
        assert server._resolve_checkpoints_enabled(
            {"checkpoints": {"enabled": False}}
        ) is True

    def test_env_var_false_overrides_config_true(self, monkeypatch):
        monkeypatch.setenv("HERMES_TUI_CHECKPOINTS", "0")
        assert server._resolve_checkpoints_enabled(
            {"checkpoints": {"enabled": True}}
        ) is False


class TestMakeAgentCheckpointsFallback:
    """The env-var → config fallback inside ``_make_agent`` (the wiring).

    Addresses review: unit tests of the helper alone could false-pass if
    ``_make_agent`` never actually passed the value through. These drive
    ``_make_agent`` end-to-end with a fake ``AIAgent`` that captures
    kwargs, proving the config-derived value reaches the agent.
    """

    def _make_agent_with_checkpoints(self, monkeypatch, cfg, env_val="__unset__"):
        import types

        captured = {}

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(model=kwargs.get("model"))

        if env_val == "__unset__":
            monkeypatch.delenv("HERMES_TUI_CHECKPOINTS", raising=False)
        else:
            monkeypatch.setenv("HERMES_TUI_CHECKPOINTS", env_val)
        monkeypatch.delenv("HERMES_MODEL", raising=False)
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
        monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
        monkeypatch.setattr(
            server, "_load_cfg",
            lambda: {**cfg, "model": {"default": "gpt-5.5", "provider": "openai-codex"}},
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None, target_model=None: {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "token",
                "api_mode": "codex_responses",
                "credential_pool": None,
            },
        )
        monkeypatch.setattr("run_agent.AIAgent", fake_agent)
        monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: ["file"])
        monkeypatch.setattr(server, "_get_db", lambda: None)
        server._make_agent("sid", "session-key")
        return captured

    def test_desktop_config_enabled_reaches_agent(self, monkeypatch):
        """Issue #79625: config ``enabled: true`` (env unset) → agent has
        checkpoints_enabled=True. On the old code this was always False."""
        captured = self._make_agent_with_checkpoints(
            monkeypatch, {"checkpoints": {"enabled": True}}
        )
        assert captured.get("checkpoints_enabled") is True

    def test_desktop_config_disabled_reaches_agent(self, monkeypatch):
        captured = self._make_agent_with_checkpoints(
            monkeypatch, {"checkpoints": {"enabled": False}}
        )
        assert captured.get("checkpoints_enabled") is False

    def test_cli_env_override_still_wins(self, monkeypatch):
        """Legacy ``--tui --checkpoints`` (env '1') overrides config False."""
        captured = self._make_agent_with_checkpoints(
            monkeypatch, {"checkpoints": {"enabled": False}}, env_val="1"
        )
        assert captured.get("checkpoints_enabled") is True

    def test_cli_env_off_still_wins(self, monkeypatch):
        captured = self._make_agent_with_checkpoints(
            monkeypatch, {"checkpoints": {"enabled": True}}, env_val="0"
        )
        assert captured.get("checkpoints_enabled") is False
