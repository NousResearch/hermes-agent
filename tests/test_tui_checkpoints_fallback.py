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
