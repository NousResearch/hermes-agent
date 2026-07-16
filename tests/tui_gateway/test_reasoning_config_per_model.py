"""Tests for per-model reasoning_effort override in TUI gateway _load_reasoning_config."""

import pytest

import tui_gateway.server as tui_server


class TestTUIPerModelReasoningConfig:
    """Test tui_gateway _load_reasoning_config respects per-model overrides."""

    def test_per_model_override_takes_precedence(self, monkeypatch):
        """Per-model override wins over global reasoning_effort."""
        fake_cfg = {
            "model": {"default": "anthropic/claude-opus-4.5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {
                    "anthropic/claude-opus-4.5": "xhigh",
                },
            },
        }
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: fake_cfg)

        result = tui_server._load_reasoning_config()
        assert result is not None
        assert result["enabled"] is True
        assert result["effort"] == "xhigh"

    def test_global_fallback_when_no_override(self, monkeypatch):
        """Global reasoning_effort applies when no per-model override matches."""
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {
                    "anthropic/claude-opus-4.5": "xhigh",
                },
            },
        }
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: fake_cfg)

        result = tui_server._load_reasoning_config()
        assert result is not None
        assert result["effort"] == "high"

    def test_spelling_tolerant_match(self, monkeypatch):
        """Override matches even with different spelling (provider prefix)."""
        fake_cfg = {
            "model": {"default": "claude-opus-4.5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {
                    "anthropic/claude-opus-4.5": "high",  # key has prefix, model doesn't
                },
            },
        }
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: fake_cfg)

        result = tui_server._load_reasoning_config()
        assert result is not None
        assert result["effort"] == "high"

    def test_parity_with_gateway_loader(self, monkeypatch):
        """TUI and gateway loaders return identical results for same config."""
        import gateway.run as gateway_run

        fake_cfg = {
            "model": {"default": "openrouter/anthropic/claude-sonnet-4.6"},
            "agent": {
                "reasoning_effort": "low",
                "reasoning_overrides": {
                    "claude-sonnet-4.6": "high",
                },
            },
        }
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: fake_cfg)
        monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: fake_cfg)

        tui_result = tui_server._load_reasoning_config()
        gw_result = gateway_run.GatewayRunner._load_reasoning_config()
        assert tui_result == gw_result

    def test_global_fallback_with_yaml_false(self, monkeypatch):
        """YAML boolean False must reach parse_reasoning_effort uncoerced.

        Regression: str(... or "").strip() turned False into "", silently
        re-enabling thinking. The raw value must pass through so
        parse_reasoning_effort(False) returns {'enabled': False}.
        """
        fake_cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": False,  # YAML boolean, not string
            },
        }
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: fake_cfg)

        result = tui_server._load_reasoning_config()
        assert result is not None
        assert result.get("enabled") is False
