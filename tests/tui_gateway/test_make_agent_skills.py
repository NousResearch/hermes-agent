"""Tests for _make_agent handling of unknown preloaded skills."""

import logging
from unittest.mock import patch

import pytest


def test_make_agent_warns_for_unknown_skill(caplog):
    """_make_agent logs warning instead of crashing on unknown skill."""
    caplog.set_level(logging.WARNING)

    with patch("tui_gateway.server._parse_tui_skills_env", return_value=["valid-skill", "oops-typo"]), patch(
        "tui_gateway.server._resolve_startup_runtime", return_value=("test-model", "test-provider"),
    ), patch("tui_gateway.server._load_cfg", return_value={}), patch(
        "agent.skill_commands.build_preloaded_skills_prompt",
        return_value=("", ["valid-skill"], ["oops-typo"]),
    ), patch("run_agent.AIAgent"), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}
    ):
        from tui_gateway.server import _make_agent

        agent = _make_agent("sid", "key", "session-123")

        assert agent is not None
        assert "Unknown skill(s) requested, skipping: oops-typo" in caplog.text


def test_make_agent_handles_all_skills_unknown(caplog):
    """_make_agent starts properly when ALL skills are unknown (edge case)."""
    caplog.set_level(logging.WARNING)

    with patch("tui_gateway.server._parse_tui_skills_env", return_value=["typo-one", "typo-two"]), patch(
        "tui_gateway.server._resolve_startup_runtime", return_value=("test-model", "test-provider"),
    ), patch("tui_gateway.server._load_cfg", return_value={}), patch(
        "agent.skill_commands.build_preloaded_skills_prompt",
        return_value=("", [], ["typo-one", "typo-two"]),
    ), patch("run_agent.AIAgent"), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}
    ):
        from tui_gateway.server import _make_agent

        agent = _make_agent("sid", "key", "session-456")

        assert agent is not None
        assert "typo-one" in caplog.text
        assert "typo-two" in caplog.text

