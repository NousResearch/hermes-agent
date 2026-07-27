"""Regression coverage for live TUI image-routing identity."""

from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway.server import _active_image_routing_identity


def test_live_agent_identity_wins_after_model_switch():
    agent = SimpleNamespace(provider="alibaba", model="qwen3.7-plus")

    with patch(
        "agent.auxiliary_client._read_main_provider",
        side_effect=AssertionError("stale provider fallback used"),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        side_effect=AssertionError("stale model fallback used"),
    ):
        identity = _active_image_routing_identity(agent)

    assert identity == ("alibaba", "qwen3.7-plus")


def test_missing_agent_identity_uses_runtime_fallback():
    agent = SimpleNamespace(provider="", model=None)

    with patch(
        "agent.auxiliary_client._read_main_provider", return_value="openai-codex"
    ), patch(
        "agent.auxiliary_client._read_main_model", return_value="gpt-5.5-codex"
    ):
        identity = _active_image_routing_identity(agent)

    assert identity == ("openai-codex", "gpt-5.5-codex")
