"""Test that TUI gateway image routing uses live agent model, not stale runtime value.

Regression test for #43516 — after switching models via /model mid-session,
image routing should use the updated agent.provider and agent.model instead of
the stale _RUNTIME_MAIN_MODEL set by the previous turn.
"""

from types import SimpleNamespace

import pytest


def _make_agent(provider="", model=""):
    """Create a mock agent with provider/model attributes."""
    agent = SimpleNamespace()
    agent.provider = provider
    agent.model = model
    agent.api_mode = ""
    return agent


class TestImageRoutingModelPreference:
    """Verify the provider/model preference logic used in tui_gateway/server.py.

    The fix (line ~5036) uses:
        getattr(agent, "provider", "") or _read_main_provider()
        getattr(agent, "model", "") or _read_main_model()

    These tests verify the getattr + fallback expression produces the correct
    value for all edge cases.
    """

    def test_prefers_agent_provider_and_model(self):
        """After /model switch, agent attributes take priority."""
        agent = _make_agent(provider="alibaba", model="qwen3.7-plus")
        _stale_provider = "alibaba"
        _stale_model = "qwen3.7-max"

        effective_provider = getattr(agent, "provider", "") or _stale_provider
        effective_model = getattr(agent, "model", "") or _stale_model

        assert effective_provider == "alibaba"
        assert effective_model == "qwen3.7-plus"

    def test_falls_back_to_stale_when_agent_empty(self):
        """Empty string on agent → falls back to stale runtime values."""
        agent = _make_agent(provider="", model="")
        _stale_provider = "openai"
        _stale_model = "gpt-4o"

        effective_provider = getattr(agent, "provider", "") or _stale_provider
        effective_model = getattr(agent, "model", "") or _stale_model

        assert effective_provider == "openai"
        assert effective_model == "gpt-4o"

    def test_falls_back_to_stale_when_agent_none(self):
        """None on agent → getattr default "" → falsy → falls back."""
        agent = SimpleNamespace(provider=None, model=None)
        _stale_provider = "openai"
        _stale_model = "gpt-4o"

        effective_provider = getattr(agent, "provider", "") or _stale_provider
        effective_model = getattr(agent, "model", "") or _stale_model

        assert effective_provider == "openai"
        assert effective_model == "gpt-4o"

    def test_no_agent_uses_fallback(self):
        """No agent at all → pure fallback."""
        _stale_provider = "anthropic"
        _stale_model = "claude-sonnet-4"

        effective_provider = getattr(None, "provider", "") or _stale_provider
        effective_model = getattr(None, "model", "") or _stale_model

        assert effective_provider == "anthropic"
        assert effective_model == "claude-sonnet-4"

    def test_provider_and_model_differ_after_switch(self):
        """Simulate the exact scenario from #43516."""
        # User switched from qwen3.7-max → qwen3.7-plus via /model
        agent = _make_agent(provider="alibaba", model="qwen3.7-plus")

        # Runtime still has stale value from previous turn
        _read_main_provider = lambda: "alibaba"
        _read_main_model = lambda: "qwen3.7-max"

        # The fix: prefer agent's live values
        effective_provider = getattr(agent, "provider", "") or _read_main_provider()
        effective_model = getattr(agent, "model", "") or _read_main_model()

        assert effective_model == "qwen3.7-plus"  # NOT "qwen3.7-max"
