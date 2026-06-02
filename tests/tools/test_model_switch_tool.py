"""Tests for the agent-callable ``model_switch`` tool (upstream #16525).

These tests use a stub agent (no real AIAgent / network) and mock the shared
``hermes_cli.model_switch.switch_model`` resolver so they stay hermetic. They
cover: session-scope switching, turn-scope revert, graceful handling of an
unknown model slug, the optional allowlist safeguard, and the opt-in config
gate.
"""

import json
from types import SimpleNamespace

import pytest

from tools.model_switch_tool import (
    TURN_REVERT_ATTR,
    check_model_switch_requirements,
    model_switch_tool,
    revert_turn_model_switch,
)


class _StubAgent:
    """Minimal stand-in for AIAgent that records switch_model() calls.

    Why: model_switch_tool mutates a live agent; a recording stub lets us
    assert the swap was applied without booting a real agent.
    """

    def __init__(self):
        self.model = "openai/gpt-4o-mini"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-old"
        self.api_mode = "openai"
        self.switch_calls = []

    def switch_model(self, new_model, new_provider, api_key="", base_url="", api_mode=""):
        self.switch_calls.append({
            "new_model": new_model,
            "new_provider": new_provider,
            "api_key": api_key,
            "base_url": base_url,
            "api_mode": api_mode,
        })
        # Mirror the real in-place swap so subsequent reads see new values.
        self.model = new_model
        self.provider = new_provider
        if base_url:
            self.base_url = base_url
        if api_key:
            self.api_key = api_key
        self.api_mode = api_mode


def _ok_result(new_model="deepseek/deepseek-r1", provider="openrouter"):
    """Build a successful ModelSwitchResult-shaped object."""
    return SimpleNamespace(
        success=True,
        new_model=new_model,
        target_provider=provider,
        api_key="sk-new",
        base_url="https://openrouter.ai/api/v1",
        api_mode="openai",
        error_message="",
    )


def _err_result(message="Invalid model"):
    return SimpleNamespace(
        success=False,
        new_model="",
        target_provider="",
        api_key="",
        base_url="",
        api_mode="",
        error_message=message,
    )


@pytest.fixture(autouse=True)
def _no_config(monkeypatch):  # pyright: ignore[reportUnusedFunction]
    """Default: empty config so providers/allowlist/gate read as unset."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})


def test_session_scope_switch_returns_dict(monkeypatch):
    """Basic session switch applies the swap and returns the contract dict."""
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kw: _ok_result("deepseek/deepseek-r1"),
    )
    agent = _StubAgent()

    raw = model_switch_tool(agent, slug="deepseek/deepseek-r1",
                            reason="Need deep reasoning", scope="session")
    out = json.loads(raw)

    assert out == {
        "old_model": "openai/gpt-4o-mini",
        "new_model": "deepseek/deepseek-r1",
        "scope": "session",
        "applied_at": "next_turn",
    }
    assert len(agent.switch_calls) == 1
    assert agent.switch_calls[0]["new_model"] == "deepseek/deepseek-r1"
    # Session scope leaves no pending turn revert.
    assert getattr(agent, TURN_REVERT_ATTR, None) is None


def test_turn_scope_reverts_after_one_turn(monkeypatch):
    """Turn-scope switch snapshots the old model and reverts on next turn."""
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kw: _ok_result("deepseek/deepseek-r1"),
    )
    agent = _StubAgent()

    raw = model_switch_tool(agent, slug="deepseek/deepseek-r1",
                            reason="One-shot escalation", scope="turn")
    out = json.loads(raw)
    assert out["scope"] == "turn"
    assert agent.model == "deepseek/deepseek-r1"
    # A revert snapshot of the original runtime is pending.
    snapshot = getattr(agent, TURN_REVERT_ATTR)
    assert snapshot["model"] == "openai/gpt-4o-mini"

    # The loop calls this at the start of the next turn.
    reverted = revert_turn_model_switch(agent)
    assert reverted is True
    assert agent.model == "openai/gpt-4o-mini"
    assert getattr(agent, TURN_REVERT_ATTR) is None
    # Second revert is a no-op (idempotent).
    assert revert_turn_model_switch(agent) is False


def test_unknown_model_slug_handled_gracefully(monkeypatch):
    """A resolver failure returns an error string, never raises."""
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kw: _err_result("Unknown model 'totally-fake'"),
    )
    agent = _StubAgent()

    raw = model_switch_tool(agent, slug="totally-fake", reason="oops")
    out = json.loads(raw)

    assert "error" in out
    assert "totally-fake" in out["error"]
    # Nothing was applied.
    assert agent.switch_calls == []
    assert agent.model == "openai/gpt-4o-mini"


def test_empty_slug_rejected():
    """An empty slug short-circuits before any resolution."""
    agent = _StubAgent()
    out = json.loads(model_switch_tool(agent, slug="   ", reason="x"))
    assert "error" in out
    assert agent.switch_calls == []


def test_allowlist_blocks_disallowed_model(monkeypatch):
    """model_switch_allowlist restricts which slugs are permitted."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model_switch_allowlist": ["gpt-4o", "sonnet"]},
    )
    # switch_model must never be reached for a blocked slug.
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kw: pytest.fail("switch_model should not run for blocked slug"),
    )
    agent = _StubAgent()

    out = json.loads(model_switch_tool(agent, slug="deepseek/deepseek-r1",
                                       reason="blocked"))
    assert "error" in out
    assert "allowlist" in out["error"].lower()
    assert agent.switch_calls == []


def test_allowlist_allows_listed_model(monkeypatch):
    """A slug present in the allowlist passes through (case-insensitive)."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model_switch_allowlist": ["GPT-4O"]},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kw: _ok_result("gpt-4o"),
    )
    agent = _StubAgent()

    out = json.loads(model_switch_tool(agent, slug="gpt-4o", reason="ok"))
    assert out["new_model"] == "gpt-4o"
    assert len(agent.switch_calls) == 1


def test_check_requirements_gates_on_config(monkeypatch):
    """The tool is opt-in via agent.allow_self_model_switch."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert check_model_switch_requirements() is False

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"agent": {"allow_self_model_switch": True}},
    )
    assert check_model_switch_requirements() is True

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"agent": {"allow_self_model_switch": False}},
    )
    assert check_model_switch_requirements() is False


def test_apply_failure_clears_turn_snapshot(monkeypatch):
    """If switch_model raises, a turn snapshot must not be left stranded."""
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kw: _ok_result("deepseek/deepseek-r1"),
    )

    class _BoomAgent(_StubAgent):
        def switch_model(self, *a, **k):
            raise RuntimeError("bad credentials")

    agent = _BoomAgent()
    out = json.loads(model_switch_tool(agent, slug="deepseek/deepseek-r1",
                                       reason="will fail", scope="turn"))
    assert "error" in out
    assert getattr(agent, TURN_REVERT_ATTR, None) is None
