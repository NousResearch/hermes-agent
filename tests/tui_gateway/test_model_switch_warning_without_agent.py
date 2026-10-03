"""The switch-cost note on the surface whose session has no resident agent.

``config.set model`` with an explicit ``--provider`` skips the agent build, so the TUI/Desktop
warning lives or dies on the history the session already has — not on ``session["agent"]``.
"""

from __future__ import annotations

import pytest

from hermes_cli.model_switch import ModelSwitchResult


def _result(model: str = "large-model") -> ModelSwitchResult:
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider="openrouter",
        provider_changed=False,
        api_key="k",
        base_url="https://example.com/v1",
        api_mode="chat_completions",
        provider_label="openrouter",
        model_info={"context_length": 200_000},
    )


@pytest.fixture
def tui_switch(monkeypatch):
    """``_apply_model_switch`` as the gateway installs it, with everything but the warning
    assessment doubled at its seams (route discovery, guards, persistence, provider resolution)."""
    import tui_gateway.server as server

    monkeypatch.setattr(
        server, "_current_model_runtime", lambda agent, explicit: ("openrouter", "small-model", "", ""))
    monkeypatch.setattr(server, "_expensive_model_confirm", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.resolve_persist_behavior", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **k: _result())
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length", lambda *a, **k: 200_000)
    return server


def test_explicit_provider_switch_warns_from_history_without_an_agent(tui_switch):
    """A large live session whose agent slot is empty must still be told what the switch costs."""
    session = {
        "agent": None,
        "history": [{"role": "user", "content": "x" * 20_000} for _ in range(30)],
        "follow_profile_config": False,
    }

    out = tui_switch._apply_model_switch("sid", session, "large-model --provider openrouter")

    assert "preflight compression" in out["warning"]


def test_no_history_stays_silent(tui_switch):
    """Without a conversation there is nothing to size: an agentless session is not warned."""
    session = {"agent": None, "history": [], "follow_profile_config": False}

    out = tui_switch._apply_model_switch("sid", session, "large-model --provider openrouter")

    assert out["warning"] == ""


def test_agentless_switch_reports_the_configured_disabled_policy(tui_switch, monkeypatch):
    """F3: an agentless assessment must describe the destination's policy, not an optimistic default.
    With ``compression.enabled`` false the same large session gets no compression notice at all —
    paired with the neighbouring test, which pins the enabled case."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"compression": {"enabled": False}})
    session = {
        "agent": None,
        "follow_profile_config": False,
        "history": [{"role": "user" if i % 2 == 0 else "assistant",
                     "content": "x" * 20_000} for i in range(30)],
    }

    out = tui_switch._apply_model_switch("sid", session, "large-model --provider openrouter")

    assert session["model_override"]["model"] == "large-model"
    assert "preflight compression" not in out["warning"]
    # The switch still costs a re-read, so the cost note survives the disabled pass.
    assert "no warm prefix cache" in out["warning"]
