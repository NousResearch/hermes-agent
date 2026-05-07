"""Tests for AIAgent._user_status_turn_context (cross-bot user-status injection).

Task C / issue #21122 / kanban t_315c0bfc — verifies the per-turn ephemeral
helper that formats agent.user_status state into a short line for injection
alongside ``_plugin_turn_context`` in ``run_conversation()``.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and reload agent.user_status."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    if hasattr(hermes_constants, "_profile_fallback_warned"):
        hermes_constants._profile_fallback_warned = False
    from agent import user_status as us
    importlib.reload(us)
    return tmp_path, us


@pytest.fixture
def agent_stub():
    """Minimal stub exposing the bound helper from AIAgent.

    We don't want to construct a full AIAgent (60+ ctor params, network deps).
    The helper only reads ``agent.user_status`` and a module-level ``logger``,
    so binding the unbound method to a SimpleNamespace is sufficient.
    """
    from run_agent import AIAgent
    stub = SimpleNamespace()
    stub._user_status_turn_context = AIAgent._user_status_turn_context.__get__(stub)
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_file_returns_empty_string(hermes_home, agent_stub):
    """No user_status.json on disk → helper returns ''."""
    _, _us = hermes_home
    assert agent_stub._user_status_turn_context() == ""


def test_phone_mode_emits_expected_line(hermes_home, agent_stub):
    """Fresh device_mode='phone' → hard claim line with ≤100 word guidance."""
    _, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    out = agent_stub._user_status_turn_context()
    assert out
    assert "Device: phone" in out
    assert "\u2264100 words" in out  # ≤100 words
    assert "deep-dives go to file" in out
    # Hard claim, not a soft hint.
    assert "may be" not in out
    assert "stale" not in out


def test_stale_field_is_soft_hint_not_hard_claim(hermes_home, agent_stub):
    """A device_mode older than 12h → softened wording, not a hard claim."""
    _, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")

    # Backdate the per_field_updated_at timestamp past the 12h staleness.
    state_file = hermes_home[0] / "state" / "user_status.json"
    import json
    data = json.loads(state_file.read_text())
    backdated = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    data["per_field_updated_at"]["device_mode"] = backdated
    state_file.write_text(json.dumps(data))

    out = agent_stub._user_status_turn_context()
    assert out
    # Soft hint markers — must not be a definitive "Device: phone —" claim.
    assert "stale" in out.lower()
    assert "may be" in out.lower() or "possibly" in out.lower()
    # Must NOT contain the hard-claim sentence from the fresh case.
    assert "Device: phone \u2014 keep replies" not in out


def test_multiple_fields_compose_correctly(hermes_home, agent_stub):
    """device_mode + focus_project both fresh → both lines appear, order stable."""
    _, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    us.save_field("focus_project", "SpainExpat migration", writer="discord")

    out = agent_stub._user_status_turn_context()
    assert "Device: phone" in out
    assert "Focus: SpainExpat migration" in out
    # Device should precede focus per the helper's ordering.
    assert out.index("Device:") < out.index("Focus:")
    # Neither should be marked stale.
    assert "stale" not in out.lower()


def test_quiet_hours_in_future_emits_line(hermes_home, agent_stub):
    """quiet_hours_until in the future → injected; uses its own timestamp."""
    _, us = hermes_home
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    us.save_field("quiet_hours_until", future, writer="telegram")
    out = agent_stub._user_status_turn_context()
    assert "Quiet hours until" in out
    assert "minimize proactive notifications" in out


def test_quiet_hours_expired_omitted(hermes_home, agent_stub):
    """quiet_hours_until in the past → omitted (its own timestamp gates it)."""
    _, us = hermes_home
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    us.save_field("quiet_hours_until", past, writer="telegram")
    out = agent_stub._user_status_turn_context()
    assert "Quiet hours" not in out
