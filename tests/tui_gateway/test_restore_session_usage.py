"""Tests for _restore_session_usage — token counter restore on session.resume.

When the TUI gateway resumes a session, a fresh agent is built whose
session_*_tokens counters all start at zero.  _restore_session_usage copies
the cumulative counts from the stored session row so that session.info /
session.usage reflects actual usage instead of 0/1.0M-0%.
"""

import pytest


class _FakeAgent:
    """Minimal stand-in with the same counter attributes as AIAgent."""

    def __init__(self):
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"


def _import_restore():
    from tui_gateway.server import _restore_session_usage
    return _restore_session_usage


# ── Unit tests for _restore_session_usage ──────────────────────────


def test_restore_basic_token_counts():
    """All five token fields are populated from the stored session row."""
    restore = _import_restore()
    agent = _FakeAgent()
    stored = {
        "input_tokens": 12000,
        "output_tokens": 3500,
        "cache_read_tokens": 800,
        "cache_write_tokens": 200,
        "reasoning_tokens": 1500,
        "api_call_count": 7,
    }
    restore(agent, stored)

    assert agent.session_input_tokens == 12000
    assert agent.session_output_tokens == 3500
    assert agent.session_cache_read_tokens == 800
    assert agent.session_cache_write_tokens == 200
    assert agent.session_reasoning_tokens == 1500
    assert agent.session_total_tokens == 12000 + 3500 + 800 + 200 + 1500
    assert agent.session_api_calls == 7


def test_restore_total_is_sum():
    """session_total_tokens equals the sum of all five token components."""
    restore = _import_restore()
    agent = _FakeAgent()
    stored = {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_tokens": 300,
        "cache_write_tokens": 400,
        "reasoning_tokens": 500,
    }
    restore(agent, stored)
    assert agent.session_total_tokens == 100 + 200 + 300 + 400 + 500


def test_restore_handles_none_values():
    """Missing or None fields default to zero."""
    restore = _import_restore()
    agent = _FakeAgent()
    stored = {
        "input_tokens": None,
        "output_tokens": 50,
    }
    restore(agent, stored)

    assert agent.session_input_tokens == 0
    assert agent.session_output_tokens == 50
    assert agent.session_cache_read_tokens == 0
    assert agent.session_cache_write_tokens == 0
    assert agent.session_reasoning_tokens == 0
    assert agent.session_total_tokens == 50
    assert agent.session_api_calls == 0


def test_restore_preserves_cost_estimate():
    """estimated_cost_usd and cost_status are carried over when present."""
    restore = _import_restore()
    agent = _FakeAgent()
    stored = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "estimated_cost_usd": 0.042,
        "cost_status": "estimated",
    }
    restore(agent, stored)

    assert agent.session_estimated_cost_usd == pytest.approx(0.042)
    assert agent.session_cost_status == "estimated"


def test_restore_skips_cost_when_absent():
    """Cost fields stay at defaults when not in the stored row."""
    restore = _import_restore()
    agent = _FakeAgent()
    stored = {"input_tokens": 100, "output_tokens": 50}
    restore(agent, stored)

    assert agent.session_estimated_cost_usd == 0.0
    assert agent.session_cost_status == "unknown"


def test_restore_empty_dict():
    """An empty stored dict leaves the agent at zero counters."""
    restore = _import_restore()
    agent = _FakeAgent()
    restore(agent, {})

    assert agent.session_input_tokens == 0
    assert agent.session_output_tokens == 0
    assert agent.session_total_tokens == 0
    assert agent.session_api_calls == 0


# ── _get_usage picks up restored counters ──────────────────────────


def test_get_usage_reflects_restored_counters():
    """After restore, _get_usage returns the stored token counts."""
    from tui_gateway.server import _get_usage

    agent = _FakeAgent()
    restore = _import_restore()
    stored = {
        "input_tokens": 5000,
        "output_tokens": 2000,
        "cache_read_tokens": 100,
        "cache_write_tokens": 50,
        "reasoning_tokens": 300,
        "api_call_count": 3,
        "estimated_cost_usd": 0.015,
        "cost_status": "estimated",
    }
    restore(agent, stored)

    usage = _get_usage(agent)

    assert usage["input"] == 5000
    assert usage["output"] == 2000
    assert usage["cache_read"] == 100
    assert usage["cache_write"] == 50
    assert usage["reasoning"] == 300
    assert usage["total"] == 5000 + 2000 + 100 + 50 + 300
    assert usage["calls"] == 3
