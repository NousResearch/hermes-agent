"""B3 per-child usage telemetry: manifest token fields and summary surfacing.

Three surfaces, one data flow:
- conversation loop maintains authoritative session counters per response
  (run_agent.py session_prompt_tokens / session_completion_tokens);
- AIAgent.get_activity_summary surfaces them as prompt_tokens /
  completion_tokens in the activity snapshot;
- delegate_tool.py manifests record total_prompt_tokens /
  total_completion_tokens / tokens_per_sec for every child entry (success
  and error/timeout paths alike).
"""

import pytest

from tools.delegate_tool import _tokens_per_sec


# --------------------------------------------------------------------- #
# _tokens_per_sec
# --------------------------------------------------------------------- #

def test_tokens_per_sec_basic_math():
    """120 completion tokens over 60 active seconds -> 2.0 t/s."""
    assert _tokens_per_sec(120, 60) == 2.0


def test_tokens_per_sec_zero_duration():
    """Zero / negative / None duration must never divide by zero."""
    assert _tokens_per_sec(120, 0) == 0.0
    assert _tokens_per_sec(120, -5) == 0.0
    assert _tokens_per_sec(120, None) == 0.0


def test_tokens_per_sec_defensive_none_and_garbage():
    """None / non-numeric tokens or duration coerce to 0.0, no exception."""
    assert _tokens_per_sec(None, 60) == 0.0
    assert _tokens_per_sec("120", 60) == 2.0  # numeric strings coerce
    assert _tokens_per_sec("garbage", 60) == 0.0
    assert _tokens_per_sec(120, "garbage") == 0.0


def test_tokens_per_sec_rounding():
    """Rounds to 2 decimal places."""
    assert _tokens_per_sec(100, 3) == 33.33


# --------------------------------------------------------------------- #
# get_activity_summary surfacing
# --------------------------------------------------------------------- #

def _bare_summary_subject():
    """Minimal object exposing only what get_activity_summary touches."""
    from run_agent import AIAgent

    class _Bare:
        # get_activity_summary is a plain method on AIAgent; bind it to a
        # bare object via the unbound function so we don't pay full agent
        # construction in a unit test.
        get_activity_summary = AIAgent.get_activity_summary

        _last_activity_provenance = None
        _last_activity_ts = None
        _last_activity_desc = None
        _current_tool = None
        _api_call_count = 3
        session_prompt_tokens = 1234
        session_completion_tokens = 567
        max_iterations = 25
        session_api_calls = 3
        session_total_tokens = 1801

        class iteration_budget:
            used = 7
            max_total = 25

    return _Bare()


def test_activity_summary_surfaces_session_token_counters():
    """prompt_tokens / completion_tokens read the authoritative session
    counters, so delegation manifests get per-child token totals from the
    same accounting the success path uses."""
    subject = _bare_summary_subject()
    snap = subject.get_activity_summary()
    assert snap["prompt_tokens"] == 1234
    assert snap["completion_tokens"] == 567
    assert snap["api_call_count"] == 3


def test_activity_summary_defaults_zero_when_uninitialized():
    """Partially initialized agents (no session counters yet) report 0s,
    not AttributeErrors."""
    from run_agent import AIAgent

    class _Bare:
        get_activity_summary = AIAgent.get_activity_summary

        _last_activity_provenance = None
        _last_activity_ts = None
        _last_activity_desc = None
        _current_tool = None
        _api_call_count = 0
        max_iterations = 25

        class iteration_budget:
            used = 0
            max_total = 25

    snap = _Bare().get_activity_summary()
    assert snap["prompt_tokens"] == 0
    assert snap["completion_tokens"] == 0
