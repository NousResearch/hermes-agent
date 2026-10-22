"""Session spend on the usage payload (``server._get_usage``).

Split from ``test_tui_gateway_server.py``: that module is over the code-health line cap, so new
behaviour's tests go in their own topical file.
"""
from __future__ import annotations

from tui_gateway import server


def test_reports_session_spend_with_its_status():
    """cost_usd/cost_status ride the same payload as the context/cache meters. The status is what
    lets the renderer label an estimate as one (``fd2a35b169`` withdrew the unlabeled figure)."""

    class _CostAgent:
        model = "x"
        session_api_calls = 3
        session_estimated_cost_usd = 1.2345678
        session_cost_status = "estimated"

    usage = server._get_usage(_CostAgent())
    assert usage["cost_usd"] == 1.234568
    assert usage["cost_status"] == "estimated"


def test_omits_spend_before_the_first_call():
    """A resumed session is rebuilt with every counter at 0. Emitting that as spend would overwrite
    the renderer's persisted-row figure with a $0 that reads as "free" instead of "not measured"."""

    class _ResumedAgent:
        model = "x"
        session_api_calls = 0
        session_estimated_cost_usd = 0.0
        session_cost_status = "unknown"

    usage = server._get_usage(_ResumedAgent())
    assert "cost_usd" not in usage
    assert "cost_status" not in usage
